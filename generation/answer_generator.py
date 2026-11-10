"""End-to-end answering pipeline.

    retrieve -> rerank -> sufficiency gate (abstain) -> LLM -> JSON validation ->
    citation verification (+ verifier retry loop) -> computed confidence

Three things can end in "INSUFFICIENT EVIDENCE": the gate (weak evidence, the LLM is never called),
the model itself (it says the passages do not answer), and the verifier (no citation of the draft
survived the checks). Nothing the model says is shown as grounded unless a citation was verified.
"""
from __future__ import annotations

import json
import re
import time
from typing import Callable

from pydantic import ValidationError

from config import Settings, get_settings
from generation import prompts
from generation.ollama_client import OllamaClient, OllamaError
from retrieval import hybrid
from schemas import AnswerResult, Conflict, LLMAnswer, VerifiedClaim
from services import db, indexer
from textutil import clause_key
from verification import abstention, citation_validator, conflict_detector, evidence_checker
from verification.citation_validator import CaseDoc
from verification.evidence_checker import Sufficiency


class StaleIndexError(RuntimeError):
    """The case was indexed by an older version of the code, so its documents carry no metadata for the verifier."""


class AnswerFormatError(RuntimeError):
    """The model did not return a valid answer object, even after one repair request."""


def parse_llm_json(text: str) -> LLMAnswer:
    """Parse and validate the model's reply; tolerate ```json fences and prose around the object."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        data = json.loads(text)
    except ValueError:
        start = text.find("{")
        if start < 0:
            raise ValueError("No JSON object found in model output.") from None
        data, _ = json.JSONDecoder().raw_decode(text[start:])      # first complete object, braces in strings handled
    return LLMAnswer.model_validate(data)


def case_docs(case_id: str) -> list[CaseDoc]:
    """What the verifier needs: each document's name, page count and clause labels."""
    chunks = indexer.case_chunks(case_id)
    docs = []
    for m in indexer.case_documents(case_id):
        labels = {clause_key(x) for x in m["labels"]}
        labels |= {clause_key(c.clause) for c in chunks if c.document_id == m["document_id"] and c.clause}
        docs.append(CaseDoc(m["document_id"], m["filename"], m["page_count"], labels))
    return docs


def compute_confidence(suff: Sufficiency, verified: list[VerifiedClaim], n_hard_conflicts: int,
                       s: Settings) -> tuple[str, list[str]]:
    """HIGH / MEDIUM / LOW computed from verification results and evidence strength, never from the LLM.
    Returns the level and the list of factors that explains it."""
    n = len(verified)
    n_ok = sum(1 for c in verified if c.status == "VERIFIED")
    top = suff.top_score
    factors = [f"{n_ok}/{n} claims have citations that passed all verification checks."]
    if suff.rerank_scored and top is not None:
        factors.append(f"Best passage relevance score: {top:.2f}.")
    factors.append(f"{suff.coverage:.0%} of the question's key terms occur in the evidence.")
    if n_hard_conflicts:
        factors.append(f"{n_hard_conflicts} conflicting figure(s) found across sources: shown, not resolved.")
    # the relevance thresholds are cross-encoder logits: without reranking they do not apply
    strong = not suff.rerank_scored or (top is not None and top >= s.high_conf_rerank)
    relevant = not suff.rerank_scored or (top is not None and top >= s.medium_conf_rerank)
    if n and n_ok == n and strong and suff.coverage >= 0.8 and not n_hard_conflicts:
        level = "HIGH"
    elif n_ok >= max(1, n / 2) and relevant and not (n_hard_conflicts and n_ok < n):
        level = "MEDIUM"
    else:
        level = "LOW"
    if n_hard_conflicts and level == "HIGH":
        level = "MEDIUM"
    return level, factors


class AnswerService:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient.from_settings()

    # ------------------------------------------------------------------ one LLM call (+ one repair)
    def _generate(self, messages: list[dict], on_token) -> tuple[LLMAnswer, dict]:
        r = self.client.chat(messages, fmt=prompts.answer_schema(), on_token=on_token)
        usage = {"prompt": r.prompt_tokens, "completion": r.completion_tokens}
        try:
            return parse_llm_json(r.content), {**usage, "raw": r.content}
        except (ValueError, ValidationError):
            fix = messages + [{"role": "assistant", "content": r.content[:3000]},
                              {"role": "user", "content": "That was not valid JSON for the required shape. "
                                                          "Return ONLY the corrected JSON object."}]
            r2 = self.client.chat(fix, fmt=prompts.answer_schema(), on_token=on_token)
            usage = {"prompt": usage["prompt"] + r2.prompt_tokens, "completion": usage["completion"] + r2.completion_tokens}
            try:
                return parse_llm_json(r2.content), {**usage, "raw": r2.content}
            except (ValueError, ValidationError) as e:
                raise AnswerFormatError("The model did not return a valid JSON answer, even after a repair request.") from e

    # ------------------------------------------------------------------ the pipeline
    def answer(self, case_id: str, question: str, mode: str | None = None, model: str | None = None,
               rerank: bool | None = None, use_retrieval: bool = True, final_k: int | None = None,
               on_token: Callable[[str], None] | None = None,
               on_status: Callable[[str], None] | None = None, verify: bool = True) -> AnswerResult:
        """Answer one question about a case and record it in the question log.

        ``verify=False`` is the ablation "without the verifier": the first draft is shown as it is, with no retry and no
        refusal when no citation checks out. The citations are still checked, so their status can be reported, but
        nothing is done about a failure. It exists to measure what the verifier contributes, not for normal use."""
        res = self._answer(case_id, question, mode, model, rerank, use_retrieval, final_k, on_token, on_status, verify)
        try:
            db.log_qa(res.case_id, res.question, res.abstained, res.confidence, res.mode, res.model,
                      res.timings.get("total_s", 0.0))
        except Exception:  # noqa: BLE001 - logging must never break answering
            pass
        return res

    def _answer(self, case_id: str, question: str, mode: str | None, model: str | None, rerank: bool | None,
                use_retrieval: bool, final_k: int | None, on_token, on_status, verify: bool = True) -> AnswerResult:
        s = get_settings()
        status = on_status or (lambda m: None)
        mode = mode or s.retrieval_mode
        if model:
            self.client.model = model
        t_start = time.perf_counter()
        docs = case_docs(case_id)
        res = AnswerResult(question=question, case_id=case_id, answer="", abstained=False, confidence="LOW",
                           mode=mode if use_retrieval else "none", model=self.client.model,
                           use_retrieval=use_retrieval)
        try:
            if not use_retrieval:
                return self._no_retrieval(res, question, on_token, t_start)
            status("Retrieving passages (BM25 + dense + RRF)" if mode == "hybrid" else f"Retrieving passages ({mode})")
            hits, tm = hybrid.retrieve(case_id, question, mode, final_k=final_k, rerank=rerank)
            res.retrieved = hits
            known = {d.document_id for d in docs}
            if any(h.chunk.document_id not in known for h in hits):
                raise StaleIndexError(
                    "This case's index is out of date: its documents have no metadata, so no citation could be verified "
                    "and the model was not asked. Re-index the documents (Cases and documents > Re-index), or delete "
                    "storage/cases and storage/chroma and upload them again.")
            res.timings.update({"retrieval_s": tm["search_s"], "rerank_s": tm["rerank_s"]})

            suff = evidence_checker.assess(question, hits, s)
            names = [d.filename for d in docs]
            if not suff.sufficient:
                return self._abstain(res, question, hits, names, suff, t_start)

            res.comparisons = conflict_detector.compare_amounts(question, hits)
            figures = "\n".join(f"- {c.result}" for c in res.comparisons)
            messages = [{"role": "system", "content": prompts.SYSTEM_PROMPT},
                        {"role": "user", "content": prompts.build_user_prompt(question, hits, figures)}]
            status("Generating answer with the local LLM")
            t_gen = time.perf_counter()
            verified: list[VerifiedClaim] = []
            llm: LLMAnswer | None = None
            tokens = {"prompt": 0, "completion": 0}
            retries = s.verify_retries if verify else 0
            for attempt in range(retries + 1):
                llm, info = self._generate(messages, on_token)
                tokens = {k: tokens[k] + info[k] for k in tokens}
                if llm.abstained:
                    break
                status("Verifying citations")
                verified = citation_validator.validate_answer(llm, hits, docs, s.quote_match_min)
                if not verify:
                    break                                   # ablation: report the checks, act on none of them
                problems = [f"Claim '{c.claim[:80]}': " + ("; ".join(n for e in c.evidence for n in e.notes)
                                                          or "no citation given")
                            for c in verified if c.status != "VERIFIED"]
                if not problems and verified:
                    break
                if attempt < retries:               # tell the model exactly what the verifier rejected
                    res.retry_log.append(f"Attempt {attempt + 1} rejected by the verifier: " + " | ".join(problems)[:600])
                    messages = messages + [{"role": "assistant", "content": info["raw"][:3000]},
                                           {"role": "user", "content": prompts.build_feedback_prompt(
                                               problems or ["No claims with citations."])}]
                    status("Verifier rejected the draft - reformulating")
            res.timings["generation_s"] = round(time.perf_counter() - t_gen, 2)
            res.tokens = tokens

            if llm is None or llm.abstained:
                return self._abstain(res, question, hits, names, suff, t_start,
                                     ["The model found that the passages do not answer the question."])
            if verify and not any(c.status == "VERIFIED" for c in verified):
                out = self._abstain(res, question, hits, names, suff, t_start,
                                    ["The generated answer could not be grounded: none of its citations passed verification."])
                out.abstention["draft_rejected"] = llm.answer       # shown for transparency, never as the answer
                return out
            res.claims = verified
            res.answer = llm.answer
            res.reasoning_summary = llm.reasoning_summary
            res.conflicts = conflict_detector.detect_conflicts(hits)
            for lc in llm.conflicts:        # a conflict reported by the model is kept only if it cites retrieved evidence
                if lc.description and any(citation_validator.resolve_chunk(src, hits, docs)[0] for src in lc.sources):
                    res.conflicts.append(Conflict(kind="llm", topic="reported by the model", classification="CONFLICT",
                                                  explanation=lc.description, sources=lc.sources))
            n_hard = sum(1 for c in res.conflicts if c.classification == "CONFLICT")
            res.confidence, res.confidence_factors = compute_confidence(suff, verified, n_hard, s)
        except StaleIndexError as e:
            res.error = str(e)
            res.answer = f"This case cannot be answered: {e}"
            res.confidence = "LOW"
        except (OllamaError, AnswerFormatError) as e:
            res.error = str(e)
            res.answer = f"The local LLM could not be used: {e}"
            res.confidence = "LOW"
        res.timings["total_s"] = round(time.perf_counter() - t_start, 2)
        return res

    # ------------------------------------------------------------------ the other outcomes
    def _no_retrieval(self, res: AnswerResult, question: str, on_token, t_start) -> AnswerResult:
        """Baseline for the evaluation: the same model WITHOUT any documents (ungrounded, no citations)."""
        t = time.perf_counter()
        r = self.client.chat([{"role": "system", "content": prompts.NO_RETRIEVAL_SYSTEM},
                              {"role": "user", "content": question}], on_token=on_token)
        res.answer = r.content.strip()
        res.reasoning_summary = ("Generated without document retrieval: this answer has NO citations and is "
                                 "not grounded in the uploaded documents.")
        res.confidence_factors = ["No retrieval: answer is ungrounded."]
        res.timings = {"retrieval_s": 0.0, "rerank_s": 0.0, "generation_s": round(time.perf_counter() - t, 2),
                       "total_s": round(time.perf_counter() - t_start, 2)}
        res.tokens = {"prompt": r.prompt_tokens, "completion": r.completion_tokens}
        return res

    def _abstain(self, res: AnswerResult, question, hits, names, suff, t_start, extra=None) -> AnswerResult:
        info = abstention.build_abstention(question, hits, names, suff, extra)
        res.abstained, res.confidence = True, "INSUFFICIENT EVIDENCE"
        res.answer = f"{info['answer']}\n\n{info['explanation']}"
        res.abstention = info
        res.confidence_factors = info["reasons"]
        res.timings.setdefault("generation_s", 0.0)
        res.timings["total_s"] = round(time.perf_counter() - t_start, 2)
        return res
