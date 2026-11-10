"""Run the evaluation on the held-out questions.

    python -m evaluation.evaluate GDPR.pdf ePrivacy.pdf DataAct.pdf --retrieval-only        # about 1 minute, no LLM
    python -m evaluation.evaluate GDPR.pdf ePrivacy.pdf DataAct.pdf --gen-configs no_retrieval hybrid   # + LLM answers

Two mandatory comparisons: dense vs hybrid retrieval, and no-retrieval vs retrieval-enabled answers.
LLM answers are slow (about 80 s each on a CPU), so every answer is cached as it is produced: an interrupted run
resumes where it stopped. Results go to evaluation/results/: retrieval_by_question.csv, generation_by_question.csv,
summary.json, summary.md (tables ready for the README) and meta.json (what exactly was run).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Callable

from config import ROOT, get_settings
from evaluation import grading, region_eval
from evaluation.dataset import EVAL_DIR, TEST_SET, EvalQuestion, load_dataset
from evaluation.metrics import (abstention_stats, aggregate, answer_correct_proxy, citation_counts,
                                         citation_precision)
from generation.answer_generator import AnswerService
from services import indexer
from scripts.eval_retrieval import evaluate as evaluate_retrieval
from scripts.search_pdf import ensure_indexed
from services.paths import GlobPaths

RESULTS = EVAL_DIR / "results"
# the answer configurations that can be compared (the retrieval ones always rerank)
GEN_CONFIGS: dict[str, dict] = {
    "no_retrieval": {"use_retrieval": False},
    "bm25": {"use_retrieval": True, "mode": "bm25", "rerank": True},
    "dense": {"use_retrieval": True, "mode": "dense", "rerank": True},
    "hybrid": {"use_retrieval": True, "mode": "hybrid", "rerank": True},
    # the ablation "without the verifier": the first draft is shown unchecked (see AnswerService.answer, verify=False)
    "hybrid_noverify": {"use_retrieval": True, "mode": "hybrid", "rerank": True, "verify": False},
}
ABLATION_CONFIGS = {"hybrid_noverify"}      # measured automatically only: never part of the hand-graded sheet


# ------------------------------------------------------------------------------ cache of LLM answers
def settings_signature() -> dict:
    """Everything that can change an answer: if any of it changes, cached answers must not be reused."""
    s = get_settings()
    keys = ["ollama_model", "ollama_temperature", "ollama_num_ctx", "ollama_num_predict", "ollama_think", "seed",
            "final_top_k", "initial_top_k", "rrf_k", "query_decomposition", "abstain_rerank_min",
            "abstain_coverage_min", "verify_retries", "quote_match_min", "chunk_max_chars", "chunk_overlap",
            "embedding_model", "reranker_model", "high_conf_rerank", "medium_conf_rerank"]
    return {k: getattr(s, k) for k in keys}


class GenCache:
    """One JSON line per answer, appended as soon as it is produced."""

    def __init__(self, path: Path | None):
        self.path, self.rows = path, {}
        if path and path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    self.rows[item["key"]] = item["row"]

    @staticmethod
    def key(case_id: str, config: str, question: str) -> str:
        """Answers that use retrieval depend on the index, so the key includes its fingerprint: indexing again makes the
        old answers unusable. The no-retrieval baseline never looks at the index, so its key (and the grades it earned)
        stays valid."""
        payload = {"case": case_id, "config": config, "q": question, "settings": settings_signature()}
        if config != "no_retrieval":
            payload["index"] = indexer.index_fingerprint(case_id)
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def put(self, key: str, row: dict) -> None:
        self.rows[key] = row
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "row": row}) + "\n")


# ------------------------------------------------------------------------------ generation
def citation_details(res: dict) -> list[dict]:
    """Every emitted citation as {document, page, clause, quote, status}. The document is taken from the passage the
    verifier matched (the real file name) when there is one, otherwise from what the model wrote."""
    passages = {h["chunk"]["chunk_id"]: h["chunk"] for h in res.get("retrieved", [])}
    out = []
    for claim in res["claims"]:
        for ev in claim["evidence"]:
            ref, src = ev["ref"], passages.get(ev.get("matched_chunk_id") or "")
            out.append({"document": src["document_name"] if src else ref["document"],
                        "page": ref["page"] or (src["page_number"] if src else None), "clause": ref["clause"],
                        "quote": ref["supporting_text"][:400], "status": ev["status"]})
    return out


def generation_row(q: EvalQuestion, config: str, res: dict) -> dict:
    """One row per (question, configuration), measured against the labels."""
    verified, emitted = citation_counts(res["claims"])
    shown = [] if res["abstained"] else res["claims"]
    gold, _ = citation_precision(res["claims"], q.evidence_dicts())
    t, tok = res["timings"], res.get("tokens") or {}
    abstention = res.get("abstention") or {}
    reasons = abstention.get("reasons", [])
    if res["error"]:
        stage = "error"
    elif res["abstained"]:                  # who said no? the evidence gate, the model itself, or the citation verifier
        stage = ("verifier" if any("could not be grounded" in r for r in reasons) else
                 "model" if any("do not answer the question" in r for r in reasons) else "gate")
    else:
        stage = ""
    return {
        "id": q.id, "category": q.category, "config": config, "answerable": q.answerable,
        "abstained": res["abstained"], "confidence": res["confidence"], "answer": res["answer"].strip(),
        "n_claims": len(res["claims"]), "verified_cites": verified, "total_cites": emitted, "gold_cites": gold,
        "correct_proxy": answer_correct_proxy(q.key_facts, res["answer"]) if q.answerable else None,
        "retrieval_s": t.get("retrieval_s", 0.0), "rerank_s": t.get("rerank_s", 0.0),
        "generation_s": t.get("generation_s", 0.0), "total_s": t.get("total_s", 0.0),
        "prompt_tokens": tok.get("prompt", 0), "completion_tokens": tok.get("completion", 0),
        "retries": len(res["retry_log"]), "error": res["error"], "refusal_stage": stage,
        "refusal_reason": " | ".join(reasons)[:300], "verifier_note": (res["retry_log"][0] if res["retry_log"] else "")[:300],
        "rejected_draft": (abstention.get("draft_rejected") or "")[:200],
        "claims_shown": len(shown), "unsupported_shown": sum(1 for c in shown if c["status"] != "VERIFIED"),
        "citations": citation_details(res),
    }


def run_generation(case_id: str, questions: list[EvalQuestion], configs: list[str], service, cache: GenCache,
                   fresh: bool = False, log: Callable[[str], None] = lambda m: None,
                   refresh_citations: bool = False) -> list[dict]:
    """Answer every question under every configuration (cached answers are reused unless ``fresh``)."""
    rows, total, n = [], len(configs) * len(questions), 0
    for config in configs:
        opts = GEN_CONFIGS[config]
        for q in questions:
            n += 1
            key = GenCache.key(case_id, config, q.question)
            row = None if fresh else cache.rows.get(key)
            if row is not None and row["error"]:                  # a failure is not an answer: ask again
                row = None
            if row is not None and refresh_citations and opts["use_retrieval"] and "citations" not in row:
                row = None                                        # cached before citation detail was stored
            if row is None:
                extra = {"verify": False} if opts.get("verify") is False else {}
                res = service.answer(case_id, q.question, opts.get("mode"), None, opts.get("rerank"),
                                     opts["use_retrieval"], **extra).model_dump()
                row = generation_row(q, config, res)
                if not row["error"]:
                    cache.put(key, row)
                outcome = f"ERROR: {row['error'][:60]}" if row["error"] else (
                    f"ABSTAINED ({row['refusal_stage']})" if row["abstained"] else "answered")
                log(f"[{n}/{total}] {config:<12} {q.id:<4} {row['total_s']:>6.1f}s  {outcome}")
            else:
                log(f"[{n}/{total}] {config:<12} {q.id:<4} (cached)")
            rows.append(row)
    return rows


def summarize_generation(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for config in dict.fromkeys(r["config"] for r in rows):
        rs = [r for r in rows if r["config"] == config]
        ans = [r for r in rs if r["answerable"]]
        cites = sum(r["total_cites"] for r in rs)
        by_cat = {c: round(mean(1.0 if r["correct_proxy"] else 0.0 for r in ans if r["category"] == c), 4)
                  for c in sorted({r["category"] for r in ans})}
        out[config] = {
            "n": len(rs), "n_answerable": len(ans), "n_unanswerable": len(rs) - len(ans),
            "answer_correct_proxy": round(mean(1.0 if r["correct_proxy"] else 0.0 for r in ans), 4) if ans else None,
            "answer_correct_proxy_by_category": by_cat,
            "citations_verified_rate": round(sum(r["verified_cites"] for r in rs) / cites, 4) if cites else None,
            "citation_precision": round(sum(r["gold_cites"] for r in rs) / cites, 4) if cites else None,
            **abstention_stats(rs),
            "mean_retrieval_s": round(mean(r["retrieval_s"] + r["rerank_s"] for r in rs), 2),
            "mean_generation_s": round(mean(r["generation_s"] for r in rs), 2),
            "mean_total_s": round(mean(r["total_s"] for r in rs), 2),
            "mean_prompt_tokens": round(mean(r["prompt_tokens"] for r in rs)),
            "mean_completion_tokens": round(mean(r["completion_tokens"] for r in rs)),
            "n_errors": sum(1 for r in rs if r["error"]),
            "unsupported_claim_rate": (round(sum(r.get("unsupported_shown", 0) for r in rs) / n_shown, 4)
                                       if (n_shown := sum(r.get("claims_shown", 0) for r in rs)) else None),
            "answers_with_unsupported_claim": (round(sum(1 for r in answered if r.get("unsupported_shown", 0)) / len(answered), 4)
                                               if (answered := [r for r in rs if not r["abstained"] and r.get("claims_shown", 0)]) else None),
            "answerable_refusals_by_stage": {st: sum(1 for r in ans if r.get("refusal_stage", "") == st)
                                 for st in ("gate", "model", "verifier", "error")},
            "answerable_refused_ids": [r["id"] for r in ans if r["abstained"]],
        }
    return out


# ------------------------------------------------------------------------------ retrieval
def run_retrieval(case_id: str, questions: list[EvalQuestion]) -> tuple[dict, dict, list[dict]]:
    """(summary per configuration, factual/multihop breakdown, one CSV row per configuration and question)."""
    answerable = [q for q in questions if q.answerable]
    summary, per_question = evaluate_retrieval(
        case_id, [{"id": q.id, "question": q.question, "expected_evidence": q.evidence_dicts()} for q in answerable])
    category = {q.id: q.category for q in answerable}
    by_category, rows = {}, []
    for label, qrows in per_question.items():
        for cat in ("factual", "multihop"):
            sub = [r for r in qrows if category[r["id"]] == cat]
            if sub:
                by_category[f"{cat} / {label}"] = aggregate(sub, ["recall@1", "recall@5", "full@5"])
        rows += [{"config": label, "category": category[r["id"]], **{k: round(v, 3) if isinstance(v, float) else v
                                                                    for k, v in r.items()}} for r in qrows]
    return summary, by_category, rows


# ------------------------------------------------------------------------------ report
def _fmt(v) -> str:
    return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def markdown_report(meta: dict, retrieval: dict, by_category: dict, generation: dict, manual: dict | None = None,
                    region: dict | None = None) -> str:
    out = [f"Run {meta['date']} - model `{meta['ollama_model']}`, seed {meta['seed']}, "
           f"{meta['n_questions']} held-out questions ({meta['n_answerable']} answerable).", ""]
    if retrieval:
        cols = ["recall@1", "recall@3", "recall@5", "full@5", "precision@5", "mrr", "latency_ms"]
        out += ["### Retrieval (answerable questions)", "", "| Configuration | " + " | ".join(cols) + " |",
                "|---|" + "---|" * len(cols)]
        out += [f"| {label} | " + " | ".join(_fmt(m.get(c)) if c != "latency_ms" else f"{m[c]:.0f}" for c in cols) + " |"
                for label, m in retrieval.items()]
        out += ["", "`recall@k` = share of the required evidence items (document + clause) in the top k; `full@k` = share of "
                "questions whose evidence is ALL in the top k (strict for multi-hop).", ""]
    if by_category:
        out += ["| Category / configuration | recall@1 | recall@5 | full@5 |", "|---|---|---|---|"]
        out += [f"| {k} | {_fmt(v['recall@1'])} | {_fmt(v['recall@5'])} | {_fmt(v['full@5'])} |" for k, v in by_category.items()]
        out.append("")
    if generation:
        names = list(generation)
        rows = [("Answer correct, automatic key-fact PROXY (answerable)", "answer_correct_proxy"),
                ("Citations verified by the validator", "citations_verified_rate"),
                ("Citation precision vs labelled evidence", "citation_precision"),
                ("Claims shown that the validator does not verify", "unsupported_claim_rate"),
                ("Answers shown with at least one such claim", "answers_with_unsupported_claim"),
                ("Abstention accuracy (all questions)", "abstention_accuracy"),
                ("False-abstention rate (answerable refused)", "false_abstention_rate"),
                ("Correct-abstention rate (unanswerable refused)", "correct_abstention_rate"),
                ("Mean retrieval + rerank time (s)", "mean_retrieval_s"), ("Mean generation time (s)", "mean_generation_s"),
                ("Mean total time (s)", "mean_total_s"), ("Mean completion tokens", "mean_completion_tokens"),
                ("Answers that failed with an error", "n_errors")]
        out += ["### Generation", "", "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        out += [f"| {label} | " + " | ".join(_fmt(generation[n].get(key)) for n in names) + " |" for label, key in rows]
        for name in names:
            st = generation[name].get("answerable_refusals_by_stage") or {}
            if st.get("verifier") or st.get("error"):
                out += ["", f"**WARNING ({name}):** {st.get('verifier', 0)} answerable question(s) were refused AFTER the model "
                        f"answered because no citation passed verification, and {st.get('error', 0)} failed with an error. "
                        "That usually points to a problem (an out-of-date index, the wrong model, Ollama not running), not to "
                        "a property of the system. Read `refusal_reason` and `verifier_note` in generation_by_question.csv "
                        "before using these numbers."]
        out += ["", "The correctness row is an AUTOMATIC proxy (every key fact occurs in the answer); it misses paraphrases. "
                "It is not a human judgement. Grade the answers yourself with `--export-grading` / `--apply-grades`.", ""]
    if region and region.get("n_citations"):
        out += ["### Citation precision at region level", "",
                f"Configuration `{region['config']}`, {region['n_citations']} emitted citations. A citation is a REGION hit when the quote "
                "it points at lies inside the labelled clause on the page (see `evaluation/region_eval.py`).", "",
                "| Measure | Value |", "|---|---|",
                f"| Quote found on the cited page (can be drawn) | {_fmt(region['located_rate'])} |",
                f"| Page-level precision | {_fmt(region['page_precision'])} |",
                f"| Region-level precision | {_fmt(region['region_precision'])} |",
                f"| Region-level precision, of the located quotes | {_fmt(region['region_precision_of_located'])} |",
                f"| Citations on the right page but outside the labelled region | {region['page_hits_outside_the_region']} |", ""]
    if manual:
        names = list(manual["configs"])
        rows = [("Answers graded", None), ("Answerable: mean score (1 / 0.5 / 0)", "answerable_mean_score"),
                ("Answerable: strictly correct (grade 1)", "answerable_strictly_correct"),
                ("Unanswerable: handled correctly", "unanswerable_handled_correctly"),
                ("All questions: behaviour correct", "behaviour_correct_all")]
        out += ["### Manual grades", "", f"Graded by: **{manual['grader']}** (file `{manual['file']}`). Scale: {manual['scale']}.", "",
                "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for label, key in rows:
            cells = [f"{manual['configs'][n]['n_graded']} / {manual['configs'][n]['n_total']}" if key is None
                     else _fmt(manual["configs"][n][key]) for n in names]
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        if any(c["n_stale"] for c in manual["configs"].values()):
            out += ["", "Some grades were IGNORED because the answer changed after it was graded: "
                    + ", ".join(f"{n}: {c['n_stale']}" for n, c in manual["configs"].items() if c["n_stale"])
                    + ". Grade those answers again (they are blank in a fresh --export-grading sheet)."]
        if any(c["n_graded"] < c["n_total"] for c in manual["configs"].values()):
            out += ["", "Grading is INCOMPLETE: the manual numbers cover only the graded answers."]
        out.append("")
    return "\n".join(out)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_meta(pdfs: list[Path], dataset_path: Path, questions: list[EvalQuestion]) -> dict:
    s = get_settings()

    def version(pkg: str) -> str:
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return "?"
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    from ingestion.pdf_parser import sha256_file
    return {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), "git_commit": commit,
            "n_questions": len(questions), "n_answerable": sum(q.answerable for q in questions),
            "dataset": dataset_path.name, "dataset_sha256": sha256_file(dataset_path),
            "documents": [{"name": p.name, "sha256": sha256_file(p)} for p in pdfs],
            "python": sys.version.split()[0],
            "libraries": {p: version(p) for p in ("torch", "sentence-transformers", "chromadb", "rank-bm25", "pymupdf")},
            **settings_signature()}


# ------------------------------------------------------------------------------ command line
def main(argv: list[str] | None = None, service=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="+", action=GlobPaths)
    ap.add_argument("--dataset", type=Path, default=TEST_SET)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--retrieval-only", action="store_true", help="skip the LLM")
    ap.add_argument("--gen-configs", nargs="+", choices=list(GEN_CONFIGS), default=["no_retrieval", "hybrid"])
    ap.add_argument("--limit", type=int, help="only the first N questions (for a quick trial)")
    ap.add_argument("--fresh", action="store_true", help="ignore cached answers")
    ap.add_argument("--refresh-citations", action="store_true",
                    help="ask again the cached retrieval answers that were stored without citation detail")
    ap.add_argument("--export-grading", nargs="?", const="", metavar="CSV",
                    help="write a sheet to grade the answers by hand (default: <out>/manual_grades.csv)")
    ap.add_argument("--apply-grades", type=Path, metavar="CSV", help="add the manual grades from this sheet to the report")
    ap.add_argument("--grader", default="unspecified", help="who graded (stated in the report)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    dataset = load_dataset(args.dataset)
    questions = dataset.questions[:args.limit] if args.limit else dataset.questions
    log = lambda m: print(m, file=sys.stderr, flush=True)                        # noqa: E731
    case_id = ensure_indexed(args.pdfs)
    out = args.out
    old = json.loads((out / "summary.json").read_text(encoding="utf-8")) if (out / "summary.json").is_file() else {}
    meta = run_meta(args.pdfs, args.dataset, questions)
    summary = {"meta": meta, "retrieval": {}, "by_category": {}, "generation": {}}

    retrieval, by_category, retrieval_rows = run_retrieval(case_id, questions)
    summary.update(retrieval=retrieval, by_category=by_category)
    write_csv(out / "retrieval_by_question.csv", retrieval_rows)

    if args.retrieval_only:
        summary["generation"] = old.get("generation", {})                         # keep earlier answer results
    else:
        rows = run_generation(case_id, questions, args.gen_configs, service or AnswerService(),
                              GenCache(out / "gen_cache.jsonl"), args.fresh, log, args.refresh_citations)
        write_csv(out / "generation_by_question.csv", [{k: v for k, v in r.items() if k != "citations"} for r in rows])
        summary["generation"] = summarize_generation(rows)
        if any(r["config"] == "hybrid" for r in rows):
            region = region_eval.region_report(indexer.case_chunks(case_id), questions, rows, {p.name: p for p in args.pdfs})
            summary["region"] = region
            if region["rows_without_detail"]:
                log(f"{region['rows_without_detail']} cached hybrid answers have no citation detail: run again with "
                    "--refresh-citations to ask those questions once more (a grade is dropped if the answer changes)")
        graded_rows = [r for r in rows if r["config"] not in ABLATION_CONFIGS]
        if args.export_grading is not None:
            sheet = Path(args.export_grading) if args.export_grading else out / "manual_grades.csv"
            n, kept, dropped = grading.export_template(graded_rows, {q.id: q.expected_answer for q in questions},
                                                       {q.id: q.question for q in questions}, sheet)
            log(f"grading sheet with {n} answers written to {sheet}: {kept} earlier grades kept, "
                f"{dropped} dropped because the answer changed")
        if args.apply_grades:
            summary["manual"] = {"grader": args.grader, "file": args.apply_grades.name, "scale": grading.SCALE,
                                 "configs": grading.summarize_grades(graded_rows, grading.load_grades(args.apply_grades))}

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    report = markdown_report(meta, retrieval, by_category, summary["generation"], summary.get("manual"),
                             summary.get("region"))
    (out / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
