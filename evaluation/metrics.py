"""Retrieval metrics, computed against labelled evidence (document + clause).

A question lists the evidence a correct answer needs, e.g. [{"document": "GDPR.pdf", "clause": "33(1)"}].
A retrieved chunk "matches" an item when it comes from that document and its clause label equals the
expected one or is a sub-clause of it (expected "83(5)" is matched by "83(5)(a)").

  recall@k     share of the expected evidence items found in the top k   (averaged over questions)
  hit@k        1 if at least one expected item is in the top k
  full@k       1 if ALL expected items are in the top k (strict, for multi-hop questions)
  precision@k  share of the top-k chunks that match any expected item
  mrr          1 / rank of the first matching chunk (0 if none)

Generation (an answer to one question):
  answer_correct_proxy  AUTOMATIC key-fact check: every key fact of the question occurs in the answer. A proxy, not
                        a human judgement (it misses paraphrases); see evaluation/grading.py for human grades.
  citation_precision    emitted citations that point at the labelled evidence / emitted citations
  abstention_*          did the system refuse exactly when the question is unanswerable?
"""
from __future__ import annotations

from statistics import mean

from schemas import Chunk
from textutil import clause_key, normalize, numbers, stem, words


def clause_matches(chunk_clause: str, expected: str) -> bool:
    c, e = clause_key(chunk_clause), clause_key(expected)
    return bool(c) and (c == e or c.startswith(e + "(") or c.startswith(e + "."))


def chunk_matches(chunk: Chunk, evidence: dict) -> bool:
    return chunk.document_name == evidence["document"] and clause_matches(chunk.clause, evidence["clause"])


def retrieval_scores(ranked: list[Chunk], expected: list[dict], ks=(1, 3, 5)) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        top = ranked[:k]
        covered = [any(chunk_matches(c, e) for c in top) for e in expected]
        out[f"recall@{k}"] = sum(covered) / len(expected)
        out[f"hit@{k}"] = float(any(covered))
        out[f"full@{k}"] = float(all(covered))
        out[f"precision@{k}"] = (sum(any(chunk_matches(c, e) for e in expected) for c in top) / k) if top else 0.0
    out["mrr"] = next((1 / i for i, c in enumerate(ranked, 1) if any(chunk_matches(c, e) for e in expected)), 0.0)
    return out


def aggregate(rows: list[dict], keys: list[str] | None = None) -> dict[str, float]:
    """Mean of each numeric key over the rows."""
    if not rows:
        return {}
    keys = keys or [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    return {k: round(mean(r[k] for r in rows if r.get(k) is not None), 4) for k in keys}


# ------------------------------------------------------------------------------ generation
def fact_present(fact: str, answer: str) -> bool:
    """Does the key fact occur in the answer? Exact text first; a purely numeric fact compares quantities
    ('20 000 000' matches 'EUR 20 million', '4 %' matches '4%'); otherwise every word (stemmed) must occur."""
    if normalize(fact) in normalize(answer):
        return True
    fact_numbers = numbers(fact, strip_refs=False)
    if fact_numbers and not [w for w in words(fact) if not w.isdigit()]:
        return fact_numbers <= numbers(answer, strip_refs=False)
    return {stem(w) for w in words(fact)} <= {stem(w) for w in words(answer)}


def answer_correct_proxy(key_facts: list[str], answer: str) -> bool:
    return bool(key_facts) and all(fact_present(f, answer) for f in key_facts)


def citation_counts(claims: list[dict]) -> tuple[int, int]:
    """(citations whose status is VERIFIED, all citations) over the claims of an AnswerResult."""
    refs = [ev for c in claims for ev in c["evidence"]]
    return sum(1 for ev in refs if ev["status"] == "VERIFIED"), len(refs)


def citation_precision(claims: list[dict], expected: list[dict]) -> tuple[int, int]:
    """(citations pointing at expected evidence, all citations). A citation to a valid NEIGHBOURING clause counts
    as a miss here, which is why this is reported next to the verified rate."""
    good = total = 0
    for c in claims:
        for ev in c["evidence"]:
            total += 1
            ref = ev["ref"]
            if any(ref["document"] == e["document"] and clause_matches(ref["clause"], e["clause"]) for e in expected):
                good += 1
    return good, total


def abstention_stats(rows: list[dict]) -> dict:
    """rows: {'answerable': bool, 'abstained': bool}. accuracy counts a correct decision either way; the false
    abstention rate is the share of ANSWERABLE questions that were refused; the correct abstention rate is the
    share of UNANSWERABLE questions that were refused."""
    if not rows:
        return {}
    ans, unans = [r for r in rows if r["answerable"]], [r for r in rows if not r["answerable"]]
    return {
        "abstention_accuracy": round(mean(1.0 if r["abstained"] == (not r["answerable"]) else 0.0 for r in rows), 4),
        "false_abstention_rate": round(mean(1.0 if r["abstained"] else 0.0 for r in ans), 4) if ans else None,
        "correct_abstention_rate": round(mean(1.0 if r["abstained"] else 0.0 for r in unans), 4) if unans else None,
    }
