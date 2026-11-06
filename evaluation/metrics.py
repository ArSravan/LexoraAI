"""Retrieval metrics, computed against labelled evidence (document + clause).

A question lists the evidence a correct answer needs, e.g. [{"document": "GDPR.pdf", "clause": "33(1)"}].
A retrieved chunk "matches" an item when it comes from that document and its clause label equals the
expected one or is a sub-clause of it (expected "83(5)" is matched by "83(5)(a)").

  recall@k     share of the expected evidence items found in the top k   (averaged over questions)
  hit@k        1 if at least one expected item is in the top k
  full@k       1 if ALL expected items are in the top k (strict, for multi-hop questions)
  precision@k  share of the top-k chunks that match any expected item
  mrr          1 / rank of the first matching chunk (0 if none)
"""
from __future__ import annotations

from statistics import mean

from schemas import Chunk
from textutil import clause_key


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
