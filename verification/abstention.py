"""Abstention: an honest "INSUFFICIENT EVIDENCE" answer that never uses outside knowledge."""
from __future__ import annotations

from schemas import RetrievedChunk
from verification.evidence_checker import Sufficiency

ABSTAIN_TEXT = "INSUFFICIENT EVIDENCE"


def build_abstention(question: str, hits: list[RetrievedChunk], searched_docs: list[str],
                     suff: Sufficiency | None, extra_reasons: list[str] | None = None) -> dict:
    """Explain what was searched and which related passages exist, so the refusal is useful, not just 'no'."""
    related = [{
        "document": h.chunk.document_name, "page": h.chunk.page_number, "clause": h.chunk.clause,
        "section": h.chunk.section, "score": h.rerank_score if h.rerank_score is not None else h.dense_score,
        "preview": h.chunk.text[:220],
    } for h in hits[:3]]
    reasons = list(suff.reasons if suff else []) + (extra_reasons or [])
    missing = suff.missing_terms if suff else []
    explanation = (
        "The uploaded documents do not contain enough evidence to answer this question reliably. "
        + (f"Key terms not found in the retrieved text: {', '.join(missing)}. " if missing else "")
        + (f"Related passages were found ({len(related)}) but they do not answer the question. "
           if related else "No related passages were found. ")
        + "Lexora AI does not fill gaps with general legal knowledge."
    )
    return {"answer": ABSTAIN_TEXT, "explanation": explanation, "searched_documents": searched_docs,
            "looked_for": question, "related_passages": related, "reasons": reasons,
            "features": {"top_score": suff.top_score if suff else None,
                         "coverage": suff.coverage if suff else None}}
