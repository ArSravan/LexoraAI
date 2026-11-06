"""Evidence sufficiency: can the retrieved passages support an answer AT ALL?

The model must never fill gaps from general knowledge. Before the LLM is even called, two signals
decide whether to answer or to abstain:

  * relevance  the best cross-encoder score among the retrieved passages (is anything on topic?)
  * coverage   the share of the question's salient terms that occur in the top passages
               ("what does Article 99 say about blockchain?" -> 'blockchain' appears nowhere)

Both thresholds are settings, calibrated on a development set that is separate from the test questions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config import Settings
from retrieval.bm25 import STOPWORDS
from schemas import RetrievedChunk
from textutil import stem, words

# words that appear in many questions but say nothing about the topic
GENERIC = frozenset("""what which who whom whose does under document documents agreement agreements contract contracts
case uploaded provided state states say says according following stated regarding mention mentioned apply
applies provision provisions clause clauses article articles section sections page pages please tell
explain describe define defined whether amount number maximum minimum total much many long often kind kinds
violation violations required require requires requirement exceed exceeds compare combined combine""".split())

DENSE_MIN = 0.3         # fallback relevance threshold (cosine) when reranking is switched off


@dataclass
class Sufficiency:
    sufficient: bool
    top_score: float | None
    coverage: float
    missing_terms: list[str]
    n_relevant: int
    reasons: list[str] = field(default_factory=list)
    rerank_scored: bool = False     # True: top_score is a cross-encoder logit; False: a cosine similarity or None


def _salient(question: str) -> dict[str, str]:
    """{stem: the word as typed} for the topical words of a question, in order. Matching uses the
    stems; messages shown to the user use the original words."""
    out: dict[str, str] = {}
    for w in words(question):
        if w in STOPWORDS or w in GENERIC or len(w) < 3:
            continue
        out.setdefault(stem(w), w)
    return out


def salient_terms(question: str) -> list[str]:
    """The topical words of a question as stems, de-duplicated, in order."""
    return list(_salient(question))


def coverage(question: str, hits: list[RetrievedChunk], top_n: int = 5) -> tuple[float, list[str]]:
    """(share of salient terms found in the top ``top_n`` passages, the missing terms as typed)."""
    terms = _salient(question)
    if not terms:
        return 1.0, []
    text = " ".join(h.chunk.index_text or h.chunk.text for h in hits[:top_n])
    present = {stem(w) for w in words(text)}
    missing = [word for st, word in terms.items() if st not in present]
    return 1 - len(missing) / len(terms), missing


def assess(question: str, hits: list[RetrievedChunk], s: Settings) -> Sufficiency:
    """Decide whether the evidence is sufficient; ``reasons`` explains every failed test."""
    if not hits:
        return Sufficiency(False, None, 0.0, list(_salient(question).values()), 0,
                           ["No passages were retrieved for this question."])
    cov, missing = coverage(question, hits)
    reasons: list[str] = []
    scores = [h.rerank_score for h in hits if h.rerank_score is not None]
    if scores:
        top = max(scores)
        n_relevant = sum(1 for x in scores if x >= s.abstain_rerank_min)
        if top < s.abstain_rerank_min:
            reasons.append(f"Best passage relevance ({top:.2f}) is below the threshold ({s.abstain_rerank_min:.2f}).")
    else:                                   # reranking disabled: fall back to dense similarity
        dense = [h.dense_score for h in hits if h.dense_score is not None]
        top = max(dense) if dense else None
        n_relevant = sum(1 for x in dense if x >= DENSE_MIN)
        if top is None or top < DENSE_MIN:
            reasons.append("Retrieved passages have low semantic similarity to the question.")
    if cov < s.abstain_coverage_min:
        reasons.append(f"Only {cov:.0%} of the question's key terms occur in the retrieved passages "
                       f"(missing: {', '.join(missing) or 'n/a'}).")
    return Sufficiency(not reasons, top, cov, missing, n_relevant, reasons, rerank_scored=bool(scores))
