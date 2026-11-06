"""Local cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2).

First-stage retrieval (BM25 / dense) scores the query and each passage SEPARATELY, which is fast
but coarse. A cross-encoder reads the query and one passage TOGETHER and outputs a single
relevance logit, which is much more accurate but too slow to run over the whole corpus. So it
re-scores only the ~20 candidates the first stage returned and we keep the best few.
"""
from __future__ import annotations

import math
import threading

from config import get_settings
from schemas import RetrievedChunk

_models: dict[str, object] = {}
_lock = threading.Lock()


def get_reranker(name: str | None = None):
    """Load a CrossEncoder once per process (CPU). The first call downloads about 90 MB."""
    name = name or get_settings().reranker_model
    with _lock:
        if name not in _models:
            from sentence_transformers import CrossEncoder
            _models[name] = CrossEncoder(name, device="cpu", max_length=512)
        return _models[name]


def sigmoid(x: float) -> float:
    """Logit -> probability in (0, 1). Clamped so huge logits cannot overflow exp()."""
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


def score_pairs(pairs: list[tuple[str, str]], model_name: str | None = None) -> list[float]:
    """Relevance logit for each (query, passage) pair. Higher = more relevant; can be negative."""
    if not pairs:
        return []
    return [float(s) for s in get_reranker(model_name).predict(pairs, batch_size=16, show_progress_bar=False)]


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int,
           model_name: str | None = None) -> list[RetrievedChunk]:
    """Score every candidate against the query and return the best ``top_k``, renumbered 1..k."""
    if not candidates:
        return []
    scores = score_pairs([(query, c.chunk.index_text or c.chunk.text) for c in candidates], model_name)
    for c, s in zip(candidates, scores):
        c.rerank_score = s
        c.rerank_prob = sigmoid(s)
    ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)[:top_k]
    for i, c in enumerate(ranked, 1):
        c.final_rank = i
    return ranked
