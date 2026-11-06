"""First-stage retrieval: BM25, dense (ChromaDB) or hybrid (Reciprocal Rank Fusion).

Every hit keeps the evidence of each stage (BM25 score and rank, dense similarity and rank,
RRF score) so a result can always be explained.
"""
from __future__ import annotations

import json
import threading
import time

from config import get_settings
from retrieval import vector_store
from retrieval.bm25 import BM25Index
from retrieval.embeddings import embed_query
from retrieval.reranker import rerank as cross_rerank
from schemas import Chunk, RetrievedChunk
from services.paths import chunks_file

MODES = ("bm25", "dense", "hybrid")

_cache: dict[str, tuple[tuple[int, int], dict[str, Chunk], BM25Index]] = {}
_lock = threading.Lock()


def load_case_index(case_id: str) -> tuple[dict[str, Chunk], BM25Index]:
    """The chunks of a case (by id) and their BM25 index.

    BM25 is not stored on disk: it is rebuilt from chunks.json and cached in memory. The cache
    entry is keyed by the file's (mtime, size), so a changed file always triggers a rebuild.
    """
    f = chunks_file(case_id)
    if not f.exists():
        return {}, BM25Index([])
    st = f.stat()
    stamp = (st.st_mtime_ns, st.st_size)
    with _lock:
        hit = _cache.get(case_id)
        if hit and hit[0] == stamp:
            return hit[1], hit[2]
        chunks = [Chunk(**d) for d in json.loads(f.read_text(encoding="utf-8"))]
        entry = (stamp, {c.chunk_id: c for c in chunks}, BM25Index(chunks))
        _cache[case_id] = entry
        return entry[1], entry[2]


def invalidate(case_id: str) -> None:
    with _lock:
        _cache.pop(case_id, None)


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d)), ranks start at 1.

    It merges ranked lists WITHOUT comparing their scores (BM25 scores and cosine similarities
    live on different scales): only the positions matter. A chunk ranked high by both lists
    beats a chunk ranked first by only one.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def search(case_id: str, query: str, mode: str | None = None, top_k: int | None = None,
           rrf_k: int | None = None, dense_min_similarity: float | None = None,
           document_ids: list[str] | None = None) -> list[RetrievedChunk]:
    """Return up to ``top_k`` chunks for ``query``, best first, using the chosen mode."""
    s = get_settings()
    mode = mode or s.retrieval_mode
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    top_k = top_k or s.initial_top_k
    rrf_k = rrf_k or s.rrf_k
    dense_min = s.dense_min_similarity if dense_min_similarity is None else dense_min_similarity
    by_id, bm25 = load_case_index(case_id)
    if not by_id:
        return []
    allowed = None
    if document_ids:
        wanted = set(document_ids)
        allowed = {cid for cid, c in by_id.items() if c.document_id in wanted}

    bm_hits = bm25.search(query, top_k, allowed) if mode in ("bm25", "hybrid") else []
    de_hits = ([(i, sc) for i, sc in vector_store.query(case_id, embed_query(query), top_k, document_ids)
                if sc >= dense_min] if mode in ("dense", "hybrid") else [])
    bm_rank = {i: r for r, (i, _) in enumerate(bm_hits, 1)}
    de_rank = {i: r for r, (i, _) in enumerate(de_hits, 1)}
    bm_score, de_score = dict(bm_hits), dict(de_hits)

    fused = rrf_fuse([[i for i, _ in bm_hits], [i for i, _ in de_hits]], rrf_k)
    if mode == "bm25":
        order = [i for i, _ in bm_hits]
    elif mode == "dense":
        order = [i for i, _ in de_hits]
    else:
        order = sorted(fused, key=lambda i: (-fused[i], i))[:top_k]       # ties broken by id: deterministic

    out: list[RetrievedChunk] = []
    for cid in order:
        if cid not in by_id:                                              # stale vector with no chunk row
            continue
        out.append(RetrievedChunk(
            chunk=by_id[cid], bm25_score=bm_score.get(cid), dense_score=de_score.get(cid),
            rrf_score=fused.get(cid) if mode == "hybrid" else None,
            bm25_rank=bm_rank.get(cid), dense_rank=de_rank.get(cid), final_rank=len(out) + 1))
    return out


def retrieve(case_id: str, query: str, mode: str | None = None, initial_k: int | None = None,
             final_k: int | None = None, rerank: bool | None = None, rrf_k: int | None = None,
             document_ids: list[str] | None = None) -> tuple[list[RetrievedChunk], dict]:
    """The full retrieval stage: search (``initial_k`` candidates), then optionally rerank with the
    cross-encoder and keep ``final_k``. Returns (hits, timings in seconds)."""
    s = get_settings()
    final_k = final_k or s.final_top_k
    rerank = s.rerank_enabled if rerank is None else rerank
    t0 = time.perf_counter()
    hits = search(case_id, query, mode, initial_k, rrf_k, document_ids=document_ids)
    t1 = time.perf_counter()
    if rerank and hits:
        hits = cross_rerank(query, hits, final_k)
    else:
        hits = hits[:final_k]
        for i, h in enumerate(hits, 1):
            h.final_rank = i
    t2 = time.perf_counter()
    return hits, {"search_s": round(t1 - t0, 3), "rerank_s": round(t2 - t1, 3)}
