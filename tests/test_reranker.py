import math

import pytest

from retrieval import reranker
from retrieval.hybrid import retrieve, search
from retrieval.reranker import rerank, score_pairs, sigmoid
from schemas import RetrievedChunk
from tests.conftest import index_chunks, make_chunk
from tests.test_hybrid import CORPUS


def rc(cid, text, rank):
    return RetrievedChunk(chunk=make_chunk(cid, text), final_rank=rank)


def test_sigmoid_maps_logits_to_probabilities_and_cannot_overflow():
    assert sigmoid(0) == 0.5 and sigmoid(2.0) == pytest.approx(1 / (1 + math.exp(-2)))
    assert 0 < sigmoid(-1000) < 1e-20 and sigmoid(1000) == pytest.approx(1.0)


def test_rerank_reorders_by_the_cross_encoder_not_by_the_first_stage(fake_reranker):
    cands = [rc("a", "weather is sunny", 1), rc("b", "administrative fines for infringements", 2),
             rc("c", "fines", 3)]
    out = rerank("administrative fines", cands, top_k=2)
    assert [c.chunk.chunk_id for c in out] == ["b", "c"] and [c.final_rank for c in out] == [1, 2]
    assert out[0].rerank_score > out[1].rerank_score and out[0].rerank_prob == pytest.approx(sigmoid(out[0].rerank_score))
    assert cands[0].rerank_score < 0                       # an unrelated passage gets a negative logit


def test_rerank_sends_query_and_index_text_pairs(fake_reranker):
    cand = rc("a", "some text", 1)
    rerank("my question", [cand], top_k=1)
    assert fake_reranker.calls == [[("my question", cand.chunk.index_text)]]


def test_empty_candidates(fake_reranker):
    assert rerank("q", [], top_k=5) == [] and score_pairs([]) == [] and fake_reranker.calls == []


def test_reranker_is_loaded_once_per_name(monkeypatch):
    created = []

    class Dummy:
        def __init__(self, name, device=None, max_length=None):
            created.append((name, device, max_length))

    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", Dummy)
    monkeypatch.setattr(reranker, "_models", {})
    assert reranker.get_reranker("m1") is reranker.get_reranker("m1")
    assert created == [("m1", "cpu", 512)]


@pytest.fixture
def case(fake_embedder):
    index_chunks("c1", CORPUS)
    return "c1"


def test_retrieve_with_rerank_returns_final_k_scored_hits(case, fake_reranker):
    hits, timings = retrieve(case, "administrative fines up to 20 000 000 EUR", rerank=True, final_k=3, initial_k=8)
    assert len(hits) == 3 and hits[0].chunk.chunk_id == "c1"
    assert [h.final_rank for h in hits] == [1, 2, 3]
    assert all(h.rerank_score is not None and 0 < h.rerank_prob < 1 for h in hits)
    assert [h.rerank_score for h in hits] == sorted((h.rerank_score for h in hits), reverse=True)
    assert set(timings) == {"search_s", "rerank_s"}


def test_the_cross_encoder_scores_only_the_first_stage_candidates(case, fake_reranker):
    retrieve(case, "administrative fines", rerank=True, initial_k=4, final_k=2)
    assert len(fake_reranker.calls[0]) <= 4


def test_retrieve_without_rerank_keeps_first_stage_order(case, fake_reranker):
    first = search(case, "liability supplier", mode="hybrid")
    hits, _ = retrieve(case, "liability supplier", rerank=False, final_k=2)
    assert [h.chunk.chunk_id for h in hits] == [h.chunk.chunk_id for h in first[:2]]
    assert all(h.rerank_score is None for h in hits) and [h.final_rank for h in hits] == [1, 2]
    assert fake_reranker.calls == []


def test_rerank_setting_is_the_default(case, fake_reranker, monkeypatch):
    from config import get_settings
    monkeypatch.setenv("LEXORA_RERANK_ENABLED", "false")
    get_settings.cache_clear()
    hits, _ = retrieve(case, "administrative fines")
    assert hits and hits[0].rerank_score is None and fake_reranker.calls == []


def test_nothing_found_means_no_hits_and_no_model_call(case, fake_reranker):
    hits, _ = retrieve(case, "blockchain", mode="bm25", rerank=True)
    assert hits == [] and fake_reranker.calls == []
