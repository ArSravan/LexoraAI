import pytest

from config import get_settings
from retrieval import hybrid
from retrieval.hybrid import rrf_fuse, search
from tests.conftest import index_chunks, make_chunk

CORPUS = [
    make_chunk("c1", "Infringements shall be subject to administrative fines up to 20 000 000 EUR.", "83(5)", section="Article 83 - Fines"),
    make_chunk("c2", "Processing shall be lawful only if the data subject has given consent.", "6(1)", section="Article 6 - Lawfulness"),
    make_chunk("c3", "Nothing in this Section limits liability for gross negligence.", "7.2", doc="d2", name="Contract.pdf"),
    make_chunk("c4", "The supervisory authority shall be independent in the performance of its tasks.", "52(1)"),
    make_chunk("c5", "Personal data shall be accurate and, where necessary, kept up to date.", "5(1)(d)"),
    make_chunk("c6", "Each Member State shall provide for one or more independent public authorities.", "51(1)"),
    make_chunk("c7", "The controller shall keep records of processing activities under its responsibility.", "30(1)"),
    make_chunk("c8", "The supplier liability is capped at 10 000 000 EUR per year.", "7.1", doc="d2", name="Contract.pdf"),
]


@pytest.fixture
def case(fake_embedder):
    index_chunks("c1", CORPUS)
    return "c1"


def test_rrf_formula_is_exact():
    scores = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63) and scores["d"] == pytest.approx(1 / 63)
    assert rrf_fuse([["a"]], k=1) == {"a": pytest.approx(0.5)} and rrf_fuse([[], []]) == {}


def test_rrf_prefers_agreement_over_a_single_first_place():
    scores = rrf_fuse([["x", "m", "n"], ["y", "m", "z"]])           # m is 2nd in both lists
    assert scores["m"] > scores["x"] and scores["m"] > scores["y"]


def test_bm25_mode_records_only_bm25_evidence(case):
    hits = search(case, "administrative fines 20 000 000", mode="bm25")
    assert hits[0].chunk.chunk_id == "c1"
    h = hits[0]
    assert h.bm25_score > 0 and h.bm25_rank == 1 and h.dense_score is None and h.rrf_score is None
    assert [x.final_rank for x in hits] == list(range(1, len(hits) + 1))


def test_dense_mode_records_only_dense_evidence(case):
    hits = search(case, "administrative fines for infringements", mode="dense")
    assert hits[0].chunk.chunk_id == "c1"
    h = hits[0]
    assert 0 < h.dense_score <= 1.0001 and h.dense_rank == 1 and h.bm25_score is None and h.rrf_score is None
    assert [x.dense_score for x in hits] == sorted((x.dense_score for x in hits), reverse=True)


def test_hybrid_scores_follow_the_rrf_formula_and_order(case):
    hits = search(case, "liability cap of the supplier in EUR", mode="hybrid", rrf_k=60)
    assert hits and [h.rrf_score for h in hits] == sorted((h.rrf_score for h in hits), reverse=True)
    for h in hits:
        expected = sum(1 / (60 + r) for r in (h.bm25_rank, h.dense_rank) if r)
        assert h.rrf_score == pytest.approx(expected)
    assert {h.chunk.chunk_id for h in hits[:3]} >= {"c8", "c3"}          # both liability chunks lead
    by = {h.chunk.chunk_id: h for h in hits}
    # mirror-image ranks (BM25 1st / dense 3rd vs BM25 3rd / dense 1st) give EQUAL RRF scores
    assert by["c8"].rrf_score == pytest.approx(by["c1"].rrf_score)
    assert any(h.bm25_rank and h.dense_rank for h in hits)               # found by both retrievers


def test_a_chunk_found_by_only_one_retriever_still_appears_in_hybrid(case):
    hits = search(case, "Article 6(1)", mode="hybrid")
    assert hits[0].chunk.chunk_id == "c2"
    assert hits[0].bm25_rank == 1


def test_default_mode_comes_from_the_settings(case, monkeypatch):
    monkeypatch.setenv("LEXORA_RETRIEVAL_MODE", "bm25")
    get_settings.cache_clear()
    assert search(case, "administrative fines")[0].dense_score is None


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_document_filter_applies_to_every_mode(case, mode):
    hits = search(case, "liability shall data", mode=mode, document_ids=["d2"])
    assert hits and {h.chunk.document_id for h in hits} == {"d2"}


def test_top_k_and_dense_threshold(case):
    assert len(search(case, "shall data authority", mode="hybrid", top_k=3)) <= 3
    assert search(case, "administrative fines", mode="dense", dense_min_similarity=0.999) == []


def test_unknown_mode_and_empty_case(case):
    with pytest.raises(ValueError):
        search(case, "x", mode="magic")
    assert search("no-such-case", "anything") == []


def test_the_bm25_cache_is_rebuilt_when_chunks_change(case, fake_embedder):
    assert search(case, "blockchain ledger", mode="bm25") == []
    index_chunks(case, CORPUS + [make_chunk("c9", "A blockchain ledger records every transaction.", "9(1)")])
    assert search(case, "blockchain ledger", mode="bm25")[0].chunk.chunk_id == "c9"
    by_id, _ = hybrid.load_case_index(case)
    assert len(by_id) == 9
