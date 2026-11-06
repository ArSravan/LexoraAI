import pytest

from config import get_settings
from retrieval.hybrid import _interleave, decompose, retrieve
from schemas import RetrievedChunk
from tests.conftest import index_chunks, make_chunk
from tests.test_hybrid import CORPUS


def test_a_simple_question_is_not_split():
    q = "What is the maximum administrative fine for infringements of the basic principles?"
    assert decompose(q) == [q]
    assert decompose("What are the terms and conditions of the contract?") == ["What are the terms and conditions of the contract?"]


@pytest.mark.parametrize("question,expected", [
    ("What is the liability cap in Section 7, and which indemnities does Table 9.1 list?",
     ["What is the liability cap in Section 7", "which indemnities does Table 9.1 list"]),
    ("What are the fines under Article 83? Who is the controller?",
     ["What are the fines under Article 83", "Who is the controller"]),
    ("What fines apply and what is the cap", ["What fines apply", "what is the cap"]),
    ("Which clause covers fines; which clause covers consent", ["Which clause covers fines", "which clause covers consent"]),
])
def test_compound_questions_are_split(question, expected):
    assert decompose(question) == expected


def test_tiny_fragments_do_not_count_as_parts_and_parts_are_capped():
    assert decompose("Fines; consent") == ["Fines; consent"]
    many = "What is alpha clause; what is beta clause; what is gamma clause; what is delta clause"
    assert len(decompose(many)) == 3 and len(decompose(many, max_parts=2)) == 2


def rc(cid):
    return RetrievedChunk(chunk=make_chunk(cid, cid))


def test_interleave_is_round_robin_without_repeats():
    a, b = [rc("a1"), rc("a2"), rc("a3")], [rc("b1"), rc("a1"), rc("b3")]
    assert [h.chunk.chunk_id for h in _interleave([a, b], 4)] == ["a1", "b1", "a2", "a3"]
    assert _interleave([[], []], 3) == [] and len(_interleave([a], 10)) == 3


@pytest.fixture
def case(fake_embedder):
    index_chunks("c1", CORPUS)
    return "c1"


QUESTION = "What fines apply to infringements, and what liability cap does the supplier have?"


def test_each_half_of_a_two_part_question_keeps_its_evidence(case, fake_reranker):
    hits, timings = retrieve(case, QUESTION, rerank=True, final_k=3)
    ids = [h.chunk.chunk_id for h in hits]
    assert "c1" in ids and "c8" in ids                      # fines clause AND liability cap
    assert timings["sub_queries"] == ["What fines apply to infringements", "what liability cap does the supplier have"]
    assert len(fake_reranker.calls) == 3                    # full question + 2 sub-queries
    assert [h.final_rank for h in hits] == [1, 2, 3] and all(h.rerank_score is not None for h in hits)


def test_rerank_score_is_the_best_score_over_all_queries(case, monkeypatch):
    """A focused sub-query can match a passage better than the long full question does."""
    import numpy as np
    from retrieval import reranker
    from textutil import stem, words

    class Focused:                                   # share of the QUERY's words found in the passage
        def predict(self, pairs, batch_size=16, show_progress_bar=False):
            out = []
            for q, p in pairs:
                qs, ps = [stem(w) for w in words(q)], {stem(w) for w in words(p)}
                out.append(10.0 * sum(w in ps for w in qs) / len(qs) - 1.0)
            return np.array(out, dtype=np.float32)

    monkeypatch.setattr(reranker, "get_reranker", lambda name=None: Focused())
    hits, timings = retrieve(case, QUESTION, rerank=True, final_k=3)
    model = Focused()
    queries = [QUESTION] + timings["sub_queries"]
    for h in hits:
        per_query = [float(model.predict([(q, h.chunk.index_text)])[0]) for q in queries]
        assert h.rerank_score == pytest.approx(max(per_query))
    c8 = next(h for h in hits if h.chunk.chunk_id == "c8")
    assert c8.rerank_score > float(model.predict([(QUESTION, c8.chunk.index_text)])[0])   # a sub-query won


def test_without_reranking_the_first_stage_lists_are_interleaved(case, fake_reranker):
    hits, timings = retrieve(case, QUESTION, rerank=False, final_k=3)
    assert len(hits) == 3 and fake_reranker.calls == [] and len(timings["sub_queries"]) == 2
    assert all(h.rerank_score is None for h in hits)


def test_decomposition_can_be_switched_off(case, fake_reranker, monkeypatch):
    hits, timings = retrieve(case, QUESTION, rerank=True, decompose_query=False)
    assert timings["sub_queries"] == [] and len(fake_reranker.calls) == 1
    monkeypatch.setenv("LEXORA_QUERY_DECOMPOSITION", "false")
    get_settings.cache_clear()
    assert retrieve(case, QUESTION, rerank=True)[1]["sub_queries"] == []


def test_simple_questions_take_the_single_query_path(case, fake_reranker):
    hits, timings = retrieve(case, "administrative fines for infringements", rerank=True)
    assert timings["sub_queries"] == [] and len(fake_reranker.calls) == 1 and hits[0].chunk.chunk_id == "c1"
