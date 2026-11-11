"""One query path: the answer pipeline, the retrieval metrics, the calibration and every script give retrieval the same words."""
import pytest

from evaluation import calibrate, diagnostics
from retrieval import hybrid
from retrieval.synonyms import expand_query
from tests.conftest import index_chunks, make_chunk


@pytest.mark.parametrize("q", ["Which limit applies to vans?", "What does the GDPR say about fines?", "How do cars differ from vans?",
                               "BLEU for EN-DE and EN-FR", "What is a ZLEV?", "Plain question without any abbreviation"])
def test_expansion_is_idempotent_and_keeps_the_original_words(q):
    once = expand_query(q)
    assert once.startswith(q) and expand_query(once) == once


def test_expansion_adds_the_documents_wording_only_when_it_is_missing():
    assert expand_query("limits for vans") == "limits for vans (light commercial vehicles)"
    assert expand_query("limits for light commercial vehicles") == "limits for light commercial vehicles"
    assert expand_query("no known term here") == "no known term here"


def test_retrieve_finds_the_documents_wording_for_a_colloquial_question(fake_embedder, fake_reranker):
    """Only the vocabulary connects 'vans' to 'light commercial vehicles'; the distractor shares the word 'applies' with the question."""
    index_chunks("c1", [make_chunk("a", "Light commercial vehicles must not exceed 50 g of CO2 per kilometre.", clause="1", page=1),
                        make_chunk("b", "This clause applies to every notice sent under the Agreement.", clause="2", page=2),
                        make_chunk("c", "Notices must be delivered in writing to the registered address.", clause="3", page=3),
                        make_chunk("d", "This Agreement is governed by the laws of Germany.", clause="4", page=4)])
    hits, _ = hybrid.retrieve("c1", "What applies to vans?", mode="bm25", final_k=2, rerank=False, decompose_query=False)
    assert hits[0].chunk.chunk_id == "a"


def test_retrieve_called_twice_on_an_expanded_query_gives_the_same_result(fake_embedder, fake_reranker):
    index_chunks("c1", [make_chunk("a", "Light commercial vehicles must not exceed 50 g of CO2.", clause="1", page=1),
                        make_chunk("b", "This clause applies to every notice.", clause="2", page=2),
                        make_chunk("c", "Notices must be delivered in writing.", clause="3", page=3),
                        make_chunk("d", "This Agreement is governed by German law.", clause="4", page=4)])
    q = "What applies to vans?"
    first, _ = hybrid.retrieve("c1", q, mode="bm25", final_k=3, rerank=False, decompose_query=False)
    again, _ = hybrid.retrieve("c1", expand_query(q), mode="bm25", final_k=3, rerank=False, decompose_query=False)
    assert [h.chunk.chunk_id for h in first] == [h.chunk.chunk_id for h in again] and first[0].chunk.chunk_id == "a"


def test_the_gate_features_of_calibration_and_diagnostics_use_the_expanded_question(monkeypatch):
    seen = []
    from verification import evidence_checker

    class Q:
        id, answerable, question = "Q1", True, "What is the target for vans?"
        def evidence_dicts(self): return []

    monkeypatch.setattr(hybrid, "retrieve", lambda case, q, *a, **k: ([], {}))
    monkeypatch.setattr(evidence_checker, "assess", lambda q, hits, s: seen.append(q) or type("S", (), {"top_score": 1.0, "coverage": 1.0})())
    calibrate.collect_features("c", [Q()])
    diagnostics.features_with_gold("c", [Q()])
    assert seen == ["What is the target for vans? (light commercial vehicles)"] * 2
