import numpy as np
import pytest

from retrieval import vector_store
from tests.conftest import make_chunk


def unit(*v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


CHUNKS = [make_chunk("a", "text a", doc="d1"), make_chunk("b", "text b", doc="d1"),
          make_chunk("c", "text c", doc="d2")]
VECS = np.stack([unit(1, 0, 0), unit(1, 1, 0), unit(0, 0, 1)])


def test_nearest_chunk_first_with_cosine_similarity():
    vector_store.add_chunks("c1", CHUNKS, VECS)
    hits = vector_store.query("c1", unit(1, 0, 0), top_k=3)
    assert [i for i, _ in hits] == ["a", "b", "c"]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-4)
    assert hits[1][1] == pytest.approx(0.7071, abs=1e-3)          # cos 45 degrees
    assert hits[2][1] == pytest.approx(0.0, abs=1e-4)


def test_top_k_larger_than_the_collection_is_fine():
    vector_store.add_chunks("c1", CHUNKS, VECS)
    assert len(vector_store.query("c1", unit(1, 0, 0), top_k=50)) == 3
    assert len(vector_store.query("c1", unit(1, 0, 0), top_k=2)) == 2


def test_empty_case_returns_nothing():
    assert vector_store.query("nothing-here", unit(1, 0, 0), top_k=5) == []
    assert vector_store.count("nothing-here") == 0


def test_filter_by_document():
    vector_store.add_chunks("c1", CHUNKS, VECS)
    assert [i for i, _ in vector_store.query("c1", unit(1, 0, 0), 5, document_ids=["d2"])] == ["c"]
    assert {i for i, _ in vector_store.query("c1", unit(1, 0, 0), 5, document_ids=["d1", "d2"])} == {"a", "b", "c"}


def test_upsert_overwrites_instead_of_duplicating():
    vector_store.add_chunks("c1", CHUNKS, VECS)
    vector_store.add_chunks("c1", CHUNKS, VECS)
    assert vector_store.count("c1") == 3


def test_delete_document_and_case():
    vector_store.add_chunks("c1", CHUNKS, VECS)
    vector_store.delete_document("c1", "d1")
    assert vector_store.count("c1") == 1
    vector_store.delete_case("c1")
    vector_store.delete_case("c1")                       # deleting twice is harmless
    assert vector_store.count("c1") == 0


def test_cases_are_isolated_from_each_other():
    vector_store.add_chunks("c1", CHUNKS, VECS)
    assert vector_store.query("c2", unit(1, 0, 0), 5) == []


def test_data_survives_a_new_client():
    vector_store.add_chunks("c1", CHUNKS, VECS)
    vector_store._clients.clear()                        # simulate a restart of the application
    assert vector_store.count("c1") == 3


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        vector_store.add_chunks("c1", CHUNKS, VECS[:2])
