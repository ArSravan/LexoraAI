"""Tests that use the REAL Hugging Face models.

They download about 90 MB once (then work offline). If the models cannot be loaded (no
internet and nothing cached) the whole module is skipped instead of failing.
"""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def real_embedder():
    from retrieval.embeddings import get_model
    try:
        return get_model()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"embedding model not available: {type(e).__name__}")


def test_real_embeddings_are_384_dim_unit_vectors_and_capture_meaning(real_embedder):
    from retrieval.embeddings import embed_texts
    v = embed_texts(["The cat sits on the mat.", "A feline is resting on a rug.",
                     "Stock markets fell sharply on Monday."], use_cache=False)
    assert v.shape == (3, 384) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)
    assert v[0] @ v[1] > v[0] @ v[2] + 0.2


@pytest.fixture(scope="module")
def real_reranker():
    from retrieval.reranker import get_reranker
    try:
        return get_reranker()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"reranker model not available: {type(e).__name__}")


def test_real_cross_encoder_prefers_the_relevant_passage(real_reranker):
    from retrieval.reranker import score_pairs
    q = "What is the capital of France?"
    good, bad = score_pairs([(q, "Paris is the capital and largest city of France."),
                             (q, "Bananas are a good source of potassium.")])
    assert good > bad + 3
