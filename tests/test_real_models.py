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
