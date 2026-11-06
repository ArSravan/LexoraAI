import numpy as np

from config import get_settings
from retrieval import embeddings
from retrieval.embeddings import embed_query, embed_texts

TEXTS = ["administrative fines up to 20 million", "the controller keeps records", "consent of the data subject"]


def test_shape_dtype_and_unit_length(fake_embedder):
    v = embed_texts(TEXTS)
    assert v.shape == (3, 64) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_empty_input_returns_an_empty_array(fake_embedder):
    assert embed_texts([]).shape == (0, 0) and fake_embedder.encoded == []


def test_similar_texts_have_a_higher_cosine_than_unrelated_ones(fake_embedder):
    a, b, c = embed_texts(["administrative fines for infringements", "fines for infringements",
                           "the weather is sunny today"])
    assert a @ b > 0.6 > a @ c


def test_second_call_is_served_from_the_cache(fake_embedder):
    first = embed_texts(TEXTS)
    assert len(fake_embedder.encoded) == 3
    second = embed_texts(TEXTS)
    assert len(fake_embedder.encoded) == 3                       # nothing was re-encoded
    assert np.array_equal(first, second)
    assert (get_settings().storage_dir / "cache" / "embeddings.sqlite").exists()


def test_only_new_texts_are_encoded_and_row_order_is_kept(fake_embedder):
    embed_texts([TEXTS[1]])                                      # warm one entry in the middle
    mixed = embed_texts(TEXTS)
    assert sorted(fake_embedder.encoded) == sorted([TEXTS[1], TEXTS[0], TEXTS[2]])
    assert np.array_equal(mixed, embed_texts(TEXTS, use_cache=False))


def test_cache_is_keyed_by_model_name(fake_embedder):
    embed_texts(TEXTS, model_name="model-a")
    embed_texts(TEXTS, model_name="model-b")
    assert len(fake_embedder.encoded) == 6


def test_use_cache_false_bypasses_the_cache_completely(fake_embedder):
    embed_texts(TEXTS, use_cache=False)
    embed_texts(TEXTS, use_cache=False)
    assert len(fake_embedder.encoded) == 6
    assert not (get_settings().storage_dir / "cache" / "embeddings.sqlite").exists()


def test_embed_query_returns_one_vector_and_is_not_cached(fake_embedder):
    q = embed_query("administrative fines")
    assert q.shape == (64,) and abs(np.linalg.norm(q) - 1) < 1e-5
    embed_query("administrative fines")
    assert len(fake_embedder.encoded) == 2


def test_progress_reaches_one(fake_embedder):
    seen = []
    embed_texts([f"text number {i}" for i in range(70)], batch_size=32, progress=seen.append)
    assert seen[-1] == 1.0 and seen == sorted(seen) and len(seen) == 3
    seen.clear()
    embed_texts([f"text number {i}" for i in range(70)], progress=seen.append)   # all cached
    assert seen == [1.0]


def test_model_is_loaded_once_per_name(monkeypatch):
    created = []

    class Dummy:
        def __init__(self, name, device=None):
            created.append((name, device))

    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", Dummy)
    monkeypatch.setattr(embeddings, "_models", {})
    a, b = embeddings.get_model("m1"), embeddings.get_model("m1")
    embeddings.get_model("m2")
    assert a is b and created == [("m1", "cpu"), ("m2", "cpu")]
