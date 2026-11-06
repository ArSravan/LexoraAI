"""Local sentence-transformers embeddings with a persistent SQLite cache.

An embedding turns a text into a vector so that texts with similar meaning point in similar
directions. Vectors are L2-normalised, so cosine similarity is a plain dot product.
Embedding is the slowest part of indexing, so every (model, text) pair is cached on disk:
re-indexing a document, or two documents sharing a sentence, never recomputes a vector.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from config import get_settings

_models: dict[str, object] = {}
_lock = threading.Lock()


def get_model(name: str | None = None):
    """Load a SentenceTransformer once per process (CPU). The first call downloads the model
    (about 90 MB for MiniLM) into the Hugging Face cache; afterwards it works offline."""
    s = get_settings()
    name = name or s.embedding_model
    with _lock:
        if name not in _models:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.manual_seed(s.seed)
            _models[name] = SentenceTransformer(name, device="cpu")
        return _models[name]


@contextmanager
def _cache() -> Iterator[sqlite3.Connection]:
    path = Path(get_settings().storage_dir) / "cache" / "embeddings.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=30)) as c:
        c.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
        yield c
        c.commit()


def _key(model: str, text: str) -> str:
    return hashlib.sha1(f"{model}\x00{text}".encode("utf-8")).hexdigest()


def embed_texts(texts: list[str], model_name: str | None = None, batch_size: int | None = None,
                use_cache: bool = True, progress: Callable[[float], None] | None = None) -> np.ndarray:
    """Return an (n, dim) float32 array of L2-normalised embeddings, row i for texts[i]."""
    s = get_settings()
    model_name = model_name or s.embedding_model
    batch_size = batch_size or s.embedding_batch_size
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    keys = [_key(model_name, t) for t in texts]
    out: list[np.ndarray | None] = [None] * len(texts)
    if use_cache:
        with _cache() as c:
            for i in range(0, len(keys), 500):            # SQLite allows ~999 variables per query
                part = keys[i:i + 500]
                rows = c.execute(f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(part))})", part).fetchall()
                found = {k: np.frombuffer(v, dtype=np.float32) for k, v in rows}
                for j, k in enumerate(part):
                    if k in found:
                        out[i + j] = found[k]
    todo = [i for i, v in enumerate(out) if v is None]
    if todo:
        model = get_model(model_name)
        new_rows = []
        for b in range(0, len(todo), batch_size):
            idx = todo[b:b + batch_size]
            vecs = model.encode([texts[i] for i in idx], batch_size=batch_size, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
            for i, v in zip(idx, vecs):
                out[i] = v
                new_rows.append((keys[i], v.tobytes()))
            if progress:
                progress(min(1.0, (b + len(idx)) / len(todo)))
        if use_cache and new_rows:
            with _cache() as c:
                c.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", new_rows)
    elif progress:
        progress(1.0)
    return np.vstack(out)


def embed_query(text: str, model_name: str | None = None) -> np.ndarray:
    """One query vector. Queries are not cached: they are cheap and rarely repeated."""
    return embed_texts([text], model_name, use_cache=False)[0]
