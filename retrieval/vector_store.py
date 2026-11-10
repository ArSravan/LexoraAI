"""Persistent ChromaDB vector store: one collection per case, deletion by document.

ChromaDB keeps the vectors and a little metadata on disk (``storage/chroma``). The chunk TEXT
used for BM25 and for the prompt lives in ``chunks.json`` (see services/indexer.py), so the
vector store only has to answer one question: "which chunk ids are closest to this vector?"
"""
from __future__ import annotations

import threading

import chromadb
import numpy as np

from config import get_settings
from schemas import Chunk

_clients: dict[str, "chromadb.api.ClientAPI"] = {}
_lock = threading.Lock()


def get_client():
    """One client per storage folder (tests use a fresh folder each, the app uses one)."""
    path = get_settings().storage_dir / "chroma"
    with _lock:
        key = str(path)
        if key not in _clients:
            path.mkdir(parents=True, exist_ok=True)
            _clients[key] = chromadb.PersistentClient(
                path=key, settings=chromadb.Settings(anonymized_telemetry=False))
        return _clients[key]


def close_clients() -> None:
    """Release every open client so the storage folder can be deleted (Windows refuses to remove open files)."""
    import gc
    from chromadb.api.shared_system_client import SharedSystemClient
    with _lock:
        _clients.clear()
        SharedSystemClient.clear_system_cache()      # stops chroma's internal system, which holds the index files
    gc.collect()


def collection_name(case_id: str) -> str:
    return f"case_{case_id}"


def _collection(case_id: str):
    # cosine space: distance = 1 - cosine similarity
    return get_client().get_or_create_collection(collection_name(case_id), metadata={"hnsw:space": "cosine"})


def add_chunks(case_id: str, chunks: list[Chunk], embeddings: np.ndarray, batch: int = 256) -> None:
    """Insert (or overwrite, by chunk id) the vectors of ``chunks``; row i of ``embeddings`` is chunks[i]."""
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    col = _collection(case_id)
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        col.upsert(
            ids=[c.chunk_id for c in part],
            embeddings=embeddings[i:i + batch].tolist(),
            documents=[c.text for c in part],
            metadatas=[{"document_id": c.document_id, "document_name": c.document_name,
                        "page_number": c.page_number, "section": c.section[:200], "article": c.article,
                        "clause": c.clause, "chunk_type": c.chunk_type, "case_id": case_id}
                       for c in part])


def query(case_id: str, query_embedding: np.ndarray, top_k: int,
          document_ids: list[str] | None = None) -> list[tuple[str, float]]:
    """[(chunk_id, cosine_similarity)] for the nearest chunks, best first."""
    col = _collection(case_id)
    n = col.count()
    if n == 0:
        return []
    where = {"document_id": {"$in": list(document_ids)}} if document_ids else None
    res = col.query(query_embeddings=[query_embedding.tolist()], n_results=min(top_k, n), where=where)
    return [(i, 1.0 - float(d)) for i, d in zip(res["ids"][0], res["distances"][0])]


def delete_document(case_id: str, document_id: str) -> None:
    _collection(case_id).delete(where={"document_id": document_id})


def delete_case(case_id: str) -> None:
    try:
        get_client().delete_collection(collection_name(case_id))
    except Exception:  # noqa: BLE001 - the collection may not exist
        pass


def count(case_id: str) -> int:
    return _collection(case_id).count()
