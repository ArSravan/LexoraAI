"""Indexing: parse -> chunk -> embed -> ChromaDB + chunks.json.

Two stores are written for every document:
  * ChromaDB       the vectors   (dense search)
  * chunks.json    the chunks    (BM25 is rebuilt from it in memory, and it is the source of the
                                  text shown to the LLM and of the page/bbox used for citations)
"""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Callable

from config import get_settings
from ingestion.chunker import ChunkResult, chunk_document
from ingestion.pdf_parser import parse_pdf
from retrieval import hybrid, vector_store
from retrieval.embeddings import embed_texts
from schemas import Chunk, ParsedDocument
from services.paths import case_dir, chunks_file, doc_dir

_write_lock = threading.Lock()
Progress = Callable[[float, str], None]


def _read_rows(case_id: str) -> list[dict]:
    f = chunks_file(case_id)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def _write_rows(case_id: str, rows: list[dict]) -> None:
    """Write atomically (temp file, then rename) and drop the cached BM25 index."""
    f = chunks_file(case_id)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows), encoding="utf-8")
    tmp.replace(f)
    hybrid.invalidate(case_id)


def _write_meta(case_id: str, parsed: ParsedDocument, result: ChunkResult) -> None:
    """What the citation validator later needs to know about a document: it exists, how many pages it
    has, and which clause labels were found in it."""
    d = doc_dir(case_id, parsed.document_id)
    d.mkdir(parents=True, exist_ok=True)
    meta = {"document_id": parsed.document_id, "filename": parsed.filename, "page_count": parsed.page_count,
            "sha256": parsed.sha256, "labels": result.structure["labels"], "n_chunks": len(result.chunks),
            "warnings": parsed.warnings}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def case_documents(case_id: str) -> list[dict]:
    """The metadata of every indexed document of a case (sorted by file name)."""
    metas = [json.loads(f.read_text(encoding="utf-8")) for f in (case_dir(case_id) / "docs").glob("*/meta.json")]
    return sorted(metas, key=lambda m: m["filename"].lower())


def case_chunks(case_id: str) -> list[Chunk]:
    return [Chunk(**r) for r in _read_rows(case_id)]


def index_parsed(case_id: str, parsed: ParsedDocument, progress: Progress | None = None) -> ChunkResult:
    """Chunk and embed one parsed document and (re)place it in the case's indexes."""
    s = get_settings()
    prog = progress or (lambda f, m: None)
    prog(0.1, "Detecting legal structure and chunking")
    result = chunk_document(parsed, s.chunk_max_chars, s.chunk_overlap)
    chunks = result.chunks
    prog(0.3, f"Embedding {len(chunks)} chunks")
    emb = embed_texts([c.index_text for c in chunks], progress=lambda f: prog(0.3 + 0.6 * f, "Embedding chunks"))
    with _write_lock:
        vector_store.delete_document(case_id, parsed.document_id)       # re-indexing replaces, never duplicates
        if chunks:
            vector_store.add_chunks(case_id, chunks, emb)
        rows = [r for r in _read_rows(case_id) if r["document_id"] != parsed.document_id]
        rows.extend(c.model_dump() for c in chunks)
        _write_rows(case_id, rows)
        _write_meta(case_id, parsed, result)
    prog(1.0, "Indexed")
    return result


def index_pdf(case_id: str, document_id: str, pdf: str | Path, filename: str | None = None,
              progress: Progress | None = None) -> ChunkResult:
    """Parse a PDF with the application settings, then index it."""
    s = get_settings()
    parsed = parse_pdf(pdf, document_id, filename, max_mb=s.max_upload_mb, scanned_min_chars=s.scanned_min_chars,
                       ocr_enabled=s.ocr_enabled, ocr_lang=s.ocr_lang, ocr_dpi=s.ocr_dpi,
                       tesseract_cmd=s.tesseract_cmd or None, tables_enabled=s.tables_enabled)
    return index_parsed(case_id, parsed, progress)


def remove_document(case_id: str, document_id: str) -> None:
    with _write_lock:
        vector_store.delete_document(case_id, document_id)
        _write_rows(case_id, [r for r in _read_rows(case_id) if r["document_id"] != document_id])
    shutil.rmtree(doc_dir(case_id, document_id), ignore_errors=True)


def remove_case(case_id: str) -> None:
    vector_store.delete_case(case_id)
    hybrid.invalidate(case_id)
    shutil.rmtree(case_dir(case_id), ignore_errors=True)
