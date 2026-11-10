"""Indexing: parse -> chunk -> embed -> ChromaDB + chunks.json.

Two stores are written for every document:
  * ChromaDB       the vectors   (dense search)
  * chunks.json    the chunks    (BM25 is rebuilt from it in memory, and it is the source of the
                                  text shown to the LLM and of the page/bbox used for citations)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path
from typing import Callable

from config import get_settings
from ingestion import page_anomaly
from ingestion.chunker import ChunkResult, chunk_document
from ingestion.pdf_parser import PDFValidationError, parse_pdf
from retrieval import hybrid, vector_store
from retrieval.embeddings import embed_texts
from schemas import Chunk, ParsedDocument
from services import db
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


def index_is_current(case_id: str) -> bool:
    """True if the case has chunks AND every document in them has its metadata (page count, clause labels).

    An index built by an older version of the code lacks that metadata. Retrieval still works on such an index, but the
    verifier then knows no document, so it rejects every citation and every answer ends as "insufficient evidence"."""
    doc_ids = {r["document_id"] for r in _read_rows(case_id)}
    return bool(doc_ids) and all((doc_dir(case_id, d) / "meta.json").is_file() for d in doc_ids)


def index_fingerprint(case_id: str) -> str:
    """Identifies the exact content of a case's index. It changes whenever the case is indexed again, so results that
    were computed from an earlier index (cached answers, for example) can be recognised as out of date."""
    h = hashlib.sha1()
    f = chunks_file(case_id)
    if f.is_file():
        h.update(f.read_bytes())
    for meta in sorted((case_dir(case_id) / "docs").glob("*/meta.json")):
        h.update(meta.read_bytes())
    return h.hexdigest()[:16]


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


def pdf_path(case_id: str, doc_id: str) -> Path:
    """Where an uploaded PDF is kept. It is never served as a file: only rendered page images are."""
    return doc_dir(case_id, doc_id) / "original.pdf"


def load_parsed(case_id: str, doc_id: str) -> ParsedDocument | None:
    """The cached extraction (parsed.json) of a document, or None."""
    f = doc_dir(case_id, doc_id) / "parsed.json"
    return ParsedDocument.model_validate_json(f.read_text(encoding="utf-8")) if f.exists() else None


def load_anomaly_scores(case_id: str, doc_id: str) -> list[float] | None:
    """Per-page anomaly scores (> 1: the page looks unusual for this document), or None (short document or none computed)."""
    f = doc_dir(case_id, doc_id) / "anomaly.json"
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    except ValueError:
        return None


def _write_anomaly_scores(d: Path, parsed: ParsedDocument) -> None:
    scores = page_anomaly.score_document(parsed.pages)
    f = d / "anomaly.json"
    if scores is None:
        f.unlink(missing_ok=True)                      # a re-index of a now-short document must not keep old scores
    else:
        f.write_text(json.dumps(scores), encoding="utf-8")


def load_structure(case_id: str, doc_id: str) -> dict:
    """The detected structure summary (structure.json) of a document, or {}."""
    f = doc_dir(case_id, doc_id) / "structure.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def index_document(case_id: str, doc_id: str, force_reparse: bool = False,
                   progress: Progress | None = None) -> dict:
    """Index (or re-index) an uploaded document and keep its database record up to date.

    The extraction is cached in parsed.json, so re-indexing after changing the chunk size or the embedding
    model skips the slow part (OCR, tables); ``force_reparse`` redoes it (e.g. after installing Tesseract).
    Returns the updated database record; failures are stored in the record and re-raised.
    """
    s = get_settings()
    rec = db.get_document(doc_id)
    if rec is None:
        raise ValueError(f"Unknown document {doc_id}")
    prog = progress or (lambda f, m: None)
    d = doc_dir(case_id, doc_id)
    db.update_document(doc_id, status="indexing", error="")
    try:
        parsed = None if force_reparse else load_parsed(case_id, doc_id)
        if parsed is not None:
            prog(0.5, "Using cached extraction")
        else:
            parsed = parse_pdf(pdf_path(case_id, doc_id), doc_id, rec["filename"], max_mb=s.max_upload_mb,
                               scanned_min_chars=s.scanned_min_chars, ocr_enabled=s.ocr_enabled,
                               ocr_lang=s.ocr_lang, ocr_dpi=s.ocr_dpi, tesseract_cmd=s.tesseract_cmd or None,
                               tables_enabled=s.tables_enabled, progress=lambda f, m: prog(f * 0.5, m))
            (d / "parsed.json").write_text(parsed.model_dump_json(), encoding="utf-8")
        result = index_parsed(case_id, parsed, lambda f, m: prog(0.5 + 0.5 * f, m))
        (d / "structure.json").write_text(json.dumps(result.structure), encoding="utf-8")
        _write_anomaly_scores(d, parsed)
        scanned = [p for p in parsed.pages if p.needs_ocr]
        recovered = [p for p in scanned if p.method == "ocr"]
        db.update_document(
            doc_id, status="indexed", page_count=parsed.page_count, error="", warnings=parsed.warnings,
            ocr_status=f"OCR {len(recovered)}/{len(scanned)} scanned pages" if scanned else "n/a",
            n_sections=result.structure["n_sections"], n_clauses=result.structure["n_clauses"],
            n_chunks=len(result.chunks), n_tables=result.structure["n_tables"], n_scanned=len(scanned))
        return db.get_document(doc_id)
    except PDFValidationError as e:
        db.update_document(doc_id, status="failed", error=str(e))
        raise
    except Exception as e:  # noqa: BLE001 - recorded for the user, then re-raised for the job
        db.update_document(doc_id, status="failed", error=f"{type(e).__name__}: {e}")
        raise


def remove_document(case_id: str, document_id: str) -> None:
    with _write_lock:
        vector_store.delete_document(case_id, document_id)
        _write_rows(case_id, [r for r in _read_rows(case_id) if r["document_id"] != document_id])
    shutil.rmtree(doc_dir(case_id, document_id), ignore_errors=True)


def remove_case(case_id: str) -> None:
    vector_store.delete_case(case_id)
    hybrid.invalidate(case_id)
    shutil.rmtree(case_dir(case_id), ignore_errors=True)
