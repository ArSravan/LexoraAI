"""REST API. Uploaded PDFs are only ever exposed as rendered page images, never as raw files."""
from __future__ import annotations

import shutil
from functools import lru_cache

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from config import get_settings
from generation.ollama_client import OllamaClient
from ingestion import ocr as ocr_mod
from ingestion.pdf_parser import PDFValidationError, safe_filename, sha256_file, validate_pdf
from services import db, indexer, jobs
from services.paths import check_id

router = APIRouter()


@lru_cache(maxsize=1)
def _ocr_available() -> bool:
    return ocr_mod.configure_tesseract(get_settings().tesseract_cmd or None)


def _case_or_404(case_id: str) -> dict:
    try:
        check_id(case_id)
    except ValueError:
        raise HTTPException(400, "Invalid case id") from None
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    return case


def _doc_or_404(doc_id: str) -> dict:
    try:
        check_id(doc_id)
    except ValueError:
        raise HTTPException(400, "Invalid document id") from None
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


def _case_document_or_404(case_id: str, doc_id: str) -> dict:
    _case_or_404(case_id)
    doc = _doc_or_404(doc_id)
    if doc["case_id"] != case_id:
        raise HTTPException(404, "Document not in this case")
    return doc


# ------------------------------------------------------------------ status
@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/status")
def status():
    """Is the API up, is Ollama reachable and does it have the model, is OCR available?"""
    s = get_settings()
    return {"api": "ok", "ollama": OllamaClient.from_settings().status(),
            "settings": {"model": s.ollama_model, "embedding_model": s.embedding_model,
                         "reranker_model": s.reranker_model, "rerank_enabled": s.rerank_enabled,
                         "max_docs_per_case": s.max_docs_per_case, "max_upload_mb": s.max_upload_mb},
            "ocr_available": _ocr_available()}


@router.get("/stats")
def stats():
    return db.stats()


# ------------------------------------------------------------------ cases
class CaseIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)


@router.post("/cases")
def create_case(body: CaseIn):
    if not body.title.strip():
        raise HTTPException(422, "The title must not be blank.")
    return db.create_case(body.title, body.description)


@router.get("/cases")
def list_cases():
    return db.list_cases()


@router.get("/cases/{case_id}")
def get_case(case_id: str):
    case = _case_or_404(case_id)
    case["documents"] = db.list_documents(case_id)
    return case


@router.delete("/cases/{case_id}")
def delete_case(case_id: str):
    _case_or_404(case_id)
    indexer.remove_case(case_id)
    db.delete_case(case_id)
    return {"deleted": case_id}


# ------------------------------------------------------------------ documents
@router.post("/cases/{case_id}/documents")
async def upload_document(case_id: str, file: UploadFile = File(...)):
    """Validate and store an uploaded PDF, then index it in a background job.

    Every rejection removes what was written so far. 400 too many documents, 413 too large, 415 not a
    .pdf name, 422 not a readable PDF, 409 the same file is already in the case."""
    _case_or_404(case_id)
    s = get_settings()
    existing = db.list_documents(case_id)
    if len(existing) >= s.max_docs_per_case:
        raise HTTPException(400, f"A case can hold at most {s.max_docs_per_case} documents.")
    try:
        name = safe_filename(file.filename or "document.pdf")       # drops folders and unsafe characters
    except PDFValidationError as e:
        raise HTTPException(415, str(e)) from None
    doc_id = db.new_id("d")
    target = indexer.pdf_path(case_id, doc_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    limit, size = s.max_upload_mb * 1024 * 1024, 0
    with open(target, "wb") as f:                                   # stream to disk with a hard size cap
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                f.close()
                shutil.rmtree(target.parent, ignore_errors=True)
                raise HTTPException(413, f"File exceeds the {s.max_upload_mb} MB upload limit.")
            f.write(chunk)
    try:
        validate_pdf(target, s.max_upload_mb)
    except PDFValidationError as e:
        shutil.rmtree(target.parent, ignore_errors=True)
        raise HTTPException(422, str(e)) from None
    digest = sha256_file(target)
    if any(d["sha256"] == digest for d in existing):
        shutil.rmtree(target.parent, ignore_errors=True)
        raise HTTPException(409, "This exact file is already in the case (duplicate detected).")
    rec = db.add_document(case_id, name, digest, size, doc_id)
    job = jobs.start("index", lambda prog: indexer.index_document(case_id, doc_id, progress=prog),
                     {"case_id": case_id, "document_id": doc_id})
    return {"document": rec, "job_id": job}


@router.get("/cases/{case_id}/documents")
def list_documents(case_id: str):
    _case_or_404(case_id)
    return db.list_documents(case_id)


@router.delete("/cases/{case_id}/documents/{doc_id}")
def delete_document(case_id: str, doc_id: str):
    _case_document_or_404(case_id, doc_id)
    indexer.remove_document(case_id, doc_id)
    db.delete_document(doc_id)
    return {"deleted": doc_id}


@router.post("/cases/{case_id}/documents/{doc_id}/reindex")
def reindex_document(case_id: str, doc_id: str, force_reparse: bool = False):
    """Re-index a document (e.g. after changing chunk size). ``force_reparse`` also redoes extraction/OCR."""
    _case_document_or_404(case_id, doc_id)
    job = jobs.start("reindex", lambda prog: indexer.index_document(case_id, doc_id, force_reparse, prog),
                     {"case_id": case_id, "document_id": doc_id})
    return {"job_id": job}


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return job
