"""PDF validation and text extraction.

Validation runs before anything is parsed: an upload must be a real, readable,
unencrypted PDF of acceptable size.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pymupdf


class PDFValidationError(ValueError):
    """The file is not an acceptable PDF. The message is safe to show to the user."""


def sha256_file(path: str | Path) -> str:
    """SHA-256 of the file content (used to detect duplicate uploads)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_filename(name: str) -> str:
    """Drop directory parts and unsafe characters (path-traversal guard); require .pdf."""
    name = Path(name.replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._\- ()]", "_", name).strip(" .")
    if not name.lower().endswith(".pdf"):
        raise PDFValidationError("Only .pdf files are accepted.")
    return name


def validate_pdf(path: str | Path, max_mb: float = 50) -> int:
    """Check type, size, header, readability and encryption. Returns the page count."""
    p = Path(path)
    if not p.is_file():
        raise PDFValidationError("File not found.")
    if p.suffix.lower() != ".pdf":
        raise PDFValidationError("Only .pdf files are accepted.")
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise PDFValidationError(f"File is {size_mb:.1f} MB; the limit is {max_mb:g} MB.")
    with open(p, "rb") as f:
        if not f.read(1024).lstrip().startswith(b"%PDF"):
            raise PDFValidationError("File content is not a PDF (missing %PDF header).")
    try:
        doc = pymupdf.open(p)
    except Exception as e:  # noqa: BLE001 - pymupdf raises several error types
        raise PDFValidationError(f"PDF cannot be opened: {e}") from e
    try:
        if doc.needs_pass:
            raise PDFValidationError("Encrypted/password-protected PDFs are not supported.")
        if doc.page_count < 1:
            raise PDFValidationError("PDF has no pages.")
        return doc.page_count
    finally:
        doc.close()