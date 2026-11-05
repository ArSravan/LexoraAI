"""PDF validation and text extraction.

Validation runs before anything is parsed: an upload must be a real, readable,
unencrypted PDF of acceptable size. Extraction then reads every page with PyMuPDF and
keeps, for each line, the ORIGINAL 0-based page index and bounding box.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pymupdf

from LexoraAI.schemas import ParsedDocument, ParsedPage, TextLine


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


# ---------------------------------------------------------------------------- extraction
def _page_lines(page: "pymupdf.Page", page_index: int) -> list[TextLine]:
    """Text lines of one page in reading order, with bounding boxes and mean font size."""
    raw: list[TextLine] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:          # 0 = text block; skip image blocks
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in line["spans"]).replace("\u00a0", " ")
            text = re.sub(r"[ \t]+", " ", text).strip()
            if text:
                size = sum(s["size"] for s in spans) / len(spans)
                raw.append(TextLine(text=text, page_index=page_index,
                                    bbox=tuple(round(v, 2) for v in line["bbox"]), size=size))
    return _order_lines(raw, page.rect.width)


def _merge_rows(lines: list[TextLine]) -> list[TextLine]:
    """Sort top-to-bottom and join segments sitting on the same visual row.

    A list marker such as "(a)" is often a separate segment from its text; after this
    step "(a)" and "the controller means..." are one line.
    """
    rows: list[list[TextLine]] = []
    for ln in sorted(lines, key=lambda l: (l.bbox[1], l.bbox[0])):
        yc = (ln.bbox[1] + ln.bbox[3]) / 2
        if rows:
            top = min(x.bbox[1] for x in rows[-1])
            bottom = max(x.bbox[3] for x in rows[-1])
            same_height = (bottom - top) < 2.5 * (ln.bbox[3] - ln.bbox[1] + 1)
            if top - 1 <= yc <= bottom + 1 and same_height:
                rows[-1].append(ln)
                continue
        rows.append([ln])
    merged = []
    for row in rows:
        row.sort(key=lambda x: x.bbox[0])
        bbox = (min(x.bbox[0] for x in row), min(x.bbox[1] for x in row),
                max(x.bbox[2] for x in row), max(x.bbox[3] for x in row))
        merged.append(TextLine(text=" ".join(x.text for x in row), page_index=row[0].page_index,
                               bbox=tuple(round(v, 2) for v in bbox),
                               size=sum(x.size for x in row) / len(row)))
    return merged


def _order_lines(lines: list[TextLine], width: float) -> list[TextLine]:
    """Reading order of a page.

    One column: top to bottom. Two columns (detected from the x positions of the lines):
    the whole left column, then the whole right column. Lines that span both columns
    (a title, a full-width footer) act as separators: the page is read band by band.
    """
    if not lines:
        return lines
    mid = width / 2

    def side(ln: TextLine) -> str:
        if ln.bbox[2] <= mid + 5:
            return "L"
        if ln.bbox[0] >= mid - 5:
            return "R"
        return "S"                      # straddles the middle

    sides = [side(l) for l in lines]
    n = len(lines)
    two_col = (sides.count("L") >= 0.3 * n and sides.count("R") >= 0.3 * n
               and sides.count("S") <= 0.1 * n)
    if not two_col:
        return _merge_rows(lines)
    out: list[TextLine] = []
    left: list[TextLine] = []
    right: list[TextLine] = []
    for ln in sorted(lines, key=lambda l: (l.bbox[1], l.bbox[0])):
        s = side(ln)
        if s == "S":
            out += _merge_rows(left) + _merge_rows(right) + [ln]
            left, right = [], []
        elif s == "L":
            left.append(ln)
        else:
            right.append(ln)
    return out + _merge_rows(left) + _merge_rows(right)


def parse_pdf(path: str | Path, document_id: str, filename: str | None = None, *,
              max_mb: float = 50, scanned_min_chars: int = 30) -> ParsedDocument:
    """Validate and extract a PDF. Pages with (almost) no text but an image are flagged
    ``needs_ocr``; the OCR step itself is a separate stage."""
    path = Path(path)
    page_count = validate_pdf(path, max_mb)
    pages: list[ParsedPage] = []
    warnings: list[str] = []
    with pymupdf.open(path) as doc:
        for i in range(page_count):
            pg = doc[i]
            lines = _page_lines(pg, i)
            chars = sum(len(l.text) for l in lines)
            page = ParsedPage(page_index=i, page_number=i + 1, width=pg.rect.width,
                              height=pg.rect.height, lines=lines, char_count=chars)
            if chars < scanned_min_chars:
                if pg.get_images(full=True):
                    page.needs_ocr = True
                    page.warning = "Image-only page: no text layer, OCR needed."
                elif chars == 0:
                    page.warning = "Blank page (no text and no images)."
            pages.append(page)
        metadata = {k: str(v) for k, v in (doc.metadata or {}).items() if v}
    if not any(p.char_count or p.needs_ocr for p in pages):
        raise PDFValidationError("No text could be extracted from this PDF.")
    n_ocr = sum(p.needs_ocr for p in pages)
    if n_ocr:
        warnings.append(f"{n_ocr} page(s) have no text layer and need OCR.")
    return ParsedDocument(document_id=document_id, filename=filename or path.name,
                          sha256=sha256_file(path), page_count=page_count, metadata=metadata,
                          pages=pages, warnings=warnings)
