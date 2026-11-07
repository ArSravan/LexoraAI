"""PDF validation and text extraction.

Validation runs before anything is parsed: an upload must be a real, readable,
unencrypted PDF of acceptable size. Extraction then reads every page with PyMuPDF and
keeps, for each line, the ORIGINAL 0-based page index and bounding box.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable

import pymupdf

from ingestion.table_extractor import extract_tables
from ingestion.ocr import configure_tesseract, ocr_page
from ingestion.cleaning import mark_footnotes, strip_headers_footers
from schemas import ParsedDocument, ParsedPage, TextLine


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


_TABLE_CAPTION = re.compile(r"(?i)^table\s+\d")


def _attach_tables(pages: list[ParsedPage], path: Path, warnings: list[str]) -> None:
    """Extract tables from digital pages; move their cell text out of the page's lines.

    The caption is the nearest "Table N ..." line within 40 points above the table.
    """
    digital = [p.page_index for p in pages if p.method == "digital" and p.char_count]
    try:
        found = extract_tables(str(path), digital)
    except Exception as e:  # noqa: BLE001 - table failure must not lose the text
        warnings.append(f"Table extraction failed: {e}")
        return
    for pi, tables in found.items():
        page = pages[pi]
        for t in tables:
            x0, y0, x1, y1 = t.bbox
            page.lines = [l for l in page.lines
                          if not (x0 - 2 <= (l.bbox[0] + l.bbox[2]) / 2 <= x1 + 2
                                  and y0 - 2 <= (l.bbox[1] + l.bbox[3]) / 2 <= y1 + 2)]
            above = [l for l in page.lines if l.bbox[3] <= y0 + 2 and y0 - l.bbox[3] < 40
                     and _TABLE_CAPTION.match(l.text)]
            if above:
                t.caption = above[-1].text
            page.tables.append(t)
        page.char_count = sum(len(l.text) for l in page.lines)


def _apply_ocr(page: ParsedPage, pg: "pymupdf.Page", dpi: int, lang: str) -> None:
    """Replace an empty scanned page's lines with OCR output, recording confidence."""
    try:
        lines = ocr_page(pg, page.page_index, dpi, lang)
    except Exception as e:  # noqa: BLE001 - a failed OCR must not abort the whole document
        page.warning = f"Scanned page; OCR failed: {e}"
        return
    if not lines:
        page.warning = "Scanned page; OCR found no text."
        return
    page.lines = lines
    page.char_count = sum(len(l.text) for l in lines)
    page.method = "ocr"
    page.ocr_confidence = round(sum(l.confidence for l in lines) / len(lines), 1)
    page.warning = f"Scanned page; recovered with OCR (mean confidence {page.ocr_confidence}%)."


def parse_pdf(path: str | Path, document_id: str, filename: str | None = None, *,
              max_mb: float = 50, scanned_min_chars: int = 30, ocr_enabled: bool = True,
              ocr_lang: str = "eng", ocr_dpi: int = 200,
              tesseract_cmd: str | None = None, tables_enabled: bool = True,
              progress: Callable[[float, str], None] | None = None) -> ParsedDocument:
    """Validate and extract a PDF.

    A page with (almost) no text but an image is a scan: it is flagged ``needs_ocr`` and,
    when Tesseract is available, recovered with OCR (``method`` becomes "ocr"). Empty
    extraction is never silently accepted. ``progress(fraction, message)`` is called while pages are read.
    """
    path = Path(path)
    page_count = validate_pdf(path, max_mb)
    ocr_ok = ocr_enabled and configure_tesseract(tesseract_cmd)
    pages: list[ParsedPage] = []
    warnings: list[str] = []
    if ocr_enabled and not ocr_ok:
        warnings.append("Tesseract not found: scanned pages cannot be OCR'd (install Tesseract).")
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
                    if ocr_ok:
                        _apply_ocr(page, pg, ocr_dpi, ocr_lang)
                elif chars == 0:
                    page.warning = "Blank page (no text and no images)."
            pages.append(page)
            if progress:
                progress((i + 1) / page_count * 0.7, f"Extracting text: page {i + 1}/{page_count}")
        metadata = {k: str(v) for k, v in (doc.metadata or {}).items() if v}
    removed = strip_headers_footers(pages)
    if removed:
        warnings.append(f"Removed {removed} repeated header/footer lines.")
    for p in pages:
        mark_footnotes(p)
    if tables_enabled:
        if progress:
            progress(0.75, "Extracting tables")
        _attach_tables(pages, path, warnings)
    if progress:
        progress(1.0, "Extraction complete")
    total_chars = sum(p.char_count for p in pages) + sum(
        len(c) for p in pages for t in p.tables for r in t.rows for c in r)
    if total_chars < scanned_min_chars:
        raise PDFValidationError("No text could be extracted from this PDF (image-only PDF and OCR "
                                 "unavailable or empty). The document was not indexed.")
    n_missing = sum(p.needs_ocr and p.method != "ocr" for p in pages)
    if n_missing:
        warnings.append(f"{n_missing} page(s) have no text layer and need OCR.")
    return ParsedDocument(document_id=document_id, filename=filename or path.name,
                          sha256=sha256_file(path), page_count=page_count, metadata=metadata,
                          pages=pages, warnings=warnings)
