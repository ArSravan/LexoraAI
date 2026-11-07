"""Render a PDF page as an image with the cited evidence highlighted.

Uploaded PDFs are never served as files; the evidence viewer only ever receives a rendered PNG.
  * yellow  the quoted text, located on the page by searching for it
  * blue    the region of the clause (the chunk's bounding box)
"""
from __future__ import annotations

import re

import pymupdf

YELLOW = (1.0, 0.85, 0.0)
BLUE = (0.1, 0.3, 0.9)


def _find_quote(page: "pymupdf.Page", quote: str) -> list["pymupdf.Rect"]:
    """Rectangles of the quote on the page. A long quote may wrap or contain a hyphenated word, so if the
    whole quote is not found we retry with shorter and shorter beginnings."""
    q = re.sub(r"\s+", " ", quote.replace("...", " ")).strip()
    for length in (len(q), 120, 80, 50, 30):
        probe = q[:length].rsplit(" ", 1)[0] if length < len(q) else q
        if len(probe) < 12:
            break
        hits = page.search_for(probe)
        if hits:
            return list(hits)
    return []


def render_page_png(pdf_path: str, page_index: int, zoom: float = 1.6, quote: str = "",
                    bbox: list[float] | None = None) -> tuple[bytes, dict]:
    """PNG bytes of one page (0-based index) plus {"matches": n, "highlighted": bool, "region": bool}.
    Raises IndexError for a page that does not exist."""
    with pymupdf.open(pdf_path) as doc:
        if not 0 <= page_index < doc.page_count:
            raise IndexError(f"page {page_index} out of range")
        page = doc[page_index]
        matches = _find_quote(page, quote) if quote else []
        for r in matches:
            page.draw_rect(r, color=YELLOW, fill=YELLOW, fill_opacity=0.4, overlay=True)
        if bbox and len(bbox) == 4:
            page.draw_rect(pymupdf.Rect(*bbox), color=BLUE, width=1.5, overlay=True)
        png = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).tobytes("png")   # nothing is saved
    return png, {"matches": len(matches), "highlighted": bool(matches), "region": bool(bbox and len(bbox) == 4)}
