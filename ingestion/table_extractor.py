"""Table extraction with pdfplumber (tables drawn with ruling lines), keeping page and bbox."""
from __future__ import annotations

import pymupdf

from schemas import TableData


def _clean(cell) -> str:
    return " ".join(str(cell or "").split())


def extract_tables(pdf_path: str, page_indices: list[int]) -> dict[int, list[TableData]]:
    """Return {page_index: [TableData]} for the requested pages.

    Only tables with ruling lines are found (the "lines" strategy). Pages with fewer than
    4 drawn lines/rectangles are skipped without calling pdfplumber, which keeps long
    legal PDFs fast. Degenerate results (one row, one column, fewer than 4 filled cells)
    are discarded: they are usually layout artefacts, not tables.
    """
    import pdfplumber

    result: dict[int, list[TableData]] = {}
    with pymupdf.open(pdf_path) as fitz_doc, pdfplumber.open(pdf_path) as pdf:
        for pi in page_indices:
            if len(fitz_doc[pi].get_drawings()) < 4:
                continue
            try:
                found = pdf.pages[pi].find_tables({"vertical_strategy": "lines",
                                                   "horizontal_strategy": "lines"})
            except Exception:  # noqa: BLE001 - malformed page: skip its tables, keep its text
                continue
            for t in found:
                rows = [[_clean(c) for c in r] for r in (t.extract() or [])]
                rows = [r for r in rows if any(r)]
                if len(rows) < 2 or max((len(r) for r in rows), default=0) < 2:
                    continue
                if sum(1 for r in rows for c in r if c) < 4:
                    continue
                result.setdefault(pi, []).append(TableData(page_index=pi, bbox=tuple(t.bbox), rows=rows))
    return result
