"""Pydantic models shared across the pipeline.

Provenance rule: every piece of extracted text keeps its ORIGINAL 0-based PDF page index
and bounding box, so a citation can always be traced back to a place on a real page.
Models for chunks, retrieval and answers are added in the commits that need them.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]   # x0, y0, x1, y1 in PDF points, origin top-left


class TextLine(BaseModel):
    text: str
    page_index: int                         # 0-based, as in the PDF file
    bbox: BBox
    size: float = 0.0                       # mean font size (0 for OCR text)
    method: Literal["digital", "ocr"] = "digital"
    confidence: float | None = None         # OCR only, 0-100
    is_footnote: bool = False               # set by the footnote detector


class TableData(BaseModel):
    page_index: int
    bbox: BBox
    rows: list[list[str]]
    caption: str = ""
    header_rows: int = 1

    def to_text(self) -> str:
        """Pipe-separated rows. Each data row is followed by "(Header: value; ...)" so that a
        search for a column name still finds the cell value, even for spanning headers."""
        out = []
        header = self.rows[0] if self.rows else []
        for i, row in enumerate(self.rows):
            out.append(" | ".join(row))
            if i >= self.header_rows and header and len(header) == len(row):
                pairs = [f"{h}: {c}" for h, c in zip(header, row) if h and c]
                if pairs:
                    out.append("  (" + "; ".join(pairs) + ")")
        return "\n".join(out)


class ParsedPage(BaseModel):
    page_index: int                         # 0-based
    page_number: int                        # 1-based, what a human reads
    width: float
    height: float
    lines: list[TextLine] = Field(default_factory=list)
    tables: list[TableData] = Field(default_factory=list)
    char_count: int = 0
    needs_ocr: bool = False                 # no text layer but an image: the page is a scan
    method: Literal["digital", "ocr"] = "digital"   # "ocr" once OCR recovered the text
    ocr_confidence: float | None = None     # mean line confidence, 0-100
    warning: str = ""


class ParsedDocument(BaseModel):
    document_id: str
    filename: str
    sha256: str
    page_count: int
    metadata: dict[str, str] = Field(default_factory=dict)
    pages: list[ParsedPage] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def page(self, page_number: int) -> ParsedPage:
        """Return a page by its 1-based number."""
        if not 1 <= page_number <= len(self.pages):
            raise IndexError(f"page {page_number} not in 1..{len(self.pages)}")
        return self.pages[page_number - 1]


class Region(BaseModel):
    """A rectangle on one 0-based PDF page, in PDF points (used to highlight a quote later)."""
    page_index: int
    bbox: list[float]                       # x0, y0, x1, y1


class Chunk(BaseModel):
    """The unit that is embedded, searched, retrieved and cited."""
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int                        # 1-based page where the chunk starts
    page_index: int                         # the same page, 0-based (never lost)
    page_end: int = 0                       # last page the chunk spans (1-based)
    section: str = ""                       # heading context, e.g. "Article 83 - General conditions ..."
    article: str = ""                       # "83" / "Annex I" / contract section number
    clause: str = ""                        # "83(5)", "7.2", "Recital 4", "Table 9.1"
    text: str                               # what the LLM sees and what quotes are checked against
    chunk_type: str = "clause"              # clause|subpoint|child|recital|table|text|footnote
    bbox: list[float] | None = None         # bbox of the first region
    regions: list[Region] = Field(default_factory=list)
    extraction_method: str = "digital"      # digital|ocr
    ocr_confidence: float | None = None
    parent_clause: str = ""
    lead_in: str = ""                       # opening sentence of the parent clause (for sub-point chunks)
    index_text: str = ""                    # contextualised text used for embeddings / BM25
