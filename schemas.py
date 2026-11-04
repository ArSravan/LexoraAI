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


class ParsedPage(BaseModel):
    page_index: int                         # 0-based
    page_number: int                        # 1-based, what a human reads
    width: float
    height: float
    lines: list[TextLine] = Field(default_factory=list)
    char_count: int = 0
    needs_ocr: bool = False                 # (almost) no text but the page holds an image
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