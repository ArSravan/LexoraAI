import pytest
from pydantic import ValidationError

from LexoraAI.schemas import ParsedDocument, ParsedPage, TextLine


def _doc() -> ParsedDocument:
    line = TextLine(text="Article 1", page_index=0, bbox=(10, 20, 80, 32), size=11.0)
    return ParsedDocument(document_id="d1", filename="a.pdf", sha256="x", page_count=1,
                          pages=[ParsedPage(page_index=0, page_number=1, width=595, height=842,
                                            lines=[line], char_count=9)])


def test_json_round_trip_keeps_page_index_and_bbox():
    doc = _doc()
    again = ParsedDocument.model_validate_json(doc.model_dump_json())
    assert again == doc
    assert again.pages[0].lines[0].bbox == (10, 20, 80, 32)
    assert again.pages[0].lines[0].page_index == 0


def test_page_lookup_is_one_based():
    doc = _doc()
    assert doc.page(1).page_index == 0
    with pytest.raises(IndexError):
        doc.page(0)
    with pytest.raises(IndexError):
        doc.page(2)


def test_invalid_extraction_method_is_rejected():
    with pytest.raises(ValidationError):
        TextLine(text="x", page_index=0, bbox=(0, 0, 1, 1), method="magic")