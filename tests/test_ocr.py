import pymupdf
import pytest

from ingestion import pdf_parser
from ingestion.ocr import configure_tesseract
from ingestion.pdf_parser import PDFValidationError, parse_pdf
from tests.conftest import make_pdf

TEXT = ["Governing law of this agreement is the law of Delaware.",
        "The liability cap is limited to one million euros.",
        "Either party may terminate on thirty days written notice."]


@pytest.fixture(scope="module")
def tesseract_ok() -> bool:
    return configure_tesseract(None)


def _scan_of(tmp_path, name="scan.pdf", digital_first=True):
    """A PDF whose last page is a picture of text (a 'scan'), like a photocopied addendum."""
    src = make_pdf(tmp_path / "src.pdf", [[(60, 700 - 40 * i, t, 16) for i, t in enumerate(TEXT)]])
    with pymupdf.open(src) as s:
        png = tmp_path / "page.png"
        s[0].get_pixmap(dpi=200).save(png)
    out = tmp_path / name
    d = pymupdf.open()
    if digital_first:
        d.new_page().insert_text((60, 700), "This first page has a normal digital text layer for the parser.")
    d.new_page(width=595.28, height=841.89).insert_image(pymupdf.Rect(0, 0, 595.28, 841.89), filename=str(png))
    d.save(out)
    d.close()
    return out


def test_scanned_page_is_recovered_with_ocr(tmp_path, tesseract_ok):
    if not tesseract_ok:
        pytest.skip("Tesseract is not installed")
    doc = parse_pdf(_scan_of(tmp_path), "d")
    first, scan = doc.pages
    assert first.method == "digital" and not first.needs_ocr
    assert scan.needs_ocr and scan.method == "ocr"
    text = " ".join(l.text for l in scan.lines).lower()
    assert "governing law" in text and "delaware" in text and "liability cap" in text
    assert 0 < scan.ocr_confidence <= 100
    for l in scan.lines:
        assert l.method == "ocr" and l.page_index == 1 and 0 <= l.confidence <= 100
        x0, y0, x1, y1 = l.bbox                       # converted back to PDF points
        assert 0 <= x0 < x1 <= scan.width and 0 <= y0 < y1 <= scan.height
    assert "recovered with OCR" in scan.warning
    assert not doc.warnings                            # fully recovered: nothing left to warn about


def test_ocr_disabled_keeps_the_page_flagged_but_empty(tmp_path):
    doc = parse_pdf(_scan_of(tmp_path), "d", ocr_enabled=False)
    scan = doc.pages[1]
    assert scan.needs_ocr and scan.method == "digital" and scan.char_count == 0
    assert any("need OCR" in w for w in doc.warnings)


def test_ocr_failure_does_not_abort_the_document(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_parser, "configure_tesseract", lambda cmd=None: True)

    def boom(*a, **k):
        raise RuntimeError("tesseract crashed")
    monkeypatch.setattr(pdf_parser, "ocr_page", boom)
    doc = parse_pdf(_scan_of(tmp_path), "d")
    assert "OCR failed: tesseract crashed" in doc.pages[1].warning
    assert doc.pages[0].char_count > 0


def test_missing_tesseract_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_parser, "configure_tesseract", lambda cmd=None: False)
    doc = parse_pdf(_scan_of(tmp_path), "d")
    assert any("Tesseract not found" in w for w in doc.warnings)


def test_a_document_that_is_only_a_scan_is_rejected_when_ocr_is_off(tmp_path):
    pdf = _scan_of(tmp_path, digital_first=False)
    with pytest.raises(PDFValidationError, match="No text could be extracted"):
        parse_pdf(pdf, "d", ocr_enabled=False)
