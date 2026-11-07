import pymupdf
import pytest
from PIL import Image

from ingestion.pdf_parser import PDFValidationError, parse_pdf
from tests.conftest import make_pdf


def test_pages_text_and_numbering(simple_pdf):
    doc = parse_pdf(simple_pdf, "d1")
    assert doc.page_count == 2 and len(doc.pages) == 2
    assert [p.page_number for p in doc.pages] == [1, 2]
    assert [p.page_index for p in doc.pages] == [0, 1]
    assert [l.text for l in doc.page(1).lines] == [
        "Article 1", "This Regulation lays down rules on data protection."]
    assert "20 000 000 EUR" in doc.page(2).lines[1].text
    assert doc.filename == "simple.pdf" and len(doc.sha256) == 64


def test_every_line_keeps_page_index_and_a_bbox_inside_the_page(simple_pdf):
    doc = parse_pdf(simple_pdf, "d1")
    for page in doc.pages:
        for line in page.lines:
            x0, y0, x1, y1 = line.bbox
            assert line.page_index == page.page_index
            assert 0 <= x0 < x1 <= page.width and 0 <= y0 < y1 <= page.height


def test_reading_order_is_top_to_bottom(tmp_path):
    # drawn bottom-first on purpose; extraction must still return top-to-bottom
    pdf = make_pdf(tmp_path / "order.pdf", [[(60, 100, "third line of the page"), (60, 400, "second line of the page"),
                                             (60, 700, "first line of the page")]])
    assert [l.text.split()[0] for l in parse_pdf(pdf, "d").page(1).lines] == ["first", "second", "third"]


def test_list_marker_and_its_text_become_one_line(tmp_path):
    pdf = make_pdf(tmp_path / "marker.pdf", [[(60, 700, "(a)"), (90, 700, "controller means the body"),
                                              (60, 680, "(b)"), (90, 680, "processor means another body")]])
    lines = [l.text for l in parse_pdf(pdf, "d").page(1).lines]
    assert len(lines) == 2
    assert lines[0].startswith("(a)") and "controller" in lines[0]
    assert lines[1].startswith("(b)") and "processor" in lines[1]


def test_page_with_little_text_and_no_image_is_not_flagged_for_ocr(tmp_path):
    pdf = make_pdf(tmp_path / "mixed.pdf", [[(60, 700, "A page with plenty of readable text on it.")], []])
    doc = parse_pdf(pdf, "d")
    assert not doc.pages[0].needs_ocr
    assert not doc.pages[1].needs_ocr and "Blank page" in doc.pages[1].warning


def test_image_only_page_is_flagged_for_ocr(tmp_path):
    img = tmp_path / "scan.png"
    Image.new("RGB", (400, 200), "white").save(img)
    pdf = tmp_path / "scan.pdf"
    d = pymupdf.open()
    d.new_page().insert_text((60, 700), "This page has a normal digital text layer for the parser.")
    d.new_page().insert_image(pymupdf.Rect(50, 50, 450, 250), filename=str(img))
    d.save(pdf)
    d.close()
    doc = parse_pdf(pdf, "d")
    assert not doc.pages[0].needs_ocr
    assert doc.pages[1].needs_ocr and doc.pages[1].char_count == 0
    assert any("need OCR" in w for w in doc.warnings)


def test_pdf_without_any_text_or_image_is_rejected(tmp_path):
    pdf = make_pdf(tmp_path / "empty.pdf", [[]])
    with pytest.raises(PDFValidationError, match="No text"):
        parse_pdf(pdf, "d")


def test_invalid_input_is_rejected_by_validation(tmp_path):
    f = tmp_path / "x.pdf"
    f.write_bytes(b"not a pdf")
    with pytest.raises(PDFValidationError):
        parse_pdf(f, "d")


def test_parsed_document_survives_json_round_trip(simple_pdf):
    from schemas import ParsedDocument
    doc = parse_pdf(simple_pdf, "d1")
    assert ParsedDocument.model_validate_json(doc.model_dump_json()) == doc


def _column_items(prefix, x, n=12):
    return [(x, 760 - 16 * i, f"{prefix}{i:02d} text in this column continues here") for i in range(n)]


def test_two_columns_are_read_left_then_right(tmp_path):
    pdf = make_pdf(tmp_path / "cols.pdf", [_column_items("LEFT", 50) + _column_items("RIGHT", 320)])
    texts = [l.text for l in parse_pdf(pdf, "d").page(1).lines]
    assert [t.split()[0] for t in texts] == [f"LEFT{i:02d}" for i in range(12)] + [f"RIGHT{i:02d}" for i in range(12)]


def test_full_width_title_and_footer_are_kept_around_two_columns(tmp_path):
    items = ([(50, 800, "A TITLE THAT SPANS BOTH COLUMNS OF THE PAGE, FROM LEFT TO RIGHT EDGE")]
             + _column_items("LEFT", 50) + _column_items("RIGHT", 320)
             + [(50, 40, "A FOOTER THAT SPANS BOTH COLUMNS OF THE PAGE, FROM LEFT TO RIGHT EDGE")])
    pdf = make_pdf(tmp_path / "cols2.pdf", [items])
    heads = [l.text.split()[0] for l in parse_pdf(pdf, "d").page(1).lines]
    assert heads[0] == "A" and heads[-1] == "A"
    assert heads[1:-1] == [f"LEFT{i:02d}" for i in range(12)] + [f"RIGHT{i:02d}" for i in range(12)]
    assert len(heads) == 26


def test_single_column_page_is_not_mistaken_for_two_columns(tmp_path):
    items = [(60, 760 - 16 * i, f"Line {i:02d} of an ordinary single column page that is long enough to cross the middle")
             for i in range(10)]
    pdf = make_pdf(tmp_path / "one.pdf", [items])
    assert [l.text.split()[1] for l in parse_pdf(pdf, "d").page(1).lines] == [f"{i:02d}" for i in range(10)]


def test_progress_is_reported_per_page_and_ends_at_one(simple_pdf):
    seen = []
    parse_pdf(simple_pdf, "d", progress=lambda f, m: seen.append((f, m)))
    assert [m for _, m in seen[:2]] == ["Extracting text: page 1/2", "Extracting text: page 2/2"]
    assert [f for f, _ in seen] == sorted(f for f, _ in seen) and seen[-1] == (1.0, "Extraction complete")
