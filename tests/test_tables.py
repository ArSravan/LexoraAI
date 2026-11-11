from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ingestion.pdf_parser import parse_pdf
from schemas import TableData

ROWS = [["Item", "Amount (EUR)"], ["Data breach", "5,000,000"],
        ["IP infringement", "3,000,000"], ["Total", "8,000,000"]]


def table_pdf(path, ruled=True):
    st = getSampleStyleSheet()["Normal"]
    t = Table(ROWS)
    if ruled:
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.8, colors.black)]))
    SimpleDocTemplate(str(path), pagesize=A4).build([
        Paragraph("9. Indemnities. The supplier indemnifies the customer as set out below.", st),
        Spacer(1, 12), Paragraph("Table 9.1 Indemnity obligations", st), Spacer(1, 4), t, Spacer(1, 12),
        Paragraph("The amounts above apply per calendar year and are not cumulative.", st)])
    return path


def test_ruled_table_is_extracted_with_caption_and_bbox(tmp_path):
    doc = parse_pdf(table_pdf(tmp_path / "t.pdf"), "d")
    (t,) = doc.page(1).tables
    assert t.rows == ROWS
    assert t.caption.startswith("Table 9.1")
    x0, y0, x1, y1 = t.bbox
    assert 0 <= x0 < x1 <= doc.page(1).width and 0 <= y0 < y1 <= doc.page(1).height
    assert t.page_index == 0


def test_cell_text_is_removed_from_the_running_text(tmp_path):
    doc = parse_pdf(table_pdf(tmp_path / "t.pdf"), "d")
    texts = [l.text for l in doc.page(1).lines]
    assert not any("5,000,000" in x or "Data breach" in x for x in texts)
    assert any(x.startswith("9. Indemnities") for x in texts)           # text before the table stays
    assert any(x.startswith("The amounts above") for x in texts)        # text after the table stays
    assert doc.page(1).char_count == sum(len(x) for x in texts)


def test_tables_can_be_switched_off(tmp_path):
    doc = parse_pdf(table_pdf(tmp_path / "t.pdf"), "d", tables_enabled=False)
    assert doc.page(1).tables == []
    assert any("5,000,000" in l.text for l in doc.page(1).lines)


def test_text_laid_out_without_ruling_lines_is_not_a_table(tmp_path):
    doc = parse_pdf(table_pdf(tmp_path / "t.pdf", ruled=False), "d")
    assert doc.page(1).tables == []


def test_pages_without_drawings_have_no_tables(simple_pdf):
    assert all(p.tables == [] for p in parse_pdf(simple_pdf, "d").pages)


def test_to_text_repeats_the_header_for_each_data_row():
    text = TableData(page_index=0, bbox=(0, 0, 1, 1), rows=ROWS).to_text()
    assert text.splitlines()[0] == "Item | Amount (EUR)"
    assert "Data breach | 5,000,000" in text
    assert "(Item: Data breach; Amount (EUR): 5,000,000)" in text


def borderless_pdf(path):
    """A table without any ruling line, as papers set it: a caption above, cells placed by position."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    prose = "This sentence is long enough to run across the whole text column of the page like running text."
    page.insert_text((72, 60), prose, fontsize=10)
    page.insert_text((72, 90), "Table 2: Scores of three models on the test set.", fontsize=10)
    for x, h in ((100, "Model"), (250, "BLEU score"), (380, "Cost (FLOPs)")):
        page.insert_text((x, 120), h, fontsize=10)
    for i, row in enumerate((("Alpha net", "23.75", "1.0e20"), ("Beta net", "25.16", "9.6e18"), ("Gamma net", "28.40", "2.3e19"))):
        for x, cell in zip((100, 250, 380), row):
            page.insert_text((x, 140 + 12 * i), cell, fontsize=10)
    page.insert_text((72, 230), prose, fontsize=10)
    page.insert_text((72, 242), prose, fontsize=10)
    doc.save(str(path))
    return path


def test_a_table_without_ruling_lines_is_found_from_its_caption(tmp_path):
    doc = parse_pdf(borderless_pdf(tmp_path / "b.pdf"), "d")
    (t,) = doc.page(1).tables
    assert t.rows == [["Model", "BLEU score", "Cost (FLOPs)"], ["Alpha net", "23.75", "1.0e20"],
                      ["Beta net", "25.16", "9.6e18"], ["Gamma net", "28.40", "2.3e19"]]
    assert t.caption.startswith("Table 2:")
    texts = [l.text for l in doc.page(1).lines]
    assert not any("23.75" in x for x in texts) and any(x.startswith("This sentence") for x in texts)   # the prose stays


def test_text_that_only_mentions_a_table_is_not_a_table(tmp_path):
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    for i, line in enumerate(("Table 2 summarizes our results and compares them with earlier work in the literature today.",
                              "We estimate the number of operations by multiplying the training time and the GPUs used.")):
        page.insert_text((72, 60 + 12 * i), line, fontsize=10)
    doc.save(str(tmp_path / "m.pdf"))
    assert parse_pdf(tmp_path / "m.pdf", "d").page(1).tables == []
