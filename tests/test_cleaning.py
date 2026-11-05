from LexoraAI.ingestion.pdf_parser import parse_pdf
from tests.conftest import make_pdf


def _pages(n, with_header=True):
    pages = []
    for i in range(1, n + 1):
        items = [(60, 700, f"Body text of page {i} with enough characters to count."),
                 (60, 680, f"Second body line of page {i}.")]
        if with_header:
            items += [(60, 815, f"Official Journal EN {i}"), (280, 25, f"- {i} -")]
        pages.append(items)
    return pages


def test_repeated_header_and_page_numbers_are_removed(tmp_path):
    doc = parse_pdf(make_pdf(tmp_path / "a.pdf", _pages(4)), "d")
    for page in doc.pages:
        texts = [l.text for l in page.lines]
        assert len(texts) == 2 and all("Body" in t or "Second" in t for t in texts)
        assert page.char_count == sum(len(t) for t in texts)
    assert any("Removed 8 repeated header/footer lines" in w for w in doc.warnings)


def test_body_text_is_never_removed(tmp_path):
    doc = parse_pdf(make_pdf(tmp_path / "b.pdf", _pages(5, with_header=False)), "d")
    assert all(len(p.lines) == 2 for p in doc.pages)
    assert not doc.warnings


def test_short_documents_are_left_alone(tmp_path):
    doc = parse_pdf(make_pdf(tmp_path / "c.pdf", _pages(2)), "d")
    assert all(len(p.lines) == 4 for p in doc.pages)


def test_a_lone_page_number_in_the_margin_is_removed_without_repetition(tmp_path):
    pages = _pages(4, with_header=False)
    pages[2].append((280, 25, "Page 3 of 4"))
    doc = parse_pdf(make_pdf(tmp_path / "d.pdf", pages), "d")
    assert all("Page 3 of 4" != l.text for l in doc.page(3).lines)
    assert len(doc.page(3).lines) == 2


def _page_with_footnotes():
    body = [(60, 760 - 18 * i, f"Body line {i} of the clause that talks about processing of data.") for i in range(8)]
    notes = [(60, 140, "1 Regulation (EU) 2016/679 of the European Parliament.", 7),
             (60, 128, "2 Directive 95/46/EC of the European Parliament.", 7)]
    return body + notes


def test_footnotes_are_flagged_and_body_is_not(tmp_path):
    doc = parse_pdf(make_pdf(tmp_path / "f.pdf", [_page_with_footnotes()]), "d")
    lines = doc.page(1).lines
    assert [l.is_footnote for l in lines] == [False] * 8 + [True] * 2
    assert lines[8].text.startswith("1 Regulation")


def test_small_text_without_a_note_number_is_not_a_footnote(tmp_path):
    items = _page_with_footnotes()[:8] + [(60, 140, "Small print without any note number at all.", 7)]
    doc = parse_pdf(make_pdf(tmp_path / "g.pdf", [items]), "d")
    assert not any(l.is_footnote for l in doc.page(1).lines)


def test_pages_without_footnotes_are_untouched(simple_pdf):
    doc = parse_pdf(simple_pdf, "d")
    assert not any(l.is_footnote for p in doc.pages for l in p.lines)
