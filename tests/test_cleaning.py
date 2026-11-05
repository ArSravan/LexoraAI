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
