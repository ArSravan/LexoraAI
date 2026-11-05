from LexoraAI.ingestion.pdf_parser import parse_pdf
from LexoraAI.ingestion.structure_detector import StructureDetector
from tests.conftest import make_pdf


def detect(pdf):
    doc = parse_pdf(pdf, "d")
    det = StructureDetector()
    return det, det.run(doc)


def by_label(segs):
    return {s.label: s for s in segs if s.label}


def test_eu_style_labels(eu_style_pdf):
    det, segs = detect(eu_style_pdf)
    assert {"1", "2", "3", "4", "1(1)", "1(2)", "2(a)", "2(b)", "2(c)", "2(c)(i)", "2(c)(ii)",
            "3(1)", "3(2)", "3(2)(a)", "3(2)(b)", "Recital 1", "Recital 2"} <= det.labels


def test_eu_style_kinds_parents_and_context(eu_style_pdf):
    _, segs = detect(eu_style_pdf)
    s = by_label(segs)
    assert s["Recital 1"].kind == "recital"
    assert s["3(2)"].kind == "paragraph" and s["3(2)"].parent == "3"
    assert s["3(2)(a)"].kind == "subpoint" and s["3(2)(a)"].parent == "3(2)"
    assert s["2(c)(ii)"].kind == "subsubpoint" and s["2(c)(ii)"].parent == "2(c)"
    assert s["3(2)(a)"].section == "Article 3 - Fines"
    assert s["1(1)"].section == "Article 1 - Subject matter"
    assert "20 000 000 EUR" in s["3(2)"].text


def test_headings_outline_has_page_indexes(eu_style_pdf):
    det, _ = detect(eu_style_pdf)
    arts = {h["label"]: h["page_index"] for h in det.headings if h["type"] == "article"}
    assert arts == {"1": 0, "2": 1, "3": 1, "4": 2}
    assert any(h["type"] == "chapter" and h["label"] == "Chapter I" for h in det.headings)


def test_a_wrapped_line_that_looks_like_a_paragraph_marker_is_not_one(tmp_path):
    lines = ["Article 7", "Conditions", "1.", "The controller shall comply with the rules set out in",
             "5. of this Regulation and with every national law that applies.", "2.", "A second paragraph follows here."]
    pages = [[(60, 780 - 18 * i, t) for i, t in enumerate(lines)]]
    pages += [[(60, 700, f"Filler page {k} with enough text to be a real page.")] for k in (2, 3)]
    _, segs = detect(make_pdf(tmp_path / "w.pdf", pages))
    labels = [s.label for s in segs if s.label]
    assert "7(1)" in labels and "7(2)" in labels and "7(5)" not in labels
    assert "5. of this Regulation" in by_label(segs)["7(1)"].text


def test_letters_must_follow_the_alphabet(tmp_path):
    lines = ["Article 8", "Rules", "(a) first point;", "(c) not the next letter, so this is a continuation;",
             "(b) second point."]
    pages = [[(60, 780 - 18 * i, t) for i, t in enumerate(lines)]]
    pages += [[(60, 700, f"Filler page {k} with enough text to be a real page.")] for k in (2, 3)]
    det, _ = detect(make_pdf(tmp_path / "l.pdf", pages))
    assert {"8(a)", "8(b)"} <= det.labels and "8(c)" not in det.labels


def test_contract_sections_clauses_and_table(contract_pdf):
    det, segs = detect(contract_pdf)
    assert {"7", "7.1", "7.2", "9", "9.1", "9.2", "Table 9.1"} <= det.labels
    s = by_label(segs)
    assert s["7.1"].parent == "7" and s["7.1"].section == "Section 7 - Limitation of Liability"
    assert "EUR 10,000,000" in s["7.1"].text
    assert s["Table 9.1"].kind == "table" and s["Table 9.1"].table.rows[0] == ["Item", "Amount (EUR)"]
    assert s["Table 9.1"].section == "Section 9 - Indemnities"


def test_footnotes_are_separate_segments_and_do_not_pollute_the_clause(tmp_path):
    body = [(60, 780 - 18 * i, t) for i, t in enumerate(
        ["Article 9", "Processing", "1.", "Processing shall be lawful only if the data subject has given consent.",
         "2.", "Consent may be withdrawn at any time by the data subject.", "Closing line of the page body text."])]
    notes = [(60, 140, "1 Regulation (EU) 2016/679 of the European Parliament.", 7),
             (60, 128, "2 Directive 95/46/EC of the European Parliament.", 7)]
    _, segs = detect(make_pdf(tmp_path / "n.pdf", [body + notes]))
    foot = [s for s in segs if s.kind == "footnote"]
    assert len(foot) == 1 and "95/46/EC" in foot[0].text
    assert "Regulation (EU) 2016/679" not in by_label(segs)["9(2)"].text
