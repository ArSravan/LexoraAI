import re

from LexoraAI.ingestion.chunker import chunk_document
from LexoraAI.ingestion.pdf_parser import parse_pdf
from LexoraAI.schemas import Chunk
from tests.conftest import make_pdf


def chunks_of(pdf, **kw):
    return chunk_document(parse_pdf(pdf, "doc1", **{k: v for k, v in kw.items() if k == "ocr_enabled"}),
                          **{k: v for k, v in kw.items() if k != "ocr_enabled"})


def by_clause(result):
    return {c.clause: c for c in result.chunks if c.clause}


def test_one_chunk_per_clause_with_subpoints_inside(eu_style_pdf):
    r = chunks_of(eu_style_pdf)
    assert r.structure["mode"] == "legal-structure"
    c = by_clause(r)
    para = c["3(2)"]
    assert para.chunk_type == "clause" and para.page_number == 2 and para.page_end == 2
    assert "20 000 000 EUR" in para.text and "(a) breach of the basic principles" in para.text
    assert "(b) breach of data subject rights" in para.text
    assert para.section == "Article 3 - Fines" and para.parent_clause == "3"
    assert "3(2)(a)" not in c                      # short clause: sub-points stay inside it


def test_index_text_carries_document_section_and_clause(eu_style_pdf):
    c = by_clause(chunks_of(eu_style_pdf))["3(2)"]
    assert c.index_text.startswith("Statute.pdf | Article 3 - Fines | clause 3(2)\n")
    assert c.index_text.endswith(c.text)


def test_recitals_become_recital_chunks(eu_style_pdf):
    c = by_clause(chunks_of(eu_style_pdf))
    assert c["Recital 1"].chunk_type == "recital" and "fundamental right" in c["Recital 1"].text
    assert c["Recital 2"].chunk_type == "recital"


def test_bare_article_headings_are_not_chunks(eu_style_pdf):
    for chunk in chunks_of(eu_style_pdf).chunks:
        assert chunk.text.strip() not in {"Article 1 Subject matter", "Article 3 Fines", "CHAPTER I General provisions"}


def test_ids_are_unique_sequential_and_boxes_exist(eu_style_pdf):
    r = chunks_of(eu_style_pdf)
    assert [c.chunk_id for c in r.chunks] == [f"doc1-{i:05d}" for i in range(len(r.chunks))]
    for c in r.chunks:
        assert c.page_number == c.page_index + 1 and c.page_end >= c.page_number
        assert c.bbox and len(c.bbox) == 4 and c.regions and c.document_name == "Statute.pdf"
    assert Chunk.model_validate_json(r.chunks[0].model_dump_json()) == r.chunks[0]


def test_long_clause_is_split_at_subpoints_and_keeps_its_lead_in(eu_style_pdf):
    r = chunks_of(eu_style_pdf, max_chars=150, overlap=30)
    c = by_clause(r)
    sub = c["3(2)(a)"]
    assert sub.chunk_type == "subpoint" and sub.parent_clause == "3(2)"
    assert "breach of the basic principles" in sub.text
    assert sub.lead_in.startswith("2. Serious infringements") and "4 % of annual turnover" in sub.lead_in
    assert "3(2)(b)" in c
    assert c["3(2)"].chunk_type == "clause" and "(a) breach" not in c["3(2)"].text   # lead-in chunk only


def test_long_paragraph_without_subpoints_becomes_overlapping_child_chunks(tmp_path):
    words = [f"Sentence number {i} of the very long obligation text continues here." for i in range(40)]
    lines = ["Article 5", "Obligations", "1."] + words + ["Article 6", "Scope", "Another short article body text."]
    pages = [[(60, 800 - 15 * i, t) for i, t in enumerate(lines[:23])],
             [(60, 800 - 15 * i, t) for i, t in enumerate(lines[23:])]]
    r = chunk_document(parse_pdf(make_pdf(tmp_path / "long.pdf", pages), "d"), max_chars=400, overlap=120)
    kids = [c for c in r.chunks if c.clause == "5(1)"]
    assert len(kids) > 3 and all(c.chunk_type == "child" for c in kids)
    assert all(len(c.text) <= 400 + 80 for c in kids)                       # max + at most one line
    for a, b in zip(kids, kids[1:]):                                       # consecutive pieces overlap
        last = re.findall(r"Sentence number \d+", a.text)[-1]
        assert last in b.text
    seen = {m for c in kids for m in re.findall(r"Sentence number (\d+)", c.text)}
    assert seen == {str(i) for i in range(40)}                              # nothing is lost
    assert any(c.page_end == 2 for c in kids)                               # a chunk can span pages


def test_table_becomes_its_own_chunk_with_caption_and_region(contract_pdf):
    r = chunk_document(parse_pdf(contract_pdf, "v"))
    t = by_clause(r)["Table 9.1"]
    assert t.chunk_type == "table" and t.text.startswith("Table 9.1 Indemnity obligations")
    assert "Data breach | 5,000,000" in t.text and t.section == "Section 9 - Indemnities"
    assert t.regions[0].page_index == 0 and t.regions[0].bbox[2] > t.regions[0].bbox[0]
    c = by_clause(r)
    assert "7.1" in c and "EUR 10,000,000" in c["7.1"].text and c["7.1"].parent_clause == "7"
    assert "5,000,000" not in c["9.1"].text           # cell text lives only in the table chunk


def test_footnotes_get_their_own_chunk_type(tmp_path):
    body = [(60, 780 - 18 * i, t) for i, t in enumerate(
        ["Article 9", "Processing", "1.", "Processing shall be lawful only if the data subject has given consent.",
         "2.", "Consent may be withdrawn at any time by the data subject.", "Closing line of the page body text."])]
    notes = [(60, 140, "1 Regulation (EU) 2016/679 of the European Parliament.", 7),
             (60, 128, "2 Directive 95/46/EC of the European Parliament.", 7)]
    r = chunk_document(parse_pdf(make_pdf(tmp_path / "n.pdf", [body + notes]), "d"))
    assert [c.chunk_type for c in r.chunks if c.chunk_type == "footnote"] == ["footnote"]
    assert "95/46/EC" not in by_clause(r)["9(2)"].text


def test_unstructured_text_uses_the_fallback_with_font_size_headings(tmp_path):
    page1 = [(60, 780, "Annual Overview", 18)] + [(60, 760 - 16 * i, f"Plain narrative sentence {i} about the year.") for i in range(6)]
    page2 = [(60, 780, "Outlook", 18)] + [(60, 760 - 16 * i, f"Another forward looking sentence {i}.") for i in range(4)]
    r = chunk_document(parse_pdf(make_pdf(tmp_path / "plain.pdf", [page1, page2]), "d"))
    assert r.structure["mode"] == "fallback-paragraph"
    assert {c.section for c in r.chunks} == {"Annual Overview", "Outlook"}
    assert all(c.chunk_type == "text" and c.clause == "" for c in r.chunks)
    assert [c.page_number for c in r.chunks] == [1, 2]
    assert "Annual Overview" not in r.chunks[0].text


def test_structure_summary_is_reported(eu_style_pdf):
    s = chunks_of(eu_style_pdf).structure
    assert s["n_articles"] == 4 and s["n_tables"] == 0 and s["n_clauses"] >= 8
    assert "3(2)(a)" in s["labels"] and any(h["type"] == "chapter" for h in s["headings"])
