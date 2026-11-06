import pytest

from ingestion.chunker import chunk_document
from ingestion.pdf_parser import parse_pdf
from schemas import RetrievedChunk
from verification.conflict_detector import compare_amounts, detect_conflicts, extract_facts
from tests.conftest import make_chunk


def hit(text, doc="d1", name="A.pdf", clause="1", page=1, chunk_type="clause", cid=None):
    c = make_chunk(cid or f"{doc}-{clause}", text, clause=clause, doc=doc, name=name, page=page)
    c.chunk_type = chunk_type
    return RetrievedChunk(chunk=c)


def facts(text, **kw):
    return [(f.kind, f.value) for f in extract_facts(hit(text, **kw))]


# ------------------------------------------------------------------------------ extraction
@pytest.mark.parametrize("text,value", [
    ("fines of EUR 20 million", 20e6), ("fines up to 20 000 000 EUR", 20e6), ("a fee of €5m", 5e6),
    ("a loan of USD 1.5 billion", 1.5e9), ("pay 10,000,000 euros", 10e6), ("a cap of EUR 10,000,000.", 10e6),
])
def test_amounts_in_many_notations(text, value):
    assert facts(text) == [("amount", value)]


def test_percentages_and_dates_and_display():
    assert facts("up to 4 % of turnover, or 2,5 per cent") == [("percentage", 4.0), ("percentage", 2.5)]
    assert facts("in force on 25 May 2018 and until 2020-12-31") == [("date", "2018-05-25"), ("date", "2020-12-31")]
    assert extract_facts(hit("a cap of EUR 10,000,000"))[0].display == "EUR 10,000,000"


def test_words_that_merely_contain_a_currency_code_are_not_money():
    assert facts("The Europe 2020 strategy and 2020 Europe summit") == []
    assert facts("No figures here at all.") == []


def test_table_cells_take_their_currency_from_the_header(contract_pdf):
    chunks = chunk_document(parse_pdf(contract_pdf, "v"), 1000, 150).chunks
    table = next(RetrievedChunk(chunk=c) for c in chunks if c.chunk_type == "table")
    amounts = {(f.sentence, f.display) for f in extract_facts(table) if f.kind == "amount"}
    assert ("Total: 8,000,000", "EUR 8,000,000") in amounts and ("Data breach: 5,000,000", "EUR 5,000,000") in amounts


# ------------------------------------------------------------------------------ conflicts
REV_A = "Total revenue for FY2025 was EUR 120 million."
REV_B = "Total revenue for FY2025 was EUR 150 million."


def test_two_documents_with_different_figures_for_the_same_topic_conflict():
    out = detect_conflicts([hit(REV_A, "d1", "Report.pdf", "4.1", 12), hit(REV_B, "d2", "Board.pdf", "2.3", 3)])
    assert len(out) == 1 and out[0].classification == "CONFLICT" and out[0].kind == "amount"
    assert out[0].values == ["EUR 120,000,000", "EUR 150,000,000"] and "revenu" in out[0].topic
    assert [s.document for s in out[0].sources] == ["Report.pdf", "Board.pdf"] and out[0].sources[0].page == 12
    assert "neither value is chosen automatically" in out[0].explanation


def test_a_restatement_is_an_explained_difference_not_a_conflict():
    restated = "Total revenue for FY2025 was restated to EUR 150 million."
    out = detect_conflicts([hit(REV_A, "d1", "Report.pdf"), hit(restated, "d2", "Board.pdf")])
    assert len(out) == 1 and out[0].classification == "EXPLAINED_DIFFERENCE" and "restated" in out[0].explanation


def test_nothing_is_reported_when_there_is_no_real_disagreement():
    assert detect_conflicts([hit(REV_A, "d1"), hit(REV_A, "d2", "B.pdf")]) == []                       # same value
    assert detect_conflicts([hit(REV_A, "d1"), hit("The fine is EUR 20 million for data breaches.", "d2")]) == []  # other topic
    assert detect_conflicts([hit(REV_A, "d1", clause="1"), hit(REV_B, "d1", clause="2")]) == []        # same document
    assert detect_conflicts([hit(REV_A, "d1")]) == [] and detect_conflicts([]) == []


def test_within_one_document_only_a_qualifier_makes_a_difference_worth_reporting():
    out = detect_conflicts([hit(REV_A, "d1", clause="1"),
                            hit("Total revenue for FY2025 was amended to EUR 150 million.", "d1", clause="2")])
    assert [c.classification for c in out] == ["EXPLAINED_DIFFERENCE"]


def test_dates_conflict_only_across_documents():
    a = hit("The agreement enters into force on 1 January 2025.", "d1", "A.pdf")
    b = hit("The agreement enters into force on 1 March 2025.", "d2", "B.pdf")
    assert [c.kind for c in detect_conflicts([a, b])] == ["date"]
    c = hit("The agreement enters into force on 1 March 2025.", "d1", "A.pdf", clause="9")
    assert detect_conflicts([a, c]) == []


def test_hard_conflicts_come_first_and_there_are_at_most_five():
    explained = [hit(f"Total revenue for FY2025 was restated to EUR {200 + i} million.", "d2", "B.pdf", clause=str(i)) for i in range(7)]
    out = detect_conflicts([hit(REV_A, "d1")] + explained + [hit(REV_B, "d3", "C.pdf")])
    assert len(out) == 5 and out[0].classification == "CONFLICT"


# ------------------------------------------------------------------------------ numeric comparison
@pytest.fixture
def contract_hits(contract_pdf):
    return [RetrievedChunk(chunk=c) for c in chunk_document(parse_pdf(contract_pdf, "v"), 1000, 150).chunks]


Q = "Does the liability cap in Section 7 exceed the total indemnity obligations in Section 9?"


def test_the_cap_in_section_7_is_compared_with_the_total_in_section_9(contract_hits):
    (cmp,) = compare_amounts(Q, contract_hits)
    assert cmp.result == "EUR 10,000,000 (ref. 7) exceeds EUR 8,000,000 (ref. 9); difference 2,000,000."
    assert [(v["ref"], v["value"], v["clause"]) for v in cmp.values] == [("7", 10e6, "7.1"), ("9", 8e6, "Table 9.1")]
    assert cmp.values[0]["document"] == "Vendor_Agreement.pdf" and cmp.values[1]["page"] == 1


def test_the_comparison_works_the_other_way_round_and_for_equal_amounts(contract_hits):
    (cmp,) = compare_amounts("Does the total in Section 9 exceed the cap in Section 7?", contract_hits)
    assert cmp.result.startswith("EUR 8,000,000 (ref. 9) does not exceed EUR 10,000,000 (ref. 7)")
    a, b = hit("The cap is EUR 5 million.", clause="1.1"), hit("The total is EUR 5 million.", clause="2.1")
    (eq,) = compare_amounts("Is Section 1 larger than Section 2?", [a, b])
    assert eq.result == "Both amounts are equal (EUR 5,000,000)."


def test_no_comparison_without_comparison_words_two_references_and_amounts(contract_hits):
    assert compare_amounts("What is the liability cap in Section 7 and Section 9?", contract_hits) == []
    assert compare_amounts("Does the cap in Section 7 exceed the usual amount?", contract_hits) == []
    assert compare_amounts("Does Section 7 exceed Section 12?", contract_hits) == []         # no amounts for Section 12
    assert compare_amounts(Q, []) == []


def test_section_numbers_are_matched_exactly_not_by_prefix():
    other = hit("The cap is EUR 99 million.", clause="70.1")
    assert compare_amounts("Does Section 7 exceed Section 9?", [other, hit("The sum is EUR 1 million.", clause="9.1")]) == []
