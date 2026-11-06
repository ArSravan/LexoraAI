import pytest

from schemas import Claim, EvidenceRef, LLMAnswer, RetrievedChunk
from textutil import clause_key
from verification.citation_validator import (CaseDoc, quote_match, supports_claim, validate_answer,
                                                      validate_evidence)
from tests.conftest import make_chunk

CAP = "The Supplier's total liability under this Agreement shall not exceed EUR 10,000,000 per year."
IND = "The Supplier shall indemnify the Customer for losses arising from a data breach."
DOCS = [CaseDoc("d1", "Contract.pdf", 10, {clause_key(c) for c in ("7.2", "9.1", "7")}),
        CaseDoc("d2", "Other.pdf", 2, {clause_key("1")})]
HITS = [RetrievedChunk(chunk=make_chunk("c1", CAP, clause="7.2", page=3, name="Contract.pdf", doc="d1")),
        RetrievedChunk(chunk=make_chunk("c2", IND, clause="9.1", page=5, name="Contract.pdf", doc="d1"))]
CLAIM = "The Supplier's total liability shall not exceed EUR 10 million."


def ref(**kw):
    base = dict(passage_id="E1", document="Contract.pdf", page=3, clause="7.2",
                supporting_text="total liability under this Agreement shall not exceed EUR 10,000,000")
    return EvidenceRef(**{**base, **kw})


def check(r, claim=CLAIM, hits=HITS):
    return validate_evidence(r, claim, hits, DOCS)


def test_a_correct_citation_passes_all_six_checks():
    v = check(ref())
    assert v.status == "VERIFIED" and v.notes == []
    assert all(getattr(v.checks, f) for f in ("document_exists", "page_exists", "clause_exists",
                                              "text_in_passage", "passage_retrieved", "supports_claim"))
    assert v.matched_chunk_id == "c1" and v.document_id == "d1" and v.page_index == 2 and v.source_text == CAP


def test_a_fabricated_document_with_a_fabricated_quote_is_rejected():
    v = check(ref(passage_id="", document="Imaginary.pdf", page=1, supporting_text="a sentence nobody wrote"))
    assert v.status == "FABRICATED" and not v.checks.document_exists and "not part of this case" in " ".join(v.notes)


def test_a_wrong_document_name_with_a_real_quote_is_never_verified():
    v = check(ref(passage_id="", document="Imaginary.pdf"))
    assert not v.checks.document_exists and v.checks.text_in_passage and v.status == "PARTIAL"


def test_a_page_that_does_not_exist_is_rejected():
    v = check(ref(page=99))
    assert not v.checks.page_exists and v.status == "PARTIAL"
    assert any("does not exist" in n for n in v.notes)


def test_the_right_page_of_the_wrong_passage_is_not_verified():
    v = check(ref(page=5))                      # E1 is on page 3, page 5 exists but holds another clause
    assert v.status == "PARTIAL" and not v.checks.page_exists and any("differs from the passage page 3" in n for n in v.notes)


def test_a_clause_that_does_not_exist_in_the_document_is_rejected():
    v = check(ref(clause="99.9"))
    assert not v.checks.clause_exists and v.status != "VERIFIED" and any("not found in the extracted structure" in n for n in v.notes)


def test_a_real_clause_cited_for_the_wrong_passage_is_not_verified():
    v = check(ref(clause="9.1"))                # 9.1 exists, but E1 is clause 7.2
    assert not v.checks.clause_exists and v.status == "PARTIAL" and any("differs from the passage clause" in n for n in v.notes)


def test_sub_clauses_are_accepted_but_neighbouring_numbers_are_not():
    assert check(ref(clause="7")).status == "VERIFIED"          # 7 is the parent of 7.2
    chunk = make_chunk("c9", CAP, clause="70", page=3, name="Contract.pdf", doc="d1")
    v = validate_evidence(ref(clause="7"), CLAIM, [RetrievedChunk(chunk=chunk)], DOCS)
    assert not v.checks.clause_exists                            # '7' must not match '70'


def test_a_quote_that_is_not_in_the_passage_is_rejected():
    v = check(ref(supporting_text="the supplier is liable without any limit whatsoever"))
    assert not v.checks.text_in_passage and v.status == "UNVERIFIED" and any("not found in the source passage" in n for n in v.notes)


def test_a_missing_quote_is_flagged():
    v = check(ref(supporting_text=""))
    assert not v.checks.text_in_passage and "No supporting text was quoted." in v.notes and v.status != "VERIFIED"


def test_a_valid_citation_that_does_not_support_the_claim_is_not_verified():
    v = check(ref(), claim="The Supplier's total liability shall not exceed EUR 5 million.")
    assert v.checks.text_in_passage and v.checks.page_exists and not v.checks.supports_claim
    assert v.status == "PARTIAL" and any("does not support" in n for n in v.notes)


def test_a_passage_that_was_never_retrieved_is_fabricated():
    v = check(ref(passage_id="E9", page=8, clause="7.2", supporting_text="a sentence nobody retrieved"))
    assert not v.checks.passage_retrieved and v.status == "FABRICATED"
    assert any("not part of the retrieved evidence" in n for n in v.notes)


def test_fallback_to_document_page_and_clause_when_the_id_is_wrong():
    v = check(ref(passage_id="E9"))             # bad id, but document/page/clause identify hits[0]
    assert v.status == "VERIFIED" and v.matched_chunk_id == "c1"


def test_a_quote_from_another_retrieved_passage_is_noticed_and_downgraded():
    v = check(ref(passage_id="E2", page=5, clause="9.1"))   # cites E2 but quotes E1's sentence
    assert any("different retrieved passage" in n for n in v.notes) and v.matched_chunk_id == "c1"
    assert v.status == "PARTIAL"                              # page 5 / clause 9.1 do not match E1


def test_blank_fields_are_filled_from_the_cited_passage_but_noted():
    v = check(ref(document="", page=None, clause=""))
    assert v.ref.document == "Contract.pdf" and v.ref.page == 3 and v.ref.clause == "7.2" and v.status == "VERIFIED"
    assert "Document name taken from the cited passage id." in v.notes


def test_claims_are_verified_if_any_evidence_item_is():
    ans = LLMAnswer(answer="x", claims=[
        Claim(claim=CLAIM, evidence=[ref(passage_id="E9", document="Nope.pdf", page=1, supporting_text="invented"), ref()]),
        Claim(claim="No source for this one."),
        Claim(claim=CLAIM, evidence=[ref(page=99)])])
    out = validate_answer(ans, HITS, DOCS)
    assert out[0].status == "VERIFIED" and [e.status for e in out[0].evidence] == ["FABRICATED", "VERIFIED"]
    assert out[1].status == "UNVERIFIED" and out[1].evidence == []
    assert out[2].status == "PARTIAL"


@pytest.mark.parametrize("quote,expected", [
    ("total liability under this agreement", 1.0),
    ("TOTAL   LIABILITY\nunder this Agreement", 1.0),
    ("The Supplier’s total liability", 1.0),                         # typographic apostrophe
    ("Supplier's total ... shall not exceed EUR 10,000,000", 1.0),        # "..." = omission
    ("", 0.0),
])
def test_quote_match_exact_cases(quote, expected):
    assert quote_match(quote, CAP) == pytest.approx(expected)


def test_quote_match_gives_partial_credit_and_zero_for_unrelated_text():
    assert 0.5 < quote_match("total liability shall exceed EUR 50,000,000 per year", CAP) < 1.0
    assert quote_match("completely unrelated words here", CAP) < 0.3
    assert quote_match("shall exceed ... not liability", CAP) < 1.0          # pieces out of order


def test_supports_claim_checks_numbers_and_terms():
    ev = CAP
    assert supports_claim("liability shall not exceed EUR 10 million", "", ev)          # 10 million == 10,000,000
    assert supports_claim("liability cap is 10 000 000 EUR", "", ev)
    assert not supports_claim("liability shall not exceed EUR 12 million", "", ev)      # wrong number
    assert not supports_claim("The tenant must repaint the building annually", "", ev)  # unrelated terms
    assert supports_claim("Section 7.2 limits liability to EUR 10 million", "", ev)      # "7.2" is a reference, not a number


@pytest.mark.parametrize("blank", ["-", "–", "n/a", "None", " - ", ""])
def test_a_passage_without_a_clause_can_be_cited_with_the_placeholder_the_prompt_shows(blank):
    # unstructured documents (annual reports, scans) have no clause numbers; the prompt header says 'clause: -'
    text = "Total revenue for FY2025 was EUR 120 million."
    hits = [RetrievedChunk(chunk=make_chunk("r1", text, clause="", page=2, name="Report.pdf", doc="r"))]
    docs = [CaseDoc("r", "Report.pdf", 5, set())]
    r = EvidenceRef(passage_id="E1", document="Report.pdf", page=2, clause=blank, supporting_text="revenue for FY2025 was EUR 120 million")
    v = validate_evidence(r, "Revenue for FY2025 was EUR 120 million.", hits, docs)
    assert v.status == "VERIFIED" and v.checks.clause_exists and v.notes == []


def test_a_clause_cannot_be_invented_for_a_passage_that_has_none():
    hits = [RetrievedChunk(chunk=make_chunk("r1", "Total revenue was EUR 120 million.", clause="", page=2, name="Report.pdf", doc="r"))]
    docs = [CaseDoc("r", "Report.pdf", 5, set())]
    r = EvidenceRef(passage_id="E1", document="Report.pdf", page=2, clause="4.1", supporting_text="revenue was EUR 120 million")
    v = validate_evidence(r, "Revenue was EUR 120 million.", hits, docs)
    assert v.status != "VERIFIED" and not v.checks.clause_exists
