from retrieval.synonyms import expand_query
from verification.citation_validator import _quoted_numbers_cover


def test_colloquial_terms_get_the_documents_wording_added():
    q = expand_query("Which reduction applies to the van target from 2030?")
    assert q.startswith("Which reduction applies to the van target") and "light commercial vehicles" in q


def test_abbreviations_are_expanded_but_kept():
    q = expand_query("What is the BLEU for EN-DE?")
    assert "EN-DE" in q and "English-to-German" in q


def test_a_question_without_such_words_is_unchanged():
    assert expand_query("What is the fine?") == "What is the fine?"


def test_a_number_missing_from_the_quote_downgrades_the_claim():
    assert _quoted_numbers_cover("F1 was 92.1", ["Transformer (4 layers) semi-supervised"], "VERIFIED") == "PARTIAL"
    assert _quoted_numbers_cover("F1 was 92.1", ["semi-supervised 92.1"], "VERIFIED") == "VERIFIED"
    assert _quoted_numbers_cover("a plain claim", ["x"], "VERIFIED") == "VERIFIED"


def test_years_in_a_claim_need_not_be_in_the_quote():
    assert _quoted_numbers_cover("A 31% reduction of the 2021 target from 2030", ["target equal to a 31 % reduction"], "VERIFIED") == "VERIFIED"


def test_ocr_damaged_big_o_formulas_are_restored():
    from ingestion.ocr import fix_ocr_text
    assert fix_ocr_text("Recurrent O(n -d?) O(n) O(n)") == "Recurrent O(n·d²) O(n) O(n)"
    assert fix_ocr_text("Self-Attention O(n? - d) O(1)") == "Self-Attention O(n²·d) O(1)"
    assert fix_ocr_text("Convolutional O(k-n-d?) O(1)") == "Convolutional O(k·n·d²) O(1)"
    assert fix_ocr_text("O(n) and O(1) and O(log n)") == "O(n) and O(1) and O(log n)"
    assert fix_ocr_text("147 g CO, /km") == "147 g CO2 /km"
