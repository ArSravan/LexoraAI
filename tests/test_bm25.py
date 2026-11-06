from retrieval.bm25 import BM25Index, tokenize
from tests.conftest import make_chunk

CORPUS = [
    make_chunk("c1", "Infringements shall be subject to administrative fines up to 20 000 000 EUR.", "83(5)", section="Article 83 - Fines"),
    make_chunk("c2", "Processing shall be lawful only if the data subject has given consent.", "6(1)", section="Article 6 - Lawfulness"),
    make_chunk("c3", "Nothing in this Section limits liability for gross negligence.", "7.2", doc="d2", name="Contract.pdf"),
    make_chunk("c4", "The supervisory authority shall be independent in the performance of its tasks.", "52(1)"),
    make_chunk("c5", "Personal data shall be accurate and, where necessary, kept up to date.", "5(1)(d)"),
    make_chunk("c6", "Each Member State shall provide for one or more independent public authorities.", "51(1)"),
    make_chunk("c7", "The controller shall keep records of processing activities under its responsibility.", "30(1)"),
    make_chunk("c8", "Transfers of personal data to a third country require an adequacy decision.", "45(1)"),
]


def test_clause_numbers_survive_tokenization_and_emit_their_parts():
    assert {"83(5)(a)", "83", "5", "a"} <= set(tokenize("Article 83(5)(a) applies"))
    assert {"7.2", "7", "2"} <= set(tokenize("see 7.2"))
    assert {"cross-border", "cross", "border"} <= set(tokenize("a cross-border transfer"))
    assert "the" not in tokenize("the controller") and "shall" not in tokenize("shall comply")


def test_word_forms_match_across_query_and_document():
    assert tokenize("administrative fines") == tokenize("Administrative fined")


def test_keyword_query_finds_the_right_chunk_first():
    hits = BM25Index(CORPUS).search("administrative fines 20 000 000")
    assert hits[0][0] == "c1" and hits[0][1] > 0
    assert BM25Index(CORPUS).search("gross negligence liability")[0][0] == "c3"


def test_a_clause_number_in_the_question_finds_that_clause():
    # the clause label lives in index_text ("... | clause 6(1)"), so "6(1)" is searchable
    assert BM25Index(CORPUS).search("Article 6(1)")[0][0] == "c2"
    assert BM25Index(CORPUS).search("clause 7.2")[0][0] == "c3"


def test_unknown_terms_and_empty_queries_return_nothing():
    idx = BM25Index(CORPUS)
    assert idx.search("blockchain") == [] and idx.search("the of and") == [] and idx.search("") == []
    assert BM25Index([]).search("anything") == []


def test_top_k_scores_are_sorted_and_positive():
    hits = BM25Index(CORPUS).search("data shall controller authority", top_k=3)
    assert len(hits) == 3 and [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)
    assert all(s > 0 for _, s in hits)


def test_allowed_ids_restrict_the_result():
    hits = BM25Index(CORPUS).search("data shall controller authority", top_k=10, allowed={"c3", "c4"})
    assert {i for i, _ in hits} <= {"c3", "c4"}
