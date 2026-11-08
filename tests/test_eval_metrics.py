import json

import pytest

from evaluation.metrics import aggregate, chunk_matches, clause_matches, retrieval_scores
from scripts.eval_retrieval import main
from tests.conftest import make_chunk


def test_clause_matching_accepts_sub_clauses_but_not_siblings():
    assert clause_matches("83(5)", "83(5)") and clause_matches("83(5)(a)", "83(5)")
    assert clause_matches("Article 83(5)", "83(5)") and clause_matches("7.2.1", "7.2")
    assert not clause_matches("83(4)", "83(5)") and not clause_matches("83(50)", "83(5)")
    assert not clause_matches("", "83(5)") and not clause_matches("7.20", "7.2")


def test_chunk_matching_needs_the_right_document_too():
    ev = {"document": "A.pdf", "clause": "6(1)"}
    assert chunk_matches(make_chunk("x", "t", clause="6(1)(a)", name="A.pdf"), ev)
    assert not chunk_matches(make_chunk("x", "t", clause="6(1)", name="B.pdf"), ev)


def test_retrieval_scores_have_exact_values():
    exp = [{"document": "Doc.pdf", "clause": "1"}, {"document": "Doc.pdf", "clause": "2"}]
    ranked = [make_chunk("a", "t", clause="9"), make_chunk("b", "t", clause="2"), make_chunk("c", "t", clause="1"),
              make_chunk("d", "t", clause="8")]
    s = retrieval_scores(ranked, exp)
    assert s["recall@1"] == 0 and s["hit@1"] == 0 and s["mrr"] == pytest.approx(0.5)
    assert s["recall@3"] == 1 and s["full@3"] == 1 and s["precision@3"] == pytest.approx(2 / 3)
    assert retrieval_scores(ranked[:2], exp)["recall@5"] == 0.5 and retrieval_scores(ranked[:2], exp)["full@5"] == 0
    assert retrieval_scores([], exp)["mrr"] == 0 and retrieval_scores([], exp)["precision@3"] == 0


def test_aggregate_is_the_mean():
    assert aggregate([{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 0.5}]) == {"a": 0.5, "b": 0.25} and aggregate([]) == {}


def test_eval_script_prints_a_row_per_configuration(eu_style_pdf, contract_pdf, fake_embedder, fake_reranker, tmp_path, capsys):
    ds = tmp_path / "ds.json"
    ds.write_text(json.dumps({"questions": [
        {"id": "F1", "category": "factual", "answerable": True,
         "question": "What is the maximum fine for serious infringements, 4 % of annual turnover?",
         "expected_evidence": [{"document": "Statute.pdf", "clause": "3(2)"}]},
        {"id": "F2", "category": "factual", "answerable": True,
         "question": "What is the supplier liability cap in EUR under Section 7?",
         "expected_evidence": [{"document": "Vendor_Agreement.pdf", "clause": "7.1"}]},
        {"id": "U1", "category": "unanswerable", "answerable": False, "question": "Who won the football match?",
         "expected_evidence": []}]}))
    assert main([str(eu_style_pdf), str(contract_pdf), "--dataset", str(ds), "--misses"]) == 0
    out = capsys.readouterr().out
    assert "answerable questions: 2" in out and "BM25 only" in out and "Hybrid + rerank + query decomposition" in out
    assert "missed at 5 - BM25 only: none" in out


# ------------------------------------------------------------------------------ generation metrics
from evaluation.metrics import (abstention_stats, answer_correct_proxy, citation_counts,  # noqa: E402
                                         citation_precision, fact_present)


@pytest.mark.parametrize("fact,answer,expected", [
    ("72 hours", "Notify within 72 hours of becoming aware.", True),
    ("20 000 000", "Fines up to EUR 20 million apply.", True),                       # quantities, not strings
    ("20 000 000", "Fines up to EUR 10 million apply.", False),
    ("4 %", "or 4% of annual turnover", True),
    ("basic principles", "infringing the Basic Principles for processing", True),
    ("data subjects' rights", "the rights of data subjects", True),                   # word set, stemmed, any order
    ("data subjects' rights", "the rights of controllers", False),
    ("Delaware", "The governing law is not stated.", False),
])
def test_key_facts_are_found_as_text_quantities_or_word_sets(fact, answer, expected):
    assert fact_present(fact, answer) is expected


def test_an_answer_is_correct_only_if_every_key_fact_is_present():
    assert answer_correct_proxy(["72 hours", "supervisory authority"], "Within 72 hours to the supervisory authority.")
    assert not answer_correct_proxy(["72 hours", "supervisory authority"], "Within 72 hours.")
    assert not answer_correct_proxy([], "anything")


def claim(*refs):
    return {"claim": "c", "evidence": [{"status": s, "ref": {"document": d, "clause": cl}} for s, d, cl in refs]}


def test_citation_counts_and_precision_against_the_labelled_evidence():
    claims = [claim(("VERIFIED", "GDPR.pdf", "83(5)(a)"), ("VERIFIED", "GDPR.pdf", "Recital 148")),
              claim(("PARTIAL", "GDPR.pdf", "83(4)"), ("VERIFIED", "Other.pdf", "83(5)"))]
    expected = [{"document": "GDPR.pdf", "clause": "83(5)"}, {"document": "GDPR.pdf", "clause": "83(4)"}]
    assert citation_counts(claims) == (3, 4)
    assert citation_precision(claims, expected) == (2, 4)       # a sub-clause counts; a neighbour recital and a wrong document do not
    assert citation_counts([]) == (0, 0) and citation_precision([], expected) == (0, 0)


def test_abstention_statistics():
    rows = [{"answerable": True, "abstained": False}, {"answerable": True, "abstained": True},
            {"answerable": False, "abstained": True}, {"answerable": False, "abstained": True},
            {"answerable": False, "abstained": False}]
    s = abstention_stats(rows)
    assert s == {"abstention_accuracy": 0.6, "false_abstention_rate": 0.5, "correct_abstention_rate": 0.6667}
    assert abstention_stats([{"answerable": True, "abstained": False}])["correct_abstention_rate"] is None
    assert abstention_stats([]) == {}
