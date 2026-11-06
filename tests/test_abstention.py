from dataclasses import replace

import pytest

from config import Settings
from schemas import RetrievedChunk
from verification.abstention import ABSTAIN_TEXT, build_abstention
from verification.evidence_checker import Sufficiency, assess, coverage, salient_terms
from tests.conftest import make_chunk

S = Settings()          # abstain_rerank_min 2.5, abstain_coverage_min 0.32


def hit(text, rerank=None, dense=None, clause="1", name="Doc.pdf"):
    return RetrievedChunk(chunk=make_chunk("c", text, clause=clause, name=name), rerank_score=rerank, dense_score=dense)


def test_salient_terms_drop_stopwords_generic_words_and_duplicates():
    assert salient_terms("What is the maximum fine for violations under this agreement?") == ["fin"]
    assert salient_terms("fines and fined fines") == ["fin"]
    assert salient_terms("What does the document say?") == []


def test_coverage_counts_terms_found_in_the_top_passages():
    hits = [hit("administrative fines for infringements")]
    cov, missing = coverage("fines for infringements of blockchain rules", hits)
    assert missing == ["blockchain", "rules"] and cov == pytest.approx(0.5)         # shown as typed, matched as stems
    assert coverage("What does the document say?", hits) == (1.0, [])               # no topical terms: nothing to miss


def test_coverage_only_looks_at_the_top_five():
    hits = [hit("filler text")] * 5 + [hit("blockchain")]
    assert coverage("blockchain", hits)[1] == ["blockchain"]


def test_strong_evidence_is_sufficient():
    s = assess("administrative fines for infringements", [hit("administrative fines for infringements", rerank=7.0)], S)
    assert s.sufficient and s.reasons == [] and s.top_score == 7.0 and s.coverage == 1.0 and s.n_relevant == 1


def test_low_relevance_means_abstain():
    s = assess("administrative fines", [hit("administrative fines", rerank=0.66)], S)
    assert not s.sufficient and "0.66" in s.reasons[0] and "2.50" in s.reasons[0]


def test_missing_topic_means_abstain_even_with_a_good_score():
    s = assess("what does it say about blockchain mining", [hit("fines for infringements", rerank=9.0)], S)
    assert not s.sufficient and "blockchain" in " ".join(s.reasons) and s.coverage < S.abstain_coverage_min


def test_both_failures_are_reported():
    s = assess("blockchain mining", [hit("fines", rerank=-3.0)], S)
    assert len(s.reasons) == 2


def test_no_hits_means_abstain():
    s = assess("anything about fines", [], S)
    assert not s.sufficient and s.top_score is None and "No passages" in s.reasons[0] and s.missing_terms == ["anything", "fines"]


def test_thresholds_come_from_the_settings():
    h = [hit("administrative fines", rerank=1.0)]
    assert not assess("administrative fines", h, S).sufficient
    assert assess("administrative fines", h, replace(S, abstain_rerank_min=0.5)).sufficient


def test_without_reranking_dense_similarity_is_used():
    ok = assess("administrative fines", [hit("administrative fines", dense=0.62)], S)
    assert ok.sufficient and ok.top_score == 0.62 and ok.n_relevant == 1
    low = assess("administrative fines", [hit("administrative fines", dense=0.1)], S)
    assert not low.sufficient and "semantic similarity" in low.reasons[0]


def test_bm25_only_retrieval_has_no_relevance_score_so_only_coverage_decides():
    bm25_only = hit("administrative fines for infringements")                               # no rerank, no dense score
    ok = assess("administrative fines", [bm25_only], S)
    assert ok.sufficient and ok.top_score is None and not ok.rerank_scored and ok.coverage == 1.0
    off_topic = assess("blockchain mining", [bm25_only], S)
    assert not off_topic.sufficient and "blockchain" in off_topic.reasons[0] and len(off_topic.reasons) == 1


def test_the_abstention_explains_what_was_searched_and_what_was_found():
    hits = [hit("The controller shall keep records of processing.", rerank=0.2, clause="30(1)", name="GDPR.pdf")] * 4
    suff = Sufficiency(False, 0.2, 0.0, ["blockchain"], 0, ["Best passage relevance (0.20) is below the threshold (2.50)."])
    info = build_abstention("Who regulates blockchain?", hits, ["GDPR.pdf", "Act.pdf"], suff, ["extra reason"])
    assert info["answer"] == ABSTAIN_TEXT == "INSUFFICIENT EVIDENCE"
    assert info["searched_documents"] == ["GDPR.pdf", "Act.pdf"] and info["looked_for"] == "Who regulates blockchain?"
    assert len(info["related_passages"]) == 3 and info["related_passages"][0]["clause"] == "30(1)"
    assert info["reasons"][-1] == "extra reason" and "blockchain" in info["explanation"]
    assert "does not fill gaps with general legal knowledge" in info["explanation"]
    assert info["features"] == {"top_score": 0.2, "coverage": 0.0}


def test_abstention_without_hits_or_assessment():
    info = build_abstention("q", [], [], None)
    assert info["related_passages"] == [] and "No related passages were found." in info["explanation"] and info["features"]["coverage"] is None
