import pytest

from config import get_settings
from scripts.ask_pdf import main
from tests.conftest import FakeOllama, fabricated_answer, grounded_answer

Q = "What is the maximum fine for serious infringements, 4 % of annual turnover?"


@pytest.fixture(autouse=True)
def fake_scale(fake_embedder, fake_reranker, monkeypatch):
    for k, v in {"ABSTAIN_RERANK_MIN": "0.5", "MEDIUM_CONF_RERANK": "0.5", "HIGH_CONF_RERANK": "3.0"}.items():
        monkeypatch.setenv(f"LEXORA_{k}", v)
    get_settings.cache_clear()


def test_a_grounded_answer_is_printed_with_claims_citations_and_checks(eu_style_pdf, capsys):
    assert main([str(eu_style_pdf), "-q", Q], client=FakeOllama(grounded_answer)) == 0
    out = capsys.readouterr().out
    assert "ANSWER   [confidence: HIGH]" in out and "1. [VERIFIED]" in out
    assert "Statute.pdf p.2 clause 3(2)" in out
    assert "document:ok page:ok clause:ok quote:ok retrieved:ok supports:ok" in out and "timings:" in out


def test_a_rejected_draft_is_shown_as_an_abstention_not_as_an_answer(eu_style_pdf, capsys):
    assert main([str(eu_style_pdf), "-q", Q], client=FakeOllama(fabricated_answer)) == 0
    out = capsys.readouterr().out
    assert "[confidence: INSUFFICIENT EVIDENCE]" in out and "WHY NO ANSWER" in out and "VERIFIER RETRIES" in out
    assert "rejected draft: 'The cap is EUR 999 million.'" in out and "CLAIMS" not in out


def test_an_unanswerable_question_shows_what_was_found_instead(eu_style_pdf, capsys):
    client = FakeOllama(grounded_answer)
    main([str(eu_style_pdf), "-q", "Who won the football match yesterday?"], client=client)
    out = capsys.readouterr().out
    assert client.calls == [] and "INSUFFICIENT EVIDENCE" in out and "related: Statute.pdf" in out


def test_the_no_retrieval_baseline_is_labelled(eu_style_pdf, capsys):
    main([str(eu_style_pdf), "-q", Q, "--no-retrieval"], client=FakeOllama("General knowledge answer."))
    out = capsys.readouterr().out
    assert "[no retrieval: ungrounded]" in out and "General knowledge answer." in out


def test_unreachable_ollama_is_reported_before_any_indexing(eu_style_pdf, monkeypatch, capsys):
    monkeypatch.setenv("LEXORA_OLLAMA_BASE_URL", "http://127.0.0.1:9")
    get_settings.cache_clear()
    assert main([str(eu_style_pdf), "-q", Q]) == 2
    err = capsys.readouterr().err
    assert "Ollama is not reachable at http://127.0.0.1:9" in err and "indexing" not in err
