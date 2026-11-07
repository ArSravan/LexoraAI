"""Streamlit's headless test runner renders the page code without a browser: no exception may escape and
the important facts must be on the page, for every kind of answer."""
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from streamlit.testing.v1 import AppTest

from api.app import create_app
from schemas import (AmountComparison, AnswerResult, CheckResult, Conflict, EvidenceRef, RetrievedChunk,
                              VerifiedClaim, VerifiedEvidence)
from tests.conftest import make_chunk

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def evidence(status="VERIFIED", ok=True):
    c = {k: ok for k in CheckResult.model_fields if k != "match_ratio"}
    return VerifiedEvidence(ref=EvidenceRef(passage_id="E1", document="Contract.pdf", page=3, clause="7.2",
                                            supporting_text="total liability shall not exceed EUR 10,000,000"),
                            status=status, checks=CheckResult(**c, match_ratio=1.0), matched_chunk_id="d1-00001",
                            document_id="d1", page_index=2, bbox=[50, 50, 300, 90], notes=["a note"] if not ok else [])


def result(**kw) -> dict:
    base = dict(question="What is the cap?", case_id="c1", answer="The cap is EUR 10 million.", abstained=False,
                confidence="HIGH", confidence_factors=["1/1 claims have citations that passed all verification checks."],
                claims=[VerifiedClaim(claim="The cap is EUR 10 million.", status="VERIFIED", evidence=[evidence()])],
                retrieved=[RetrievedChunk(chunk=make_chunk("d1-00001", "The cap is EUR 10,000,000.", clause="7.2", page=3),
                                          rerank_score=7.5, final_rank=1)],
                timings={"retrieval_s": 1.2, "total_s": 9.9}, tokens={"prompt": 100, "completion": 40}, reasoning_summary="Direct.")
    return AnswerResult(**{**base, **kw}).model_dump()


def render(res: dict) -> AppTest:
    def script(res):
        from frontend.views.chat import render_result
        render_result(res, "t")
    at = AppTest.from_function(script, args=(res,), default_timeout=20)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [w.value for w in at.warning]
    parts += [i.value for i in at.info] + [e.value for e in at.error]
    return "\n".join(parts)


def test_a_verified_answer_shows_confidence_claims_citations_and_the_six_checks():
    at = render(result())
    t = text(at)
    assert "HIGH" in t and "The cap is EUR 10 million." in t and "Contract.pdf, p.3, clause 7.2" in t
    assert t.count("✅") == 6 and "❌" not in t and "Claims and citations" in t
    assert [x.label for x in at.toggle] == ["Show the cited page: Contract.pdf, p.3, clause 7.2"]


def test_a_failed_check_and_its_note_are_visible():
    claims = [VerifiedClaim(claim="Wrong page cited.", status="PARTIAL", evidence=[evidence("PARTIAL", ok=False)])]
    t = text(render(result(claims=claims, confidence="LOW")))
    assert "PARTIAL" in t and t.count("❌") == 6 and "note: a note" in t


def test_an_abstention_explains_itself_and_shows_the_related_passages_without_an_answer():
    res = result(abstained=True, confidence="INSUFFICIENT EVIDENCE", answer="INSUFFICIENT EVIDENCE", claims=[],
                 abstention={"answer": "INSUFFICIENT EVIDENCE", "explanation": "The documents do not contain enough evidence.",
                             "reasons": ["Best passage relevance (-6.33) is below the threshold (2.50)."],
                             "related_passages": [{"document": "GDPR.pdf", "page": 12, "clause": "Recital 64", "score": -6.3, "preview": "(64) The controller"}],
                             "draft_rejected": "The cap is EUR 999 million."})
    at = render(res)
    t = text(at)
    assert "INSUFFICIENT EVIDENCE" in t and "below the threshold" in t and "Claims and citations" not in t
    assert "GDPR.pdf, p.12, clause Recital 64" in t and len(at.toggle) == 0
    assert "999 million" in "\n".join(e.label for e in at.expander) or "rejected" in "\n".join(e.label for e in at.expander)


def test_comparisons_and_conflicts_are_highlighted():
    cmp = AmountComparison(description="d", values=[], result="EUR 10,000,000 (ref. 7) exceeds EUR 8,000,000 (ref. 9); difference 2,000,000.")
    conf = [Conflict(kind="amount", topic="revenue", classification="CONFLICT", explanation="Two documents differ.", sources=[]),
            Conflict(kind="amount", topic="revenue", classification="EXPLAINED_DIFFERENCE", explanation="Restated.", sources=[])]
    at = render(result(comparisons=[cmp], conflicts=conf))
    assert any("exceeds EUR 8,000,000" in i.value for i in at.info)
    assert any("Conflict" in w.value and "Two documents differ." in w.value for w in at.warning)
    assert any("Explained Difference" in i.value for i in at.info)


def test_the_no_retrieval_baseline_is_clearly_marked_ungrounded():
    t = text(render(result(use_retrieval=False, mode="none", claims=[], retrieved=[], confidence="LOW", answer="General answer.")))
    assert "ungrounded" in t and "General answer." in t and "Claims and citations" not in t


def test_an_error_result_is_shown_as_an_error():
    at = render(result(error="Ollama is not reachable at http://localhost:11434.", claims=[], confidence="LOW"))
    assert [e.value for e in at.error] == ["Ollama is not reachable at http://localhost:11434."]


def test_a_retried_answer_mentions_the_verifier():
    at = render(result(retry_log=["Attempt 1 rejected by the verifier: Claim 'x': Document 'Nope.pdf' is not part of this case."]))
    assert any("verifier rejected" in e.label for e in at.expander) and "not part of this case" in text(at)


# ------------------------------------------------------------------------------ the whole app
def test_the_app_starts_and_tells_the_user_when_the_api_is_offline(monkeypatch):
    monkeypatch.setenv("LEXORA_API_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("API offline" in m.value for m in at.sidebar.markdown) and any("python main.py" in e.value for e in at.error)


@pytest.fixture
def live_api(fake_embedder, fake_reranker, monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    while not srv.started:
        time.sleep(0.02)
    monkeypatch.setenv("LEXORA_API_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("LEXORA_OLLAMA_BASE_URL", "http://127.0.0.1:9")
    yield port
    srv.should_exit = True
    t.join(5)


def test_the_app_shows_status_and_guides_the_user_when_there_are_no_cases(live_api):
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    side = " ".join(m.value for m in at.sidebar.markdown)
    assert "API online" in side and "Ollama offline" in side
    assert any("No cases yet" in i.value for i in at.info)
