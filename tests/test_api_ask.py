import json

import pytest
from fastapi.testclient import TestClient

from api import routes
from api.app import create_app
from config import get_settings
from generation.answer_generator import AnswerService
from services import db, jobs
from tests.conftest import FakeOllama, fabricated_answer, grounded_answer

Q = "What is the maximum fine for serious infringements, 4 % of annual turnover?"


@pytest.fixture
def llm():
    return FakeOllama(grounded_answer)


@pytest.fixture
def client(fake_embedder, fake_reranker, monkeypatch, llm):
    for k, v in {"ABSTAIN_RERANK_MIN": "0.5", "MEDIUM_CONF_RERANK": "0.5", "HIGH_CONF_RERANK": "3.0"}.items():
        monkeypatch.setenv(f"LEXORA_{k}", v)
    get_settings.cache_clear()
    monkeypatch.setattr(routes, "answer_service", lambda: AnswerService(llm))
    return TestClient(create_app())


@pytest.fixture
def case_id(client, eu_style_pdf):
    cid = client.post("/api/cases", json={"title": "C"}).json()["id"]
    r = client.post(f"/api/cases/{cid}/documents", files={"file": ("Statute.pdf", eu_style_pdf.read_bytes(), "application/pdf")}).json()
    assert jobs.wait(r["job_id"])["status"] == "done"
    return cid


def test_ask_returns_the_full_verified_answer(client, case_id):
    r = client.post(f"/api/cases/{case_id}/ask", json={"question": Q})
    assert r.status_code == 200
    res = r.json()
    assert not res["abstained"] and res["confidence"] == "HIGH" and res["case_id"] == case_id and res["question"] == Q
    assert res["claims"][0]["status"] == "VERIFIED" and res["claims"][0]["evidence"][0]["checks"]["supports_claim"] is True
    assert res["claims"][0]["evidence"][0]["matched_chunk_id"] and res["retrieved"][0]["chunk"]["clause"] == "3(2)"
    assert {"retrieval_s", "generation_s", "total_s"} <= set(res["timings"]) and res["tokens"] == {"prompt": 100, "completion": 50}


def test_options_are_passed_through(client, case_id, llm):
    res = client.post(f"/api/cases/{case_id}/ask", json={"question": Q, "mode": "bm25", "rerank": False, "final_k": 2, "model": "tiny:1b"}).json()
    assert res["mode"] == "bm25" and res["model"] == "tiny:1b" and llm.model == "tiny:1b"
    assert len(res["retrieved"]) == 2 and all(h["rerank_score"] is None for h in res["retrieved"])


def test_a_dishonest_model_yields_an_abstention_through_the_api(client, case_id, llm):
    llm.script = fabricated_answer
    res = client.post(f"/api/cases/{case_id}/ask", json={"question": Q}).json()
    assert res["abstained"] and res["confidence"] == "INSUFFICIENT EVIDENCE" and res["claims"] == []
    assert res["abstention"]["draft_rejected"] == "The cap is EUR 999 million." and len(res["retry_log"]) == 1


def test_every_answer_is_logged_for_the_dashboard(client, case_id):
    client.post(f"/api/cases/{case_id}/ask", json={"question": Q})
    client.post(f"/api/cases/{case_id}/ask", json={"question": "Who won the football match yesterday?"})
    s = client.get("/api/stats").json()
    assert s["questions"] == 2 and s["abstentions"] == 1
    assert [r["confidence"] for r in s["recent"]] == ["INSUFFICIENT EVIDENCE", "HIGH"] and s["recent"][0]["case_title"] == "C"
    assert db.stats()["recent"][1]["latency"] >= 0


def test_ask_needs_an_indexed_document_except_for_the_no_retrieval_baseline(client, llm):
    llm.script = lambda messages: "A general answer."
    cid = client.post("/api/cases", json={"title": "Empty"}).json()["id"]
    r = client.post(f"/api/cases/{cid}/ask", json={"question": Q})
    assert r.status_code == 409 and "No indexed documents" in r.json()["detail"]
    base = client.post(f"/api/cases/{cid}/ask", json={"question": Q, "use_retrieval": False})
    assert base.status_code == 200 and base.json()["mode"] == "none" and base.json()["claims"] == []


@pytest.mark.parametrize("body", [{}, {"question": "hi"}, {"question": "q" * 2001}, {"question": Q, "mode": "magic"},
                                  {"question": Q, "final_k": 0}, {"question": Q, "final_k": 21}])
def test_invalid_questions_are_rejected(client, case_id, body):
    assert client.post(f"/api/cases/{case_id}/ask", json=body).status_code == 422


def test_unknown_case(client):
    assert client.post("/api/cases/nope/ask", json={"question": Q}).status_code == 404
    assert client.post("/api/cases/bad..id/ask/stream", json={"question": Q}).status_code == 400


def ndjson(resp):
    return [json.loads(line) for line in resp.text.splitlines() if line]


def test_the_stream_sends_status_tokens_and_a_final_result(client, case_id, llm):
    r = client.post(f"/api/cases/{case_id}/ask/stream", json={"question": Q})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    events = ndjson(r)
    types = [e["type"] for e in events]
    assert types[-1] == "result" and types.count("result") == 1 and "token" in types and "error" not in types
    statuses = [e["text"] for e in events if e["type"] == "status"]
    assert statuses[0].startswith("Retrieving passages") and "Verifying citations" in statuses
    assert "".join(e["text"] for e in events if e["type"] == "token") == grounded_answer(llm.calls[0])   # the raw reply
    assert events[-1]["result"]["claims"][0]["status"] == "VERIFIED"


def test_a_crash_inside_the_pipeline_becomes_an_error_event_not_a_hang(client, case_id, monkeypatch):
    class Broken:
        def answer(self, *a, **k):
            raise RuntimeError("boom")
    monkeypatch.setattr(routes, "answer_service", lambda: Broken())
    events = ndjson(client.post(f"/api/cases/{case_id}/ask/stream", json={"question": Q}))
    assert events == [{"type": "error", "text": "boom"}]
    assert client.post(f"/api/cases/{case_id}/ask/stream", json={"question": Q, "use_retrieval": True}).status_code == 200


def test_an_unreachable_llm_is_reported_in_the_result_not_as_a_server_error(client, case_id, llm):
    from generation.ollama_client import OllamaError
    llm.script = lambda messages: OllamaError("Ollama is not reachable at http://localhost:11434.")
    r = client.post(f"/api/cases/{case_id}/ask", json={"question": Q})
    assert r.status_code == 200 and r.json()["error"].startswith("Ollama is not reachable") and r.json()["confidence"] == "LOW"
