"""The UI's HTTP client against the REAL API served on a local port (no mocks of the HTTP layer)."""
import socket
import threading
import time

import pytest
import uvicorn

from api import routes
from api.app import create_app
from config import get_settings
from frontend import api, ui
from generation.answer_generator import AnswerService
from tests.conftest import FakeOllama, grounded_answer

Q = "What is the maximum fine for serious infringements, 4 % of annual turnover?"


@pytest.fixture
def server(fake_embedder, fake_reranker, monkeypatch):
    for k, v in {"ABSTAIN_RERANK_MIN": "0.5", "MEDIUM_CONF_RERANK": "0.5", "HIGH_CONF_RERANK": "3.0"}.items():
        monkeypatch.setenv(f"LEXORA_{k}", v)
    get_settings.cache_clear()
    monkeypatch.setattr(routes, "answer_service", lambda: AnswerService(FakeOllama(grounded_answer)))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    while not srv.started:
        time.sleep(0.02)
    monkeypatch.setenv("LEXORA_API_URL", f"http://127.0.0.1:{port}")
    yield port
    srv.should_exit = True
    t.join(5)


def test_the_whole_flow_through_the_client(server, eu_style_pdf):
    assert api.get("/health") == {"status": "ok"}
    case = api.post("/cases", {"title": "Via client"})
    up = api.post(f"/cases/{case['id']}/documents", files={"file": ("Statute.pdf", eu_style_pdf.read_bytes(), "application/pdf")})
    progress = []
    (job,) = api.wait_for_jobs([up["job_id"]], on_progress=lambda f, m: progress.append((f, m)), poll=0.05)
    assert job["status"] == "done" and progress and progress[-1][0] == 1.0
    docs = api.get(f"/cases/{case['id']}/documents")
    assert docs[0]["status"] == "indexed" and ui.document_row(docs[0])["Pages"] == 3

    events = list(api.ask_stream(case["id"], {"question": Q}))
    assert [e["type"] for e in events][-1] == "result" and any(e["type"] == "status" for e in events)
    result = events[-1]["result"]
    assert result["confidence"] == "HIGH"
    (req, *_) = ui.evidence_requests(result)
    png, info = api.image(f"/documents/{req['document_id']}/pages/{req['page_number']}/image", quote=req["quote"], chunk_id=req["chunk_id"])
    assert png[:4] == b"\x89PNG" and info["region"] is True
    assert api.delete(f"/cases/{case['id']}") == {"deleted": case["id"]}


def test_error_messages_are_the_servers_detail(server):
    with pytest.raises(api.ApiError, match="Case not found"):
        api.get("/cases/doesnotexist")
    with pytest.raises(api.ApiError, match="at least 1 character"):
        api.post("/cases", {"title": ""})
    case = api.post("/cases", {"title": "C"})
    with pytest.raises(api.ApiError, match="No indexed documents"):
        list(api.ask_stream(case["id"], {"question": Q}))


def test_images_that_cannot_be_fetched_are_none(server):
    assert api.image("/documents/nope/pages/1/image") == (None, {})


def test_an_unreachable_api_gives_an_actionable_message(monkeypatch):
    monkeypatch.setenv("LEXORA_API_URL", "http://127.0.0.1:9")
    with pytest.raises(api.ApiError, match="python main.py"):
        api.get("/health")
    with pytest.raises(api.ApiError, match="Cannot reach"):
        list(api.ask_stream("c1", {"question": Q}))
    assert api.image("/documents/x/pages/1/image") == (None, {})


def test_wait_for_jobs_times_out(server, monkeypatch):
    from services import jobs
    gate = threading.Event()
    jid = jobs.start("t", lambda p: gate.wait(5))
    with pytest.raises(api.ApiError, match="too long"):
        api.wait_for_jobs([jid], poll=0.02, timeout=0.1)
    gate.set()


# ------------------------------------------------------------------------------ pure helpers
def test_helpers_format_what_the_ui_shows():
    assert ui.badge("HIGH") == ":green[**HIGH**]" and ui.badge("INSUFFICIENT EVIDENCE") == ":red[**INSUFFICIENT EVIDENCE**]"
    assert ui.badge("anything else") == ":gray[**anything else**]"
    assert ui.citation_label({"document": "C.pdf", "page": 3, "clause": "7.2"}) == "C.pdf, p.3, clause 7.2"
    assert ui.citation_label({"document": "C.pdf", "page": None, "clause": ""}) == "C.pdf"
    assert ui.citation_label({}) == "unknown document"
    checks = {"document_exists": True, "page_exists": True, "clause_exists": False, "text_in_passage": True,
              "passage_retrieved": True, "supports_claim": False}
    assert [ok for _, ok in ui.check_list(checks)] == [True, True, False, True, True, False]
    assert ui.check_line(checks).count("✅") == 4 and ui.check_line(checks).count("❌") == 2


def test_page_rows_and_evidence_requests_and_timings():
    rows = ui.page_rows([{"page_number": 1, "char_count": 900, "method": "digital", "is_scanned": False, "ocr_confidence": None, "n_tables": 1, "warning": ""},
                         {"page_number": 2, "char_count": 120, "method": "ocr", "is_scanned": True, "ocr_confidence": 95.6, "n_tables": 0, "warning": "w"},
                         {"page_number": 3, "char_count": 0, "method": "digital", "is_scanned": True, "ocr_confidence": None, "n_tables": 0, "warning": ""}])
    assert [r["Read by"] for r in rows] == ["text layer", "OCR", "scan, not read"] and rows[1]["OCR confidence"] == "96%" and rows[0]["OCR confidence"] == ""
    result = {"claims": [{"claim": "c", "evidence": [
        {"document_id": "d1", "page_index": 1, "matched_chunk_id": "d1-00007", "status": "VERIFIED",
         "ref": {"document": "A.pdf", "page": 2, "clause": "3(2)", "supporting_text": "fines up to"}},
        {"document_id": "", "page_index": None, "matched_chunk_id": None, "status": "FABRICATED", "ref": {"document": "X.pdf"}}]}],
        "timings": {"retrieval_s": 1.5, "total_s": 9.0}, "tokens": {"prompt": 10, "completion": 5}}
    (req,) = ui.evidence_requests(result)
    assert req == {"key": "c1e1", "document_id": "d1", "page_number": 2, "quote": "fines up to", "chunk_id": "d1-00007",
                   "label": "A.pdf, p.2, clause 3(2)", "status": "VERIFIED"}
    assert ui.timing_caption(result) == "retrieval 1.5s | total 9.0s | tokens 10 in / 5 out"
    assert ui.evidence_requests({}) == []
    same = {"document_id": "d1", "page_index": 1, "matched_chunk_id": "d1-00007", "status": "VERIFIED",
            "ref": {"document": "A.pdf", "page": 2, "clause": "3(2)", "supporting_text": "q"}}
    other = {**same, "ref": {**same["ref"], "clause": "4(1)", "supporting_text": "r"}}
    twice = {"claims": [{"claim": "a", "evidence": [same]}, {"claim": "b", "evidence": [{**same, "ref": {**same["ref"], "supporting_text": "later"}}, other]}]}
    assert [(r["label"], r["quote"]) for r in ui.evidence_requests(twice)] == [("A.pdf, p.2, clause 3(2)", "q"), ("A.pdf, p.2, clause 4(1)", "r")]
