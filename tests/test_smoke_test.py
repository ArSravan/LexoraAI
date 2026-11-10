"""scripts/smoke_test.py drives a running API; here the API is replaced by a scripted fake."""
from pathlib import Path

import pytest
import requests

from scripts import smoke_test


class Resp:
    def __init__(self, data=None, ok=True, status=200, content=b"", ctype="application/json"):
        self._d, self.ok, self.status_code, self.content = data, ok, status, content
        self.headers = {"content-type": ctype}

    def json(self):
        return self._d


class FakeApi:
    """Just enough of the Lexora API. ``answer`` is the reply to the question; ``ollama`` the Ollama status."""

    def __init__(self, abstain=True, ollama=True, healthy=True, job_status="done", grounded=True):
        self.abstain, self.ollama, self.healthy, self.job_status, self.grounded = abstain, ollama, healthy, job_status, grounded
        self.calls, self.deleted = [], False

    def get(self, url, timeout=0):
        self.calls.append(("GET", url))
        if url.endswith("/health"):
            if not self.healthy:
                raise requests.ConnectionError("down")
            return Resp({"status": "ok"})
        if url.endswith("/status"):
            return Resp({"ollama": {"connected": self.ollama, "model_available": self.ollama, "model": "m",
                                    "base_url": "http://x:11434"}, "ocr_available": True})
        if "/jobs/" in url:
            return Resp({"status": self.job_status, "message": "m", "error": "boom" if self.job_status == "failed" else ""})
        if url.endswith("/info"):
            return Resp({"pages": [{"is_scanned": False}, {"is_scanned": True}], "headings": [1, 2, 3]})
        if url.endswith("/image"):
            return Resp(content=b"x" * 5000, ctype="image/png")
        raise AssertionError(url)

    def post(self, url, json=None, files=None, timeout=0):
        self.calls.append(("POST", url))
        if url.endswith("/cases"):
            return Resp({"id": "c1"})
        if url.endswith("/documents"):
            return Resp({"document": {"id": "d1"}, "job_id": "j1"})
        if url.endswith("/ask"):
            if json["question"] == smoke_test.UNANSWERABLE:
                return Resp({"abstained": self.abstain, "confidence": "INSUFFICIENT EVIDENCE" if self.abstain else "HIGH"})
            claims = [{"status": "VERIFIED"}] if self.grounded else [{"status": "UNVERIFIED"}]
            return Resp({"abstained": not self.grounded, "confidence": "HIGH", "claims": claims})
        raise AssertionError(url)

    def delete(self, url, timeout=0):
        self.deleted = True
        return Resp({"deleted": "c1"})


@pytest.fixture
def pdf(tmp_path) -> Path:
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return p


def verdicts(rep):
    return {c.name: c.verdict for c in rep.checks}


def test_a_healthy_system_passes_every_check_and_the_case_is_cleaned_up(pdf):
    api = FakeApi()
    rep = smoke_test.run("http://h:8000", pdf, with_llm=True, http=api, wait_s=1)
    assert rep.passed and set(verdicts(rep).values()) == {"PASS"}
    assert api.deleted and ("POST", "http://h:8000/api/cases/c1/ask") in api.calls
    assert "| Check | Result | Detail |" in rep.markdown() and "2 pages, 1 scanned, 3 headings" in rep.markdown()


def test_without_llm_the_grounded_question_and_missing_ollama_are_skipped_not_failed(pdf):
    rep = smoke_test.run("http://h", pdf, with_llm=False, http=FakeApi(ollama=False), wait_s=1)
    v = verdicts(rep)
    assert rep.passed
    assert v["Ollama reachable from the API"] == "SKIPPED" and v["Grounded question answered with verified citations"] == "SKIPPED"


def test_with_llm_a_missing_ollama_fails(pdf):
    rep = smoke_test.run("http://h", pdf, with_llm=True, http=FakeApi(ollama=False), wait_s=1)
    assert not rep.passed and verdicts(rep)["Ollama reachable from the API"] == "FAIL"


def test_an_unanswerable_question_that_is_answered_fails(pdf):
    rep = smoke_test.run("http://h", pdf, with_llm=False, http=FakeApi(abstain=False), wait_s=1)
    assert not rep.passed and verdicts(rep)["Unanswerable question is refused"] == "FAIL"


def test_a_grounded_answer_without_verified_claims_fails(pdf):
    rep = smoke_test.run("http://h", pdf, with_llm=True, http=FakeApi(grounded=False), wait_s=1)
    assert verdicts(rep)["Grounded question answered with verified citations"] == "FAIL"


def test_a_failed_indexing_job_stops_the_run_but_still_deletes_the_case(pdf):
    api = FakeApi(job_status="failed")
    rep = smoke_test.run("http://h", pdf, with_llm=False, http=api, wait_s=1)
    assert not rep.passed and verdicts(rep)["Indexing finished"] == "FAIL" and api.deleted
    assert "Pages and structure detected" not in verdicts(rep)


def test_an_unreachable_api_reports_one_failed_check_and_nothing_else(pdf):
    rep = smoke_test.run("http://h", pdf, with_llm=False, http=FakeApi(healthy=False), wait_s=0.1)
    assert [c.name for c in rep.checks] == ["API health"] and not rep.passed


def test_main_rejects_a_missing_pdf(tmp_path, capsys):
    assert smoke_test.main(["--pdf", str(tmp_path / "nope.pdf")]) == 2
    assert "PDF not found" in capsys.readouterr().err
