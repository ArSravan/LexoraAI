"""A case indexed by older code has no document metadata. That used to make EVERY citation fail silently: the model was
asked (minutes per question) and each answer became "insufficient evidence". It must be detected, repaired or reported."""
import pytest

from config import get_settings
from generation.answer_generator import AnswerService
from scripts import search_pdf
from services import indexer
from tests.conftest import FakeOllama, grounded_answer

Q = "What is the maximum fine for serious infringements, 4 % of annual turnover?"


@pytest.fixture
def scaled(monkeypatch):
    for k, v in {"ABSTAIN_RERANK_MIN": "0.5", "MEDIUM_CONF_RERANK": "0.5", "HIGH_CONF_RERANK": "3.0"}.items():
        monkeypatch.setenv(f"LEXORA_{k}", v)
    get_settings.cache_clear()


def drop_metadata(case_id):
    for f in (get_settings().storage_dir / "cases" / case_id / "docs").glob("*/meta.json"):
        f.unlink()


def test_an_index_is_current_only_when_every_document_has_its_metadata(fake_embedder, eu_style_pdf, contract_pdf):
    assert indexer.index_is_current("nothing-here") is False
    indexer.index_pdf("c1", "d0", eu_style_pdf, "Statute.pdf")
    indexer.index_pdf("c1", "d1", contract_pdf, "Contract.pdf")
    assert indexer.index_is_current("c1") is True
    (get_settings().storage_dir / "cases" / "c1" / "docs" / "d1" / "meta.json").unlink()      # one document is stale
    assert indexer.index_is_current("c1") is False


def test_the_fingerprint_changes_exactly_when_the_index_changes(fake_embedder, eu_style_pdf, contract_pdf):
    assert indexer.index_fingerprint("c1") == indexer.index_fingerprint("never-indexed")      # nothing to fingerprint
    indexer.index_pdf("c1", "d0", eu_style_pdf, "Statute.pdf")
    first = indexer.index_fingerprint("c1")
    assert first == indexer.index_fingerprint("c1") and len(first) == 16
    indexer.index_pdf("c1", "d0", eu_style_pdf, "Statute.pdf")                                  # same content again
    assert indexer.index_fingerprint("c1") == first
    indexer.index_pdf("c1", "d1", contract_pdf, "Contract.pdf")
    assert indexer.index_fingerprint("c1") != first
    changed = indexer.index_fingerprint("c1")
    drop_metadata("c1")
    assert indexer.index_fingerprint("c1") != changed


def test_a_stale_case_fails_loudly_and_the_model_is_not_asked(fake_embedder, fake_reranker, scaled, eu_style_pdf):
    indexer.index_pdf("c1", "d0", eu_style_pdf, "Statute.pdf")
    drop_metadata("c1")
    llm = FakeOllama(grounded_answer)
    res = AnswerService(llm).answer("c1", Q)
    assert llm.calls == []                                                   # no minutes wasted on answers that cannot verify
    assert not res.abstained and res.confidence == "LOW" and res.claims == []
    assert "out of date" in res.error and "Re-index" in res.error and "model was not asked" in res.error
    assert res.answer.startswith("This case cannot be answered:")


def test_a_healthy_case_still_answers(fake_embedder, fake_reranker, scaled, eu_style_pdf):
    indexer.index_pdf("c1", "d0", eu_style_pdf, "Statute.pdf")
    res = AnswerService(FakeOllama(grounded_answer)).answer("c1", Q)
    assert res.error == "" and not res.abstained and res.claims[0].status == "VERIFIED"


def test_the_stale_error_reaches_the_api_caller(fake_embedder, fake_reranker, scaled, eu_style_pdf, monkeypatch):
    from fastapi.testclient import TestClient
    from api import routes
    from api.app import create_app
    from services import db, jobs
    client = TestClient(create_app())
    cid = client.post("/api/cases", json={"title": "C"}).json()["id"]
    r = client.post(f"/api/cases/{cid}/documents", files={"file": ("Statute.pdf", eu_style_pdf.read_bytes(), "application/pdf")}).json()
    assert jobs.wait(r["job_id"])["status"] == "done"
    drop_metadata(cid)
    monkeypatch.setattr(routes, "answer_service", lambda: AnswerService(FakeOllama(grounded_answer)))
    res = client.post(f"/api/cases/{cid}/ask", json={"question": Q}).json()
    assert "out of date" in res["error"] and res["confidence"] == "LOW"


def test_ensure_indexed_rebuilds_a_stale_case_and_leaves_a_current_one_alone(fake_embedder, eu_style_pdf, monkeypatch, capsys):
    case = search_pdf.ensure_indexed([eu_style_pdf])
    assert indexer.index_is_current(case)
    calls = []
    real = indexer.index_pdf
    monkeypatch.setattr(indexer, "index_pdf", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    capsys.readouterr()
    assert search_pdf.ensure_indexed([eu_style_pdf]) == case and calls == []                  # current: nothing is redone
    assert "out of date" not in capsys.readouterr().err
    drop_metadata(case)                                                                       # becomes stale
    assert search_pdf.ensure_indexed([eu_style_pdf]) == case
    assert calls == [1] and indexer.index_is_current(case)
    err = capsys.readouterr().err
    assert "out of date: rebuilding" in err and "indexing Statute.pdf" in err
    assert len(indexer.case_chunks(case)) > 5                                                 # rebuilt, not duplicated
