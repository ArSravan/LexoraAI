import pymupdf
import pytest
from fastapi.testclient import TestClient

from api import routes
from api.app import create_app
from config import get_settings
from retrieval import vector_store
from services import db, indexer, jobs
from services.paths import case_dir, doc_dir


@pytest.fixture
def client(fake_embedder):
    return TestClient(create_app())


@pytest.fixture
def case_id(client):
    return client.post("/api/cases", json={"title": "Case"}).json()["id"]


def upload(client, case_id, pdf_or_bytes, name="doc.pdf"):
    data = pdf_or_bytes.read_bytes() if hasattr(pdf_or_bytes, "read_bytes") else pdf_or_bytes
    return client.post(f"/api/cases/{case_id}/documents", files={"file": (name, data, "application/pdf")})


def upload_ok(client, case_id, pdf, name="doc.pdf"):
    r = upload(client, case_id, pdf, name)
    assert r.status_code == 200, r.text
    assert jobs.wait(r.json()["job_id"])["status"] == "done"
    return r.json()["document"]["id"]


def no_leftovers(case_id):
    docs = case_dir(case_id) / "docs"
    return not docs.exists() or list(docs.iterdir()) == []


def test_upload_returns_at_once_and_indexes_in_the_background(client, case_id, eu_style_pdf):
    r = upload(client, case_id, eu_style_pdf, "Statute.pdf")
    body = r.json()
    assert r.status_code == 200 and body["document"]["filename"] == "Statute.pdf" and body["document"]["status"] in ("queued", "indexing")
    doc_id = body["document"]["id"]
    job = jobs.wait(body["job_id"])
    assert job["status"] == "done" and job["progress"] == 1.0 and job["meta"] == {"case_id": case_id, "document_id": doc_id}
    assert client.get(f"/api/jobs/{body['job_id']}").json()["status"] == "done"
    doc = client.get(f"/api/cases/{case_id}/documents").json()[0]
    assert doc["status"] == "indexed" and doc["page_count"] == 3 and doc["n_chunks"] > 5 and doc["n_scanned"] == 0
    assert vector_store.count(case_id) == doc["n_chunks"]
    assert [d["id"] for d in client.get(f"/api/cases/{case_id}").json()["documents"]] == [doc_id]


def test_the_stored_file_name_is_sanitised(client, case_id, eu_style_pdf, tmp_path):
    for evil, safe in [("../../evil.pdf", "evil.pdf"), ("C:\\Windows\\sys$tem.pdf", "sys_tem.pdf")]:
        doc_id = upload_ok(client, case_id, eu_style_pdf if evil.startswith("..") else _other_pdf(tmp_path), evil)
        assert db.get_document(doc_id)["filename"] == safe
    storage = get_settings().storage_dir
    assert not (storage.parent / "evil.pdf").exists() and not (storage.parent.parent / "evil.pdf").exists()
    assert list(storage.rglob("evil.pdf")) == [] and list(storage.rglob("sys*.pdf")) == []   # only the display name is kept
    assert all(p.name == "original.pdf" for p in case_dir(case_id).rglob("*.pdf"))


def _other_pdf(tmp_path):
    from tests.conftest import make_pdf
    return make_pdf(tmp_path / "other.pdf", [[(60, 700, "A different document with some readable text on it.")]])


def test_a_name_that_is_not_pdf_is_rejected_with_415(client, case_id, eu_style_pdf):
    r = upload(client, case_id, eu_style_pdf, "report.docx")
    assert r.status_code == 415 and "Only .pdf" in r.json()["detail"] and no_leftovers(case_id) and db.list_documents(case_id) == []


def test_a_file_that_is_not_a_pdf_is_rejected_with_422(client, case_id):
    r = upload(client, case_id, b"<html>not a pdf</html>", "fake.pdf")
    assert r.status_code == 422 and "%PDF" in r.json()["detail"] and no_leftovers(case_id) and db.list_documents(case_id) == []


def test_an_encrypted_pdf_is_rejected(client, case_id, eu_style_pdf, tmp_path):
    locked = tmp_path / "locked.pdf"
    d = pymupdf.open(eu_style_pdf)
    d.save(locked, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="a", owner_pw="b")
    d.close()
    assert upload(client, case_id, locked).status_code == 422 and no_leftovers(case_id)


def test_an_oversized_upload_is_cut_off_with_413(client, case_id, monkeypatch):
    monkeypatch.setenv("LEXORA_MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    r = upload(client, case_id, b"%PDF-1.4\n" + b"0" * (2 * 1024 * 1024))
    assert r.status_code == 413 and "1 MB" in r.json()["detail"] and no_leftovers(case_id) and db.list_documents(case_id) == []


def test_the_same_file_twice_is_a_409_and_the_first_stays(client, case_id, eu_style_pdf):
    doc_id = upload_ok(client, case_id, eu_style_pdf, "a.pdf")
    r = upload(client, case_id, eu_style_pdf, "renamed.pdf")                 # same bytes, other name
    assert r.status_code == 409 and "duplicate" in r.json()["detail"]
    assert [d["id"] for d in db.list_documents(case_id)] == [doc_id] and len(list((case_dir(case_id) / "docs").iterdir())) == 1


def test_a_case_holds_a_limited_number_of_documents(client, case_id, eu_style_pdf, tmp_path, monkeypatch):
    monkeypatch.setenv("LEXORA_MAX_DOCS_PER_CASE", "1")
    get_settings.cache_clear()
    upload_ok(client, case_id, eu_style_pdf)
    r = upload(client, case_id, _other_pdf(tmp_path), "two.pdf")
    assert r.status_code == 400 and "at most 1" in r.json()["detail"] and len(db.list_documents(case_id)) == 1


def test_upload_to_an_unknown_case_is_404(client, eu_style_pdf):
    assert upload(client, "nope", eu_style_pdf).status_code == 404
    assert upload(client, "bad..id", eu_style_pdf).status_code == 400


def test_deleting_a_document_removes_it_everywhere_but_keeps_the_others(client, case_id, eu_style_pdf, contract_pdf):
    a = upload_ok(client, case_id, eu_style_pdf, "Statute.pdf")
    b = upload_ok(client, case_id, contract_pdf, "Contract.pdf")
    n_b = db.get_document(b)["n_chunks"]
    assert client.delete(f"/api/cases/{case_id}/documents/{a}").json() == {"deleted": a}
    assert db.get_document(a) is None and not doc_dir(case_id, a).exists()
    assert {c.document_id for c in indexer.case_chunks(case_id)} == {b} and vector_store.count(case_id) == n_b
    assert client.delete(f"/api/cases/{case_id}/documents/{a}").status_code == 404
    other = client.post("/api/cases", json={"title": "Other"}).json()["id"]
    assert client.delete(f"/api/cases/{other}/documents/{b}").status_code == 404          # wrong case
    assert client.delete(f"/api/cases/{case_id}/documents/..%2Fx").status_code in (400, 404)


def test_reindex_runs_as_a_job_and_can_force_a_new_extraction(client, case_id, eu_style_pdf, monkeypatch):
    doc_id = upload_ok(client, case_id, eu_style_pdf)
    calls = []
    real = indexer.parse_pdf
    monkeypatch.setattr(indexer, "parse_pdf", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    assert jobs.wait(client.post(f"/api/cases/{case_id}/documents/{doc_id}/reindex").json()["job_id"])["status"] == "done"
    assert calls == []                                                                     # cache reused
    assert jobs.wait(client.post(f"/api/cases/{case_id}/documents/{doc_id}/reindex?force_reparse=true").json()["job_id"])["status"] == "done"
    assert calls == [1] and db.get_document(doc_id)["status"] == "indexed"


def test_a_failing_index_job_is_visible_through_the_job_endpoint(client, case_id, eu_style_pdf, monkeypatch):
    monkeypatch.setattr(indexer, "index_document", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model missing")))
    body = upload(client, case_id, eu_style_pdf).json()
    job = jobs.wait(body["job_id"])
    assert job["status"] == "failed" and job["error"] == "model missing"
    assert client.get(f"/api/jobs/{body['job_id']}").json()["error"] == "model missing"


def test_unknown_job(client):
    assert client.get("/api/jobs/nope").status_code == 404
