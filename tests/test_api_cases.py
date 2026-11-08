import pytest
from fastapi.testclient import TestClient

from api import routes
from api.app import create_app
from config import get_settings
from services import db


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LEXORA_OLLAMA_BASE_URL", "http://127.0.0.1:9")        # nothing listens there
    get_settings.cache_clear()
    routes._ocr_available.cache_clear()
    return TestClient(create_app())


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_status_reports_ollama_models_and_ocr(client, monkeypatch):
    st = client.get("/api/status").json()
    assert st["api"] == "ok" and st["ollama"]["connected"] is False and st["ollama"]["model"] == "qwen3.5:9b"
    assert st["settings"]["max_docs_per_case"] == 5 and st["settings"]["rerank_enabled"] is True
    assert isinstance(st["ocr_available"], bool)
    monkeypatch.setattr(routes.ocr_mod, "configure_tesseract", lambda cmd=None: True)
    routes._ocr_available.cache_clear()
    assert client.get("/api/status").json()["ocr_available"] is True


def test_case_crud(client):
    r = client.post("/api/cases", json={"title": "Acme vs Beta", "description": "fees"})
    assert r.status_code == 200
    case = r.json()
    assert case["title"] == "Acme vs Beta" and client.get("/api/cases").json()[0]["id"] == case["id"]
    got = client.get(f"/api/cases/{case['id']}").json()
    assert got["documents"] == [] and got["description"] == "fees"
    assert client.delete(f"/api/cases/{case['id']}").json() == {"deleted": case["id"]}
    assert client.get(f"/api/cases/{case['id']}").status_code == 404 and client.get("/api/cases").json() == []


@pytest.mark.parametrize("body", [{}, {"title": ""}, {"title": "   "}, {"title": "x" * 201}, {"title": "ok", "description": "d" * 2001}])
def test_invalid_cases_are_rejected(client, body):
    assert client.post("/api/cases", json=body).status_code == 422 and db.list_cases() == []


def test_unknown_and_hostile_case_ids(client):
    assert client.get("/api/cases/doesnotexist").status_code == 404
    assert client.delete("/api/cases/doesnotexist").status_code == 404
    for bad in ("..%2F..%2Fetc", "a b", "x" * 65):
        assert client.get(f"/api/cases/{bad}").status_code in (400, 404)
    assert client.get("/api/cases/bad..id").status_code == 400


def test_deleting_a_case_removes_its_files_and_records(client, tmp_path):
    from services.paths import case_dir
    cid = client.post("/api/cases", json={"title": "C"}).json()["id"]
    case_dir(cid).mkdir(parents=True)
    (case_dir(cid) / "marker.txt").write_text("x")
    client.delete(f"/api/cases/{cid}")
    assert not case_dir(cid).exists()


def test_stats(client):
    client.post("/api/cases", json={"title": "C"})
    s = client.get("/api/stats").json()
    assert s["cases"] == 1 and s["documents"] == 0 and s["questions"] == 0 and s["recent"] == []


def test_the_openapi_description_lists_the_api_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/health", "/api/status", "/api/cases", "/api/cases/{case_id}"} <= set(paths)
