import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.app import create_app
from services import jobs


@pytest.fixture
def client(fake_embedder):
    return TestClient(create_app())


@pytest.fixture
def docs(client, eu_style_pdf, contract_pdf):
    cid = client.post("/api/cases", json={"title": "C"}).json()["id"]
    out = {"case": cid}
    for key, pdf, name in (("statute", eu_style_pdf, "Statute.pdf"), ("contract", contract_pdf, "Contract.pdf")):
        r = client.post(f"/api/cases/{cid}/documents", files={"file": (name, pdf.read_bytes(), "application/pdf")}).json()
        assert jobs.wait(r["job_id"])["status"] == "done"
        out[key] = r["document"]["id"]
    return out


def test_document_info_summarises_pages_and_structure(client, docs):
    info = client.get(f"/api/documents/{docs['statute']}/info").json()
    assert info["document"]["filename"] == "Statute.pdf" and info["document"]["status"] == "indexed"
    assert [p["page_number"] for p in info["pages"]] == [1, 2, 3]
    assert all(p["char_count"] > 50 and p["is_scanned"] is False and p["method"] == "digital" for p in info["pages"])
    assert info["structure"]["mode"] == "legal-structure" and "3(2)(a)" in info["structure"]["labels"] and "headings" not in info["structure"]
    assert {h["label"] for h in info["headings"] if h["type"] == "article"} == {"1", "2", "3", "4"}


def test_page_endpoint_returns_text_chunks_and_tables(client, docs):
    p2 = client.get(f"/api/documents/{docs['statute']}/pages/2").json()
    assert "Article 3" in p2["text"] and "Official Journal" not in p2["text"]            # header/footer were removed
    assert {"3(2)", "2"} <= {c["clause"] for c in p2["chunks"]} and p2["is_scanned"] is False and p2["tables"] == []
    c = client.get(f"/api/documents/{docs['contract']}/pages/1").json()
    assert c["tables"] and c["tables"][0]["caption"].startswith("Table 9.1") and c["tables"][0]["rows"][0] == ["Item", "Amount (EUR)"]
    assert any(ch["chunk_type"] == "table" for ch in c["chunks"])


def test_missing_pages_and_documents(client, docs):
    for n in (0, 4, 99, -1):
        assert client.get(f"/api/documents/{docs['statute']}/pages/{n}").status_code == 404
        assert client.get(f"/api/documents/{docs['statute']}/pages/{n}/image").status_code == 404
    assert client.get("/api/documents/nope/info").status_code == 404
    assert client.get("/api/documents/bad..id/info").status_code == 400
    assert client.get("/api/documents/nope/pages/1/image").status_code == 404


def image(resp) -> np.ndarray:
    assert resp.headers["content-type"] == "image/png" and resp.content[:4] == b"\x89PNG"
    return np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGB"))


def yellow(a) -> int:
    return int(((a[..., 0] > 200) & (a[..., 1] > 170) & (a[..., 2] < 120)).sum())


def test_the_page_image_highlights_the_quote_and_reports_it(client, docs):
    base = f"/api/documents/{docs['statute']}/pages/2/image"
    plain = client.get(base)
    assert json.loads(plain.headers["x-highlight"])["highlighted"] is False and yellow(image(plain)) == 0
    hit = client.get(base, params={"quote": "administrative fines up to 10 000 000 EUR"})
    assert json.loads(hit.headers["x-highlight"])["highlighted"] is True and yellow(image(hit)) > 300
    assert hit.headers["cache-control"] == "no-store"
    miss = client.get(base, params={"quote": "a sentence that is not on this page at all"})
    assert json.loads(miss.headers["x-highlight"])["highlighted"] is False


def test_the_chunk_region_is_outlined_and_zoom_is_bounded(client, docs):
    chunk = next(c for c in client.get(f"/api/documents/{docs['statute']}/pages/2").json()["chunks"] if c["clause"] == "3(2)")
    base = f"/api/documents/{docs['statute']}/pages/2/image"
    r = client.get(base, params={"chunk_id": chunk["chunk_id"]})
    assert json.loads(r.headers["x-highlight"])["region"] is True
    assert json.loads(client.get(base, params={"chunk_id": "nonexistent"}).headers["x-highlight"])["region"] is False
    small, big = image(client.get(base, params={"zoom": 1.0})), image(client.get(base, params={"zoom": 2.0}))
    assert abs(big.shape[0] - 2 * small.shape[0]) <= 2
    assert client.get(base, params={"zoom": 0.1}).status_code == 422 and client.get(base, params={"zoom": 9}).status_code == 422


def test_a_chunk_of_another_document_does_not_leak_a_region(client, docs):
    other = client.get(f"/api/documents/{docs['contract']}/pages/1").json()["chunks"][0]["chunk_id"]
    r = client.get(f"/api/documents/{docs['statute']}/pages/1/image", params={"chunk_id": other})
    assert json.loads(r.headers["x-highlight"])["region"] is False


def test_pdf_files_are_never_served_directly(client, docs):
    for path in (f"/api/documents/{docs['statute']}/file", f"/api/documents/{docs['statute']}/original.pdf",
                 f"/api/cases/{docs['case']}/documents/{docs['statute']}/download"):
        assert client.get(path).status_code in (404, 405)
