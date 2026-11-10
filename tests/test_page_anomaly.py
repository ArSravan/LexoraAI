import json
import random

import numpy as np
import pytest

from evaluation import page_anomaly_eval as pae
from ingestion import page_anomaly as pa
from schemas import ParsedPage, TextLine
from services import indexer
from tests.conftest import make_pdf

SENTENCES = ["The controller shall notify the supervisory authority without undue delay.",
             "Member States shall lay down the rules on penalties applicable to infringements.",
             "Processing shall be lawful only if at least one of the following applies.",
             "The data subject shall have the right to obtain confirmation from the controller.",
             "Each supervisory authority shall be competent for the performance of its tasks.",
             "Personal data shall be adequate relevant and limited to what is necessary.",
             "The Commission shall adopt delegated acts in accordance with Article 92 of this Regulation."]


def make_page(i: int, rng: random.Random, n_lines: int = 30) -> ParsedPage:
    lines = [TextLine(text=f"{rng.choice(SENTENCES)} {rng.choice(SENTENCES)[:30]}", page_index=i,
                      bbox=(60.0, 60.0 + 20 * k, 540.0, 72.0 + 20 * k)) for k in range(n_lines)]
    return ParsedPage(page_index=i, page_number=i + 1, width=595.0, height=842.0, lines=lines,
                      char_count=sum(len(l.text) for l in lines))


@pytest.fixture
def document():
    rng = random.Random(7)
    return [make_page(i, rng, rng.randint(26, 34)) for i in range(40)]


def test_features_have_a_fixed_length_and_survive_an_empty_page():
    empty = ParsedPage(page_index=0, page_number=1, width=595, height=842)
    f = pa.page_features(empty)
    assert f.shape == (len(pa.FEATURES),) and np.isfinite(f).all()


def test_features_separate_readable_text_from_garbage(document):
    good = pa.page_features(document[0])
    garbled = pa.page_features(pae.perturb(document[0], "garbled", random.Random(1)))
    names = pa.FEATURES
    assert garbled[names.index("vowelless_word_ratio")] > good[names.index("vowelless_word_ratio")] + 0.05
    noisy = pa.page_features(pae.perturb(document[0], "symbol noise", random.Random(1)))
    assert noisy[names.index("symbol_ratio")] > good[names.index("symbol_ratio")] + 0.05
    ocr_page = document[0].model_copy(update={"lines": [l.model_copy(update={"confidence": 60.0}) for l in document[0].lines]})
    assert pa.page_features(ocr_page)[names.index("ocr_confidence")] == pytest.approx(0.6)


def test_every_defect_kind_changes_the_page(document):
    for kind in pae.KINDS[:5]:
        p = pae.perturb(document[3], kind, random.Random(3))
        assert [l.text for l in p.lines] != [l.text for l in document[3].lines], kind
        assert not np.allclose(pa.page_features(p), pa.page_features(document[3])), kind
    with pytest.raises(ValueError):
        pae.perturb(document[0], "nonsense", random.Random(0))


def test_scoring_is_deterministic(document):
    X = np.array([pa.page_features(p) for p in document])
    assert np.array_equal(pa.reconstruction_errors(X), pa.reconstruction_errors(X))
    assert not np.array_equal(pa.reconstruction_errors(X, seed=1), pa.reconstruction_errors(X, seed=2))


def test_injected_defective_pages_get_the_highest_scores_in_a_contaminated_document(document):
    rng = random.Random(0)
    bad = {5: "garbled", 17: "number dump", 29: "near empty"}
    pages = [pae.perturb(p, bad[i], rng) if i in bad else p for i, p in enumerate(document)]
    scores = pa.score_document(pages)
    top3 = sorted(range(len(pages)), key=lambda i: -scores[i])[:3]
    assert set(top3) == set(bad)
    assert all(scores[i] > 1 for i in bad)
    assert sum(s > 1 for i, s in enumerate(scores) if i not in bad) <= 4        # few false alarms among 37 intact pages


def test_a_clean_document_raises_few_alarms(document):
    scores = pa.score_document(document)
    assert len(scores) == 40 and sum(s > 1 for s in scores) <= 6


def test_short_documents_are_not_scored_and_failures_never_propagate(document, monkeypatch):
    assert pa.score_document(document[: pa.MIN_PAGES - 1]) is None
    monkeypatch.setattr(pa, "scores_from_features", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert pa.score_document(document) is None


def test_auroc_and_choose_k():
    assert pae.auroc(np.array([1, 2, 3]), np.array([4, 5])) == 1.0
    assert pae.auroc(np.array([4, 5]), np.array([1, 2, 3])) == 0.0
    assert pae.auroc(np.array([1, 2]), np.array([1, 2])) == 0.5
    assert np.isnan(pae.auroc(np.array([]), np.array([1.0])))
    assert pae.choose_k({3.0: 0.2, 4.0: 0.08, 8.0: 0.01}) == 4.0
    assert pae.choose_k({3.0: 0.5, 4.0: 0.4}) == 4.0                           # none qualifies: the most cautious K


def test_robust_threshold_ignores_a_few_outliers():
    base = np.abs(np.random.RandomState(0).normal(1.0, 0.1, 200))
    contaminated = np.concatenate([base, [50.0, 80.0, 100.0]])
    assert pa.robust_threshold(contaminated) == pytest.approx(pa.robust_threshold(base), rel=0.1)


def test_the_indexer_stores_scores_and_the_info_endpoint_exposes_them(fake_embedder, tmp_path):
    from fastapi.testclient import TestClient
    from api.app import create_app
    from services import jobs
    rng = random.Random(11)
    pages = []
    for i in range(14):
        items = [(60, 780 - 18 * k, rng.choice(SENTENCES)) for k in range(36)]
        pages.append(items)
    pages[6] = [(60, 780 - 18 * k, " ".join(str(rng.randint(0, 99999)) for _ in range(8))) for k in range(36)]
    pdf = make_pdf(tmp_path / "Long.pdf", pages)
    client = TestClient(create_app())
    cid = client.post("/api/cases", json={"title": "c"}).json()["id"]
    up = client.post(f"/api/cases/{cid}/documents", files={"file": ("Long.pdf", pdf.read_bytes(), "application/pdf")}).json()
    assert jobs.wait(up["job_id"], 120)["status"] == "done"
    doc_id = up["document"]["id"]
    scores = indexer.load_anomaly_scores(cid, doc_id)
    assert len(scores) == 14 and scores[6] == max(scores) and scores[6] > 1
    info = client.get(f"/api/documents/{doc_id}/info").json()
    assert info["pages"][6]["anomalous"] is True and info["pages"][6]["anomaly_score"] == scores[6]
    assert all(set(p) >= {"anomaly_score", "anomalous"} for p in info["pages"])


def test_short_documents_have_no_scores_in_the_info_endpoint(fake_embedder, eu_style_pdf):
    from fastapi.testclient import TestClient
    from api.app import create_app
    from services import jobs
    client = TestClient(create_app())
    cid = client.post("/api/cases", json={"title": "c"}).json()["id"]
    up = client.post(f"/api/cases/{cid}/documents", files={"file": ("S.pdf", eu_style_pdf.read_bytes(), "application/pdf")}).json()
    assert jobs.wait(up["job_id"], 120)["status"] == "done"
    assert indexer.load_anomaly_scores(cid, up["document"]["id"]) is None
    pages = client.get(f"/api/documents/{up['document']['id']}/info").json()["pages"]
    assert all(p["anomaly_score"] is None and p["anomalous"] is False for p in pages)


def test_corrupt_score_file_is_ignored(fake_embedder, tmp_path):
    from services.paths import doc_dir
    d = doc_dir("case1", "doc1")
    d.mkdir(parents=True)
    (d / "anomaly.json").write_text("{not json", encoding="utf-8")
    assert indexer.load_anomaly_scores("case1", "doc1") is None
    (d / "anomaly.json").write_text(json.dumps([0.5, 2.0]), encoding="utf-8")
    assert indexer.load_anomaly_scores("case1", "doc1") == [0.5, 2.0]
