"""Dashboard, Evaluation and Settings: the helpers that shape the data, the two new API endpoints, and the pages rendered
by Streamlit's headless runner against a live API."""
import json

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api import routes
from api.app import create_app
from config import Settings
from frontend import ui
from tests.test_frontend_views import live_api  # noqa: F401  (a fixture)

SUMMARY = {
    "meta": {"date": "2026-11-08 22:31:38 UTC", "ollama_model": "m", "seed": 42, "n_questions": 15, "n_answerable": 11, "git_commit": "abc"},
    "retrieval": {"BM25 only": {"recall@1": 0.55, "recall@3": 0.73, "recall@5": 0.85, "full@5": 0.73, "precision@5": 0.27, "mrr": 0.75, "latency_ms": 21.4}},
    "generation": {"no_retrieval": {"answer_correct_proxy": 0.55, "abstention_accuracy": 0.73, "mean_total_s": 16.1},
                   "hybrid": {"answer_correct_proxy": 0.73, "citations_verified_rate": 0.96, "unsupported_claim_rate": 0.04, "mean_total_s": 70.0}},
    "manual": {"grader": "Sravan", "configs": {"hybrid": {"n_graded": 15, "n_total": 15, "answerable_mean_score": 0.91, "answerable_strictly_correct": 0.91,
                                                          "unanswerable_handled_correctly": 1.0, "behaviour_correct_all": 0.93}}},
    "region": {"config": "hybrid", "n_citations": 30, "located_rate": 0.97, "page_precision": 0.9, "region_precision": 0.7, "region_precision_of_located": 0.72},
}
OCR = {"summary": {"clean": {"pages": 6, "cer": 0.005, "wer": 0.018, "word_recall": 0.986, "ocr_confidence": 94.3}}}
TABLES = {"cases": [{"case": "plain grid", "tables_found": 1, "tables_expected": 1, "cells_right": 20, "cells_expected": 20}]}
AE = {"chosen_k": 3.0, "defects": [{"defect": "garbled", "auroc_autoencoder": 0.878, "auroc_zscore": 0.99,
                                    "detected": {"3.0": 0.46}, "false_alarms_on_intact_pages": {"3.0": 0.05}}]}
ABL = {"modes": {"bm25": {"text only": {"recall@1": 1.0, "recall@3": 1.0, "recall@5": 1.0, "mrr": 1.0}}}}
LAT = {"configs": {"hybrid": {"n": 15, "total_s_mean": 70.0, "total_s_median": 71.0, "total_s_p95": 90.0, "generation_s_mean": 69.0, "completion_tokens_mean": 447}}}


# ------------------------------------------------------------------------------ helpers
def test_helpers_shape_every_result_file():
    assert ui.retrieval_rows(SUMMARY["retrieval"])[0] == {"Configuration": "BM25 only", "recall@1": "0.55", "recall@3": "0.73", "recall@5": "0.85",
                                                           "full@5": "0.73", "precision@5": "0.27", "mrr": "0.75", "latency ms": "21"}
    gen = {r["Measure"]: r for r in ui.generation_rows(SUMMARY["generation"])}
    assert gen["Citations verified by the validator"] == {"Measure": "Citations verified by the validator", "no_retrieval": "-", "hybrid": "0.96"}
    assert ui.manual_rows(SUMMARY["manual"])[0] == {"Measure": "Answers graded", "hybrid": "15 / 15"}
    assert ui.region_rows(SUMMARY["region"])[3] == {"Measure": "Region-level precision", "Value": "0.70"}
    assert ui.ocr_rows(OCR)[0]["CER"] == "0.005" and ui.table_eval_rows(TABLES)[0]["Cells right / expected"] == "20 / 20"
    assert ui.anomaly_rows(AE)[0] == {"Defect": "garbled", "AUROC autoencoder": "0.878", "AUROC baseline": "0.990", "Detected": "46%",
                                      "False alarms on intact pages": "5.0%"}
    assert ui.ablation_rows(ABL)[0]["Index"] == "text only" and ui.latency_rows(LAT)[0]["Mean s"] == 70.0


def test_missing_results_give_empty_tables_not_errors():
    for fn in (ui.manual_rows, ui.region_rows, ui.ocr_rows, ui.table_eval_rows, ui.anomaly_rows, ui.ablation_rows, ui.latency_rows):
        assert fn(None) == []
    assert ui.region_rows({"n_citations": 0}) == [] and ui.retrieval_rows({}) == [] and ui.generation_rows({}) [0]["Measure"]


def test_settings_are_grouped_and_nothing_is_lost():
    rows = [{"name": f, "env": "LEXORA_" + f.upper(), "value": "x"} for f in Settings.__dataclass_fields__]
    groups = ui.settings_groups(rows)
    assert {"Storage and upload", "Local language model", "Retrieval", "Verification and abstention"} <= set(groups)
    shown = [item["Setting"] for g in groups.values() for item in g]
    assert sorted(shown) == sorted(Settings.__dataclass_fields__) and len(shown) == len(set(shown))
    assert groups["Local language model"][0]["Environment variable"].startswith("LEXORA_OLLAMA_")


def test_dashboard_helpers():
    stats = {"cases": 2, "documents": 3, "chunks": 400, "questions": 7, "abstentions": 2,
             "recent": [{"created_at": 1_700_000_000, "question": "q" * 200, "confidence": "HIGH", "case_title": "C", "mode": "hybrid", "model": "m", "latency": 69.04}]}
    assert ui.stat_rows(stats)[0] == ("Cases", 2) and len(ui.stat_rows(stats)) == 5
    r = ui.recent_rows(stats["recent"])[0]
    assert len(r["Question"]) == 110 and r["Seconds"] == 69.0 and r["Case"] == "C"


def test_page_rows_flag_unusual_pages():
    base = {"page_number": 1, "char_count": 9, "method": "digital", "is_scanned": False, "ocr_confidence": None, "n_tables": 0, "warning": ""}
    rows = ui.page_rows([{**base, "anomalous": True}, base])
    assert rows[0]["Unusual page"] == "⚠ yes" and rows[1]["Unusual page"] == ""


# ------------------------------------------------------------------------------ API
def test_settings_endpoint_lists_every_setting_with_its_variable(fake_embedder):
    rows = TestClient(create_app()).get("/api/settings").json()
    assert {r["name"] for r in rows} == set(Settings.__dataclass_fields__)
    assert next(r for r in rows if r["name"] == "max_upload_mb") == {"name": "max_upload_mb", "env": "LEXORA_MAX_UPLOAD_MB", "value": "50"}


def test_evaluation_endpoint_returns_existing_files_and_none_for_the_rest(fake_embedder, tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "EVAL_RESULTS", tmp_path)
    (tmp_path / "summary.json").write_text(json.dumps(SUMMARY), encoding="utf-8")
    (tmp_path / "ocr_eval.json").write_text("{half written", encoding="utf-8")
    (tmp_path / "secret.json").write_text("{}", encoding="utf-8")
    res = TestClient(create_app()).get("/api/evaluation").json()
    assert res["summary"]["meta"]["seed"] == 42 and res["ocr_eval"] is None and res["table_eval"] is None
    assert "secret" not in res and set(res) == {n.removesuffix(".json") for n in routes.EVAL_FILES}


# ------------------------------------------------------------------------------ the pages
def page(fn_name: str, module: str) -> AppTest:
    def script(module, fn_name):
        import importlib
        getattr(importlib.import_module(f"frontend.views.{module}"), fn_name)()
    at = AppTest.from_function(script, args=(module, fn_name), default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_dashboard_page_shows_the_counters_and_the_ollama_warning(live_api):
    at = page("render", "dashboard")
    assert [m.label for m in at.metric] == ["Cases", "Documents indexed", "Indexed chunks", "Questions asked", "Abstentions"]
    assert any("No questions asked yet" in i.value for i in at.info)
    assert any("Ollama is not running" in w.value for w in at.warning)


def test_settings_page_lists_the_groups_and_the_privacy_note(live_api):
    at = page("render", "settings")
    assert {s.value for s in at.subheader} >= {"Storage and upload", "Retrieval", "Local language model"}
    assert len(at.dataframe) >= 5 and any("Privacy" in i.value for i in at.info)
    at.button[0].click().run()
    assert any("Cannot reach Ollama" in e.value for e in at.error)


def test_evaluation_page_without_results_says_how_to_produce_them(live_api, tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "EVAL_RESULTS", tmp_path)
    at = page("render", "evaluation")
    assert any("python -m evaluation.evaluate" in i.value for i in at.info)
    assert any("python -m evaluation.ocr_eval" in i.value for i in at.info) and len(at.tabs) == 3


def test_evaluation_page_with_results_shows_every_table(live_api, tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "EVAL_RESULTS", tmp_path)
    for name, data in {"summary": SUMMARY, "ocr_eval": OCR, "table_eval": TABLES, "page_anomaly_eval": AE,
                       "ablation_tables": ABL, "latency_report": LAT}.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
    at = page("render", "evaluation")
    assert len(at.dataframe) == 9                      # retrieval, answers, manual, region, ocr, tables, anomaly, ablation, latency
    assert any("Graded by" in s.value or "Sravan" in s.value for s in at.subheader)
    assert "commit abc" in " ".join(c.value for c in at.caption)
    assert not any("Not run yet" in i.value for i in at.info)
