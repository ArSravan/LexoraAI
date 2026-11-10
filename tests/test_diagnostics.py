import json
from pathlib import Path

import pytest

from evaluation import diagnostics as dg
from tests.test_evaluate import QUESTIONS, case, env, questions  # noqa: F401  (fixtures and helpers)


def test_features_report_the_gate_numbers_and_where_the_labelled_evidence_ranks(env):
    rows = dg.features_with_gold(case(env), questions())
    by = {r["id"]: r for r in rows}
    assert set(by) == {"F1", "F2", "U1"} and by["U1"]["gold_ranks"] == [] and by["U1"]["answerable"] is False
    f1 = by["F1"]
    assert f1["top_score"] is not None and 0 <= f1["coverage"] <= 1
    assert len(f1["gold_ranks"]) == 1 and f1["gold_ranks"][0] in (1, 2, 3, 4, 5, None)
    assert all(isinstance(x, float) for x in f1["gold_rerank_scores"])


def test_a_higher_threshold_can_only_refuse_more(env):
    rows = dg.features_with_gold(case(env), questions())
    sw = dg.sweep(rows, 0.0)
    assert [s["rerank_min"] for s in sw] == list(dg.THRESHOLDS)
    fa = [s["false_abstention"] for s in sw]
    ca = [s["correct_abstention"] for s in sw]
    assert fa == sorted(fa) and ca == sorted(ca)
    assert sw[0]["false_abstention"] <= sw[-1]["false_abstention"]
    all_refused = dg.sweep(rows, 0.0, thresholds=(1000.0,))[0]                     # a huge threshold refuses everything
    assert all_refused["false_abstention"] == 1.0 and all_refused["correct_abstention"] == 1.0


def test_false_abstentions_lists_only_refused_answerable_questions():
    rows = [{"id": "A", "answerable": True, "top_score": 0.6, "coverage": 0.9, "gold_ranks": [2], "gold_rerank_scores": [0.6]},
            {"id": "B", "answerable": True, "top_score": 5.0, "coverage": 0.9, "gold_ranks": [1], "gold_rerank_scores": [5.0]},
            {"id": "C", "answerable": True, "top_score": 5.0, "coverage": 0.1, "gold_ranks": [1], "gold_rerank_scores": [5.0]},
            {"id": "D", "answerable": False, "top_score": -3.0, "coverage": 0.0, "gold_ranks": [], "gold_rerank_scores": []},
            {"id": "E", "answerable": True, "top_score": None, "coverage": 0.0, "gold_ranks": [None], "gold_rerank_scores": []}]
    assert [r["id"] for r in dg.false_abstentions(rows, 2.5, 0.32)] == ["A", "C", "E"]


def test_the_command_writes_json_and_a_markdown_table(env, tmp_path, monkeypatch):
    monkeypatch.setattr(dg, "TEST_SET", Path(env["dataset"]))                       # the tiny test dataset stands in for both sets
    monkeypatch.setattr(dg, "DEV_SET", Path(env["dataset"]))
    assert dg.main([*env["pdfs"], "--out", str(tmp_path / "o")]) == 0
    res = json.loads((tmp_path / "o" / "diagnostics.json").read_text())
    assert len(res["sweep_heldout"]) == len(dg.THRESHOLDS) and "shipped" in res
    md = (tmp_path / "o" / "diagnostics.md").read_text()
    assert "| rerank threshold |" in md and "Shipped thresholds" in md
