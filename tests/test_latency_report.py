import csv
import json

import pytest

from evaluation import bench_indexing, latency_report as lr


def write_rows(path, rows):
    cols = ["id", "config", "answerable", "abstained", "refusal_stage", "retrieval_s", "rerank_s", "generation_s", "total_s",
            "prompt_tokens", "completion_tokens"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{c: r.get(c, "") for c in cols} for r in rows])


def row(config, i, answerable=True, abstained=False, stage="", gen=60.0, total=None, comp=400, retr=0.5, rer=0.4):
    total = total if total is not None else (retr + rer + gen)
    return {"id": f"Q{i}", "config": config, "answerable": answerable, "abstained": abstained, "refusal_stage": stage,
            "retrieval_s": retr, "rerank_s": rer, "generation_s": gen, "total_s": total, "prompt_tokens": 1500, "completion_tokens": comp}


@pytest.fixture
def csv_file(tmp_path):
    rows = ([row("hybrid", i, gen=60 + i, comp=300 + 10 * i) for i in range(6)]
            + [row("hybrid", 10 + i, answerable=False, abstained=True, stage="gate", gen=0, total=1.0, comp=0) for i in range(4)]
            + [row("no_retrieval", i, gen=15.0, total=15.0, comp=95, retr=0, rer=0) for i in range(10)])
    p = tmp_path / "gen.csv"
    write_rows(p, rows)
    return p


def test_percentile_interpolates():
    assert lr.percentile([1, 2, 3, 4], 0.5) == 2.5 and lr.percentile([5], 0.95) == 5 and lr.percentile([], 0.5) == 0.0
    assert lr.percentile([0, 10], 0.95) == pytest.approx(9.5)


def test_per_configuration_statistics(csv_file):
    cfg = lr.per_config(lr.load_rows(csv_file))
    h, b = cfg["hybrid"], cfg["no_retrieval"]
    assert h["n"] == 10 and b["n"] == 10
    assert b["total_s_mean"] == 15.0 and b["retrieval_rerank_s_mean"] == 0.0 and b["tokens_per_second"] == pytest.approx(95 / 15, abs=0.01)
    assert h["completion_tokens_total"] == sum(300 + 10 * i for i in range(6))
    assert h["total_s_p95"] >= h["total_s_median"] and h["run_seconds_total"] > 0


def test_the_gate_saving_is_estimated_from_answered_questions_and_the_baseline_has_none(csv_file):
    gate = lr.gate_saving(lr.load_rows(csv_file))
    assert set(gate) == {"hybrid"}
    g = gate["hybrid"]
    assert g["questions_refused_at_the_gate"] == 4 and g["mean_seconds_when_refused"] == 1.0
    answered = [0.9 + 60 + i for i in range(6)]
    assert g["mean_seconds_when_answered"] == pytest.approx(sum(answered) / 6, abs=0.01)
    assert g["estimated_seconds_saved"] == pytest.approx(4 * (sum(answered) / 6 - 1.0), abs=0.2)
    assert 0 < g["share_of_the_run"] < 1


def test_no_gate_rows_means_no_gate_section(tmp_path):
    p = tmp_path / "g.csv"
    write_rows(p, [row("hybrid", i) for i in range(3)])
    rep = lr.report(lr.load_rows(p))
    assert rep["gate"] == {} and "Optimisation 1" not in lr.markdown(rep)


def test_main_writes_the_report_and_complains_about_a_missing_file(csv_file, tmp_path, capsys):
    assert lr.main(["--csv", str(csv_file), "--out", str(tmp_path / "o")]) == 0
    assert json.loads((tmp_path / "o" / "latency_report.json").read_text())["gate"]["hybrid"]["questions_refused_at_the_gate"] == 4
    md = (tmp_path / "o" / "latency_report.md").read_text()
    assert "| hybrid |" in md and "Optimisation 1" in md and "estimate" in md
    assert lr.main(["--csv", str(tmp_path / "missing.csv")]) == 2


def test_the_embedding_cache_benchmark_runs_and_the_warm_pass_costs_no_model_call(fake_embedder, simple_pdf, tmp_path, monkeypatch):
    r = bench_indexing.bench(simple_pdf)
    assert r["document"] == "simple.pdf" and r["chunks"] >= 1 and r["embed_speedup"] >= 1
    n = len(fake_embedder.encoded)
    assert n == r["chunks"]                                   # each chunk text went through the model exactly once
    md = bench_indexing.markdown(r)
    assert "| Embed the same chunks again, cache warm |" in md
