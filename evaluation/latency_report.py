"""Latency and cost report, with two measured optimisations (assignment extended feature 11).

    python -m evaluation.latency_report                       # reads evaluation/results/generation_by_question.csv
    python -m evaluation.bench_indexing sample_docs/*.pdf     # the second optimisation (embedding cache), see that file

Seconds and tokens per question for every configuration of the last evaluation run. The language model is local, so
there is no per-token fee: the cost is time and compute, and it is reported as seconds and tokens.

Optimisation 1: the sufficiency gate. A question the documents cannot answer is refused BEFORE the language model is
called. The report measures what that saves: the questions refused at the gate took about a second, and the model
time they would have taken is estimated with the mean time of the answered questions of the same configuration
(an estimate: the model was not actually called for them).

Optimisation 2: the embedding cache (evaluation/bench_indexing.py): the same texts are embedded once, then served from
SQLite.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

RESULTS = Path(__file__).resolve().parent / "results"
NUMERIC = ("retrieval_s", "rerank_s", "generation_s", "total_s", "prompt_tokens", "completion_tokens")


def load_rows(path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        for k in NUMERIC:
            r[k] = float(r.get(k) or 0)
        r["abstained"] = r["abstained"] == "True"
        r["answerable"] = r["answerable"] == "True"
        rows.append(r)
    return rows


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    pos = (len(s) - 1) * p
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def per_config(rows: list[dict]) -> dict:
    out = {}
    for config in dict.fromkeys(r["config"] for r in rows):
        rs = [r for r in rows if r["config"] == config]
        gen = [r for r in rs if r["generation_s"] > 0]
        tot = [r["total_s"] for r in rs]
        out[config] = {
            "n": len(rs), "total_s_mean": round(mean(tot), 2), "total_s_median": round(median(tot), 2),
            "total_s_p95": round(percentile(tot, 0.95), 2),
            "retrieval_rerank_s_mean": round(mean(r["retrieval_s"] + r["rerank_s"] for r in rs), 2),
            "generation_s_mean": round(mean(r["generation_s"] for r in rs), 2),
            "prompt_tokens_mean": round(mean(r["prompt_tokens"] for r in rs)),
            "completion_tokens_mean": round(mean(r["completion_tokens"] for r in rs)),
            "completion_tokens_total": int(sum(r["completion_tokens"] for r in rs)),
            "tokens_per_second": round(sum(r["completion_tokens"] for r in gen) / sum(r["generation_s"] for r in gen), 2) if gen else None,
            "run_seconds_total": round(sum(tot), 1)}
    return out


def gate_saving(rows: list[dict]) -> dict:
    """What the gate saved, per retrieval configuration (the baseline has no gate)."""
    out = {}
    for config in dict.fromkeys(r["config"] for r in rows):
        rs = [r for r in rows if r["config"] == config]
        gated = [r for r in rs if r["abstained"] and r.get("refusal_stage") == "gate"]
        answered = [r for r in rs if not r["abstained"] and r["generation_s"] > 0]
        if not gated or not answered:
            continue
        g, a = mean(r["total_s"] for r in gated), mean(r["total_s"] for r in answered)
        saved = len(gated) * (a - g)
        out[config] = {"questions_refused_at_the_gate": len(gated), "mean_seconds_when_refused": round(g, 2),
                       "mean_seconds_when_answered": round(a, 2), "estimated_seconds_saved": round(saved, 1),
                       "share_of_the_run": round(saved / (saved + sum(r["total_s"] for r in rs)), 4)}
    return out


def report(rows: list[dict]) -> dict:
    return {"configs": per_config(rows), "gate": gate_saving(rows)}


def markdown(rep: dict) -> str:
    head = ["| Configuration | Questions | Mean s | Median s | 95th pct s | Retrieval + rerank s | Generation s | Prompt tokens | Completion tokens | Tokens/s |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for c, v in rep["configs"].items():
        head.append(f"| {c} | {v['n']} | {v['total_s_mean']:.1f} | {v['total_s_median']:.1f} | {v['total_s_p95']:.1f} | "
                    f"{v['retrieval_rerank_s_mean']:.2f} | {v['generation_s_mean']:.1f} | {v['prompt_tokens_mean']} | "
                    f"{v['completion_tokens_mean']} | {v['tokens_per_second'] if v['tokens_per_second'] is not None else '-'} |")
    out = ["Latency and cost per question (local model on this machine: no per-token fee, the cost is time and compute).", "", *head, ""]
    if rep["gate"]:
        out += ["Optimisation 1: the sufficiency gate refuses an unanswerable question before the model is called.", "",
                "| Configuration | Refused at the gate | Mean s when refused | Mean s when answered | Estimated seconds saved | Share of the run |",
                "|---|---|---|---|---|---|"]
        for c, g in rep["gate"].items():
            out.append(f"| {c} | {g['questions_refused_at_the_gate']} | {g['mean_seconds_when_refused']:.1f} | "
                       f"{g['mean_seconds_when_answered']:.1f} | {g['estimated_seconds_saved']:.0f} | {g['share_of_the_run']:.0%} |")
        out += ["", "The saving is an estimate: it assumes a refused question would have taken as long as an answered one."]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=RESULTS / "generation_by_question.csv")
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    if not a.csv.is_file():
        print(f"{a.csv} not found: run evaluation.evaluate first.")
        return 2
    rep = report(load_rows(a.csv))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "latency_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    md = markdown(rep)
    (a.out / "latency_report.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
