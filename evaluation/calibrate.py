"""Choose the abstention thresholds on the DEV set (never on the held-out test set).

The system answers only if the best cross-encoder score >= ``abstain_rerank_min`` AND the share of the question's key
terms found in the evidence >= ``abstain_coverage_min``. This script measures both numbers for every dev question
(answerable ones should pass, unanswerable ones should not) and picks the thresholds that make the most correct
decisions. Ties are broken towards fewer false abstentions, then towards the middle of the best region.

    python -m evaluation.calibrate GDPR.pdf ePrivacy.pdf DataAct.pdf            # print the result
    python -m evaluation.calibrate GDPR.pdf ePrivacy.pdf DataAct.pdf --write-env   # also write them to .env
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import ROOT, get_settings
from evaluation.dataset import DEV_SET, EVAL_DIR, load_dataset
from retrieval import hybrid
from verification import evidence_checker
from scripts.search_pdf import ensure_indexed
from services.paths import GlobPaths

ENV_KEYS = {"abstain_rerank_min": "LEXORA_ABSTAIN_RERANK_MIN", "abstain_coverage_min": "LEXORA_ABSTAIN_COVERAGE_MIN"}


def collect_features(case_id: str, questions) -> list[dict]:
    """For each dev question: the best rerank score and the term coverage, exactly as the gate computes them."""
    s, rows = get_settings(), []
    for q in questions:
        hits, _ = hybrid.retrieve(case_id, q.question)
        suff = evidence_checker.assess(q.question, hits, s)
        rows.append({"id": q.id, "answerable": q.answerable, "top_score": suff.top_score, "coverage": suff.coverage})
    return rows


def would_answer(row: dict, rerank_min: float, coverage_min: float) -> bool:
    """The gate's decision (see verification/evidence_checker.assess)."""
    return row["top_score"] is not None and row["top_score"] >= rerank_min and row["coverage"] >= coverage_min


def score(rows: list[dict], rerank_min: float, coverage_min: float) -> dict:
    ok = [would_answer(r, rerank_min, coverage_min) == r["answerable"] for r in rows]
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    return {"accuracy": sum(ok) / len(rows),
            "false_abstention_rate": sum(not would_answer(r, rerank_min, coverage_min) for r in ans) / len(ans),
            "correct_abstention_rate": sum(not would_answer(r, rerank_min, coverage_min) for r in unans) / len(unans)}


def _candidates(values: list[float], low: float, high: float) -> list[float]:
    """Thresholds worth trying: one between every two neighbouring observed values, plus both extremes."""
    v = sorted(set(values))
    return [low] + [(a + b) / 2 for a, b in zip(v, v[1:])] + [high]


def _tidy(x: float, rows: list[dict], other: float, axis: str) -> float:
    """Round to 2 decimals if that does not change a single decision; otherwise keep the exact value."""
    r = round(x, 2)
    f = (lambda t: score(rows, t, other)) if axis == "rerank" else (lambda t: score(rows, other, t))
    return r if f(r) == f(x) else x


def choose_thresholds(rows: list[dict]) -> dict:
    """The (rerank_min, coverage_min) pair with the most correct decisions on ``rows``."""
    if not any(r["answerable"] for r in rows) or all(r["answerable"] for r in rows):
        raise ValueError("Calibration needs both answerable and unanswerable questions.")
    tops = [r["top_score"] for r in rows if r["top_score"] is not None]
    r_grid = _candidates(tops, min(tops) - 1, max(tops) + 1) if tops else [0.0]
    c_grid = _candidates([r["coverage"] for r in rows], 0.0, 1.01)
    results = [(rm, cm, score(rows, rm, cm)) for rm in r_grid for cm in c_grid]
    best = max(m["accuracy"] for _, _, m in results)
    top = [(rm, cm, m) for rm, cm, m in results if m["accuracy"] == best]
    fa = min(m["false_abstention_rate"] for _, _, m in top)
    top = [t for t in top if t[2]["false_abstention_rate"] == fa]
    ca = max(m["correct_abstention_rate"] for _, _, m in top)
    top = [t for t in top if t[2]["correct_abstention_rate"] == ca]
    mr, mc = sum(t[0] for t in top) / len(top), sum(t[1] for t in top) / len(top)
    span_r, span_c = (max(t[0] for t in top) - min(t[0] for t in top)) or 1.0, (max(t[1] for t in top) - min(t[1] for t in top)) or 1.0
    rm, cm, m = min(top, key=lambda t: ((t[0] - mr) / span_r) ** 2 + ((t[1] - mc) / span_c) ** 2)
    rm, cm = _tidy(rm, rows, cm, "rerank"), _tidy(cm, rows, rm, "coverage")
    m = score(rows, rm, cm)
    return {"abstain_rerank_min": rm, "abstain_coverage_min": cm, "dev_accuracy": round(m["accuracy"], 4),
            "dev_false_abstention_rate": round(m["false_abstention_rate"], 4),
            "dev_correct_abstention_rate": round(m["correct_abstention_rate"], 4), "n_dev": len(rows),
            "n_answerable": sum(r["answerable"] for r in rows), "n_unanswerable": sum(not r["answerable"] for r in rows),
            "equally_good_pairs": len(top)}


def write_env(path: Path, updates: dict[str, float]) -> None:
    """Set KEY=value lines in a .env file, keeping every other line."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    for key, value in updates.items():
        new = f"{key}={value:g}"
        for i, line in enumerate(lines):
            if line.split("=", 1)[0].strip() == key:
                lines[i] = new
                break
        else:
            lines.append(new)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="+", action=GlobPaths)
    ap.add_argument("--dataset", type=Path, default=DEV_SET)
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "results" / "calibration.json")
    ap.add_argument("--write-env", nargs="?", const="", metavar=".env", help="write the thresholds to .env (default: project .env)")
    args = ap.parse_args(argv)
    questions = load_dataset(args.dataset).questions
    rows = collect_features(ensure_indexed(args.pdfs), questions)
    result = choose_thresholds(rows)
    s = get_settings()
    now = score(rows, s.abstain_rerank_min, s.abstain_coverage_min)
    result["current"] = {"abstain_rerank_min": s.abstain_rerank_min, "abstain_coverage_min": s.abstain_coverage_min,
                         "dev_accuracy": round(now["accuracy"], 4)}
    result["features"] = [{**r, "top_score": None if r["top_score"] is None else round(r["top_score"], 3),
                           "coverage": round(r["coverage"], 3)} for r in rows]
    result["note"] = (f"Chosen on {len(rows)} dev questions only: one question moves the accuracy by "
                      f"{100 / len(rows):.0f} points, so treat the thresholds as approximate.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{'id':<5}{'answerable':<12}{'top score':>10}{'coverage':>10}")
    for r in result["features"]:
        print(f"{r['id']:<5}{str(r['answerable']):<12}{str(r['top_score']):>10}{r['coverage']:>10}")
    print(f"\nChosen: abstain_rerank_min={result['abstain_rerank_min']:g}  abstain_coverage_min={result['abstain_coverage_min']:g}"
          f"  -> dev accuracy {result['dev_accuracy']:.0%} (false abstentions {result['dev_false_abstention_rate']:.0%}, "
          f"unanswerable refused {result['dev_correct_abstention_rate']:.0%}; {result['equally_good_pairs']} equally good pairs)")
    print(f"Current settings: {s.abstain_rerank_min:g} / {s.abstain_coverage_min:g} -> dev accuracy {result['current']['dev_accuracy']:.0%}")
    print(result["note"])
    if args.write_env is not None:
        env = Path(args.write_env) if args.write_env else ROOT / ".env"
        write_env(env, {ENV_KEYS[k]: result[k] for k in ENV_KEYS})
        print(f"Written to {env}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
