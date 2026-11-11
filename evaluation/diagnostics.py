"""Post-hoc diagnostics of the abstention gate. NOT used to tune the system.

    python -m evaluation.diagnostics sample_docs/*.pdf                 # about 1 minute, no LLM

1. Sensitivity. For each rerank threshold: abstention accuracy, false-abstention rate and correct-abstention rate on the
   DEV set and on the HELD-OUT set (coverage threshold fixed at the shipped value). It shows where a false abstention on
   the held-out set comes from and what moving the threshold would cost. The shipped thresholds were chosen on the dev
   set only; choosing one because of what this table shows for the held-out set would be tuning on the test set.
2. Where the labelled evidence ranks. For each answerable question: the rank, and the cross-encoder score, of the passages
   that match the labelled evidence. A false abstention with the evidence at rank 1-5 is a SCORING problem (the
   cross-encoder gives a low score to a passage that is in fact relevant), not a retrieval problem.

Writes evaluation/results/diagnostics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import get_settings
from evaluation import metrics as M
from evaluation.calibrate import score
from evaluation.dataset import DEV_SET, EVAL_DIR, TEST_SET, load_dataset
from retrieval import hybrid
from retrieval.synonyms import expand_query
from scripts.search_pdf import ensure_indexed
from verification import evidence_checker
from services.paths import GlobPaths

THRESHOLDS = (-1.0, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)


def features_with_gold(case_id: str, questions) -> list[dict]:
    """The gate's two numbers for each question, plus where the labelled evidence ranks among the final passages."""
    s, rows = get_settings(), []
    for q in questions:
        hits, _ = hybrid.retrieve(case_id, q.question)
        suff = evidence_checker.assess(expand_query(q.question), hits, s)
        gold_ranks, gold_scores = [], []
        for e in q.evidence_dicts():
            pos = [i for i, h in enumerate(hits, 1) if M.chunk_matches(h.chunk, e)]
            gold_ranks.append(pos[0] if pos else None)
            gold_scores.extend(round(h.rerank_score, 2) for h in hits if M.chunk_matches(h.chunk, e) and h.rerank_score is not None)
        rows.append({"id": q.id, "answerable": q.answerable, "top_score": suff.top_score, "coverage": suff.coverage,
                     "gold_ranks": gold_ranks, "gold_rerank_scores": gold_scores})
    return rows


def sweep(rows: list[dict], coverage_min: float, thresholds=THRESHOLDS) -> list[dict]:
    out = []
    for t in thresholds:
        m = score(rows, t, coverage_min)
        out.append({"rerank_min": t, "accuracy": round(m["accuracy"], 3),
                    "false_abstention": round(m["false_abstention_rate"], 3), "correct_abstention": round(m["correct_abstention_rate"], 3)})
    return out


def false_abstentions(rows: list[dict], rerank_min: float, coverage_min: float) -> list[dict]:
    return [r for r in rows if r["answerable"] and (r["top_score"] is None or r["top_score"] < rerank_min or r["coverage"] < coverage_min)]


def markdown(result: dict) -> str:
    out = ["| rerank threshold | held-out: accuracy / false abstention / correct abstention | dev: accuracy / false abstention / correct abstention |",
           "|---|---|---|"]
    for h, d in zip(result["sweep_heldout"], result["sweep_dev"]):
        out.append(f"| {h['rerank_min']:g} | {h['accuracy']:.2f} / {h['false_abstention']:.2f} / {h['correct_abstention']:.2f} | "
                   f"{d['accuracy']:.2f} / {d['false_abstention']:.2f} / {d['correct_abstention']:.2f} |")
    s = result["shipped"]
    out += ["", f"Shipped thresholds: rerank >= {s['abstain_rerank_min']:g}, coverage >= {s['abstain_coverage_min']:g} (chosen on the dev set only).", ""]
    for r in result["false_abstentions_heldout"]:
        out.append(f"False abstention {r['id']}: best rerank score {r['top_score']:.2f}, coverage {r['coverage']:.2f}; labelled evidence at "
                   f"rank(s) {r['gold_ranks']} with cross-encoder score(s) {r['gold_rerank_scores']}.")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="+", action=GlobPaths)
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "results")
    a = ap.parse_args(argv)
    s = get_settings()
    case_id = ensure_indexed(a.pdfs)
    held = features_with_gold(case_id, load_dataset(TEST_SET).questions)
    dev = features_with_gold(case_id, load_dataset(DEV_SET).questions)
    result = {"note": "post-hoc analysis; the shipped thresholds were calibrated on the dev set only",
              "shipped": {"abstain_rerank_min": s.abstain_rerank_min, "abstain_coverage_min": s.abstain_coverage_min},
              "heldout_features": held, "dev_features": dev,
              "sweep_heldout": sweep(held, s.abstain_coverage_min), "sweep_dev": sweep(dev, s.abstain_coverage_min),
              "false_abstentions_heldout": false_abstentions(held, s.abstain_rerank_min, s.abstain_coverage_min)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "diagnostics.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    md = markdown(result)
    (a.out / "diagnostics.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
