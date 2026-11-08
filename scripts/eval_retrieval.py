"""Compare the retrieval configurations on a labelled question set (no LLM).

    python -m scripts.eval_retrieval GDPR.pdf ePrivacy.pdf DataAct.pdf --dataset evaluation/dataset.json

The dataset is JSON: {"questions": [{"id", "category", "answerable", "question",
"expected_evidence": [{"document": "GDPR.pdf", "clause": "33(1)"}]}]}. Only answerable questions are used.
Add --misses to list the questions whose evidence is not in the top 5.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from evaluation.metrics import aggregate, retrieval_scores
from retrieval.hybrid import retrieve
from scripts.search_pdf import CONFIGS, ensure_indexed

COLUMNS = ["recall@1", "recall@3", "recall@5", "full@5", "precision@5", "mrr"]


def evaluate(case_id: str, questions: list[dict], k: int = 5) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    summary, per_question = {}, {}
    for label, mode, rerank, decomp in CONFIGS:
        rows = []
        for q in questions:
            t = time.perf_counter()
            hits, _ = retrieve(case_id, q["question"], mode, final_k=k, rerank=rerank, decompose_query=decomp)
            scores = retrieval_scores([h.chunk for h in hits], q["expected_evidence"])
            rows.append({**scores, "latency_ms": (time.perf_counter() - t) * 1000, "id": q["id"]})
        summary[label] = aggregate(rows, COLUMNS + ["latency_ms"])
        per_question[label] = rows
    return summary, per_question


def format_table(summary: dict[str, dict], n: int) -> str:
    head = f"{'configuration':<40}" + "".join(f"{c:>12}" for c in COLUMNS) + f"{'latency ms':>12}"
    lines = [f"answerable questions: {n}", head, "-" * len(head)]
    for label, m in summary.items():
        lines.append(f"{label:<40}" + "".join(f"{m[c]:>12.2f}" for c in COLUMNS) + f"{m['latency_ms']:>12.0f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--misses", action="store_true", help="list questions missing evidence in the top 5")
    args = ap.parse_args(argv)
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    questions = [q for q in (data["questions"] if isinstance(data, dict) else data) if q.get("answerable", True)]
    case_id = ensure_indexed(args.pdfs)
    summary, per_question = evaluate(case_id, questions)
    print(format_table(summary, len(questions)))
    if args.misses:
        for label, rows in per_question.items():
            missed = [r["id"] for r in rows if r["recall@5"] < 1]
            print(f"missed at 5 - {label}: {missed or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
