"""Human grading of the generated answers.

The automatic key-fact check misses paraphrases and cannot judge a refusal, so the numbers that go into the report
should come from a person reading the answers. ``export_template`` writes a CSV with one row per answer and an
empty ``grade`` column; ``summarize_grades`` turns the filled-in file into metrics.

Scale (answerable question):    1 = correct and complete   0.5 = partly correct or incomplete   0 = wrong or ungrounded
Scale (unanswerable question):  1 = correctly says the documents do not contain it   0 = answers from outside knowledge
An unjustified refusal of an answerable question is a 0.
"""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

SCALE = ("answerable: 1 correct and complete, 0.5 partly correct, 0 wrong/ungrounded/unjustified refusal; "
         "unanswerable: 1 correctly says the documents do not contain it, 0 answers from outside knowledge")
FIELDS = ["id", "config", "category", "answerable", "question", "expected_answer", "answer", "abstained", "grade", "notes"]
VALID = {0.0, 0.5, 1.0}


def export_template(rows: list[dict], expected: dict[str, str], questions: dict[str, str], path: Path) -> int:
    """Write the grading sheet. Grades already in an existing file are kept, so re-exporting never loses work.
    ``expected`` and ``questions`` map a question id to its expected answer / text. Returns the number of rows."""
    old = {}
    if path.is_file():
        old = {(r["id"], r["config"]): (r["grade"], r["notes"]) for r in csv.DictReader(open(path, encoding="utf-8"))}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            grade, notes = old.get((r["id"], r["config"]), ("", ""))
            w.writerow({"id": r["id"], "config": r["config"], "category": r["category"], "answerable": r["answerable"],
                        "question": questions[r["id"]], "expected_answer": expected[r["id"]], "answer": r["answer"],
                        "abstained": r["abstained"], "grade": grade, "notes": notes})
    return len(rows)


def load_grades(path: Path) -> dict[tuple[str, str], float | None]:
    """{(question id, config): grade}. A blank grade is None (not graded yet). 0,5 is accepted for 0.5."""
    grades: dict[tuple[str, str], float | None] = {}
    problems = []
    for n, r in enumerate(csv.DictReader(open(path, encoding="utf-8")), start=2):
        raw = (r["grade"] or "").strip().replace(",", ".")
        if not raw:
            grades[r["id"], r["config"]] = None
            continue
        try:
            g = float(raw)
        except ValueError:
            g = -1.0
        if g not in VALID:
            problems.append(f"line {n} ({r['id']}, {r['config']}): grade {r['grade']!r} is not 0, 0.5 or 1")
        grades[r["id"], r["config"]] = g
    if problems:
        raise ValueError("Invalid grades:\n" + "\n".join(problems))
    return grades


def summarize_grades(rows: list[dict], grades: dict[tuple[str, str], float | None]) -> dict[str, dict]:
    """Per configuration: how many answers are graded, and the manual metrics over the graded ones."""
    out: dict[str, dict] = {}
    for config in dict.fromkeys(r["config"] for r in rows):
        mine = [r for r in rows if r["config"] == config]
        graded = [(r, grades[r["id"], r["config"]]) for r in mine if grades.get((r["id"], r["config"])) is not None]
        ans = [g for r, g in graded if r["answerable"]]
        unans = [g for r, g in graded if not r["answerable"]]
        out[config] = {
            "n_graded": len(graded), "n_total": len(mine),
            "answerable_mean_score": round(mean(ans), 4) if ans else None,
            "answerable_strictly_correct": round(mean(1.0 if g == 1 else 0.0 for g in ans), 4) if ans else None,
            "unanswerable_handled_correctly": round(mean(1.0 if g == 1 else 0.0 for g in unans), 4) if unans else None,
            "behaviour_correct_all": round(mean(1.0 if g == 1 else 0.0 for _, g in graded), 4) if graded else None,
        }
    return out
