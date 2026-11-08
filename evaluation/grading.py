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
from typing import NamedTuple

SCALE = ("answerable: 1 correct and complete, 0.5 partly correct, 0 wrong/ungrounded/unjustified refusal; "
         "unanswerable: 1 correctly says the documents do not contain it, 0 answers from outside knowledge")
FIELDS = ["id", "config", "category", "answerable", "question", "expected_answer", "answer", "abstained", "grade", "notes"]
VALID = {0.0, 0.5, 1.0}


class Exported(NamedTuple):
    rows: int           # answers in the sheet
    kept: int           # earlier grades carried over (the answer is unchanged)
    dropped: int        # earlier grades discarded because the answer changed since it was graded


class Grade(NamedTuple):
    value: float | None     # None = not graded yet
    answer: str             # the answer text this grade was given to


def export_template(rows: list[dict], expected: dict[str, str], questions: dict[str, str], path: Path) -> Exported:
    """Write the grading sheet. A grade already in an existing sheet is kept ONLY if the answer is still the same text:
    a grade belongs to the answer you read, so a new answer must be graded again.
    ``expected`` and ``questions`` map a question id to its expected answer / text."""
    old = {}
    if path.is_file():
        old = {(r["id"], r["config"]): (r["grade"], r["notes"], r["answer"]) for r in csv.DictReader(open(path, encoding="utf-8"))}
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            grade, notes, before = old.get((r["id"], r["config"]), ("", "", r["answer"]))
            if grade.strip() and before.strip() != r["answer"].strip():
                grade, notes, dropped = "", "", dropped + 1
            elif grade.strip():
                kept += 1
            w.writerow({"id": r["id"], "config": r["config"], "category": r["category"], "answerable": r["answerable"],
                        "question": questions[r["id"]], "expected_answer": expected[r["id"]], "answer": r["answer"],
                        "abstained": r["abstained"], "grade": grade, "notes": notes})
    return Exported(len(rows), kept, dropped)


def load_grades(path: Path) -> dict[tuple[str, str], Grade]:
    """{(question id, config): Grade}. A blank grade has value None (not graded yet). 0,5 is accepted for 0.5."""
    grades: dict[tuple[str, str], Grade] = {}
    problems = []
    for n, r in enumerate(csv.DictReader(open(path, encoding="utf-8")), start=2):
        raw = (r["grade"] or "").strip().replace(",", ".")
        if not raw:
            grades[r["id"], r["config"]] = Grade(None, r["answer"])
            continue
        try:
            g = float(raw)
        except ValueError:
            g = -1.0
        if g not in VALID:
            problems.append(f"line {n} ({r['id']}, {r['config']}): grade {r['grade']!r} is not 0, 0.5 or 1")
        grades[r["id"], r["config"]] = Grade(g, r["answer"])
    if problems:
        raise ValueError("Invalid grades:\n" + "\n".join(problems))
    return grades


def summarize_grades(rows: list[dict], grades: dict[tuple[str, str], Grade]) -> dict[str, dict]:
    """Per configuration: how many answers are graded, and the manual metrics over the graded ones. A grade whose
    answer text differs from the current answer is IGNORED (counted in ``n_stale``): it was given to another answer."""
    out: dict[str, dict] = {}
    for config in dict.fromkeys(r["config"] for r in rows):
        mine = [r for r in rows if r["config"] == config]
        given = [(r, grades.get((r["id"], r["config"]))) for r in mine]
        given = [(r, g) for r, g in given if g is not None and g.value is not None]
        stale = [r for r, g in given if g.answer.strip() != r["answer"].strip()]
        graded = [(r, g.value) for r, g in given if g.answer.strip() == r["answer"].strip()]
        ans = [g for r, g in graded if r["answerable"]]
        unans = [g for r, g in graded if not r["answerable"]]
        out[config] = {
            "n_graded": len(graded), "n_total": len(mine), "n_stale": len(stale),
            "answerable_mean_score": round(mean(ans), 4) if ans else None,
            "answerable_strictly_correct": round(mean(1.0 if g == 1 else 0.0 for g in ans), 4) if ans else None,
            "unanswerable_handled_correctly": round(mean(1.0 if g == 1 else 0.0 for g in unans), 4) if unans else None,
            "behaviour_correct_all": round(mean(1.0 if g == 1 else 0.0 for _, g in graded), 4) if graded else None,
        }
    return out
