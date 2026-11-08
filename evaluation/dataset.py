"""The labelled question sets used for evaluation and calibration.

Two disjoint sets exist, on purpose:
  * dataset.json          the HELD-OUT test set: never used to build the index or to choose a threshold
  * calibration_set.json  a DEV set: used only to choose the abstention thresholds (evaluation/calibrate.py)

Every answerable question lists the evidence a correct answer needs (document, page, clause, a snippet that is
really on that page); an unanswerable question lists none. ``verify_snippets`` proves the labels against the PDFs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import pymupdf
from pydantic import BaseModel, Field, field_validator, model_validator

EVAL_DIR = Path(__file__).resolve().parent
TEST_SET = EVAL_DIR / "dataset.json"
DEV_SET = EVAL_DIR / "calibration_set.json"


class Evidence(BaseModel):
    document: str
    page: int = Field(ge=1)
    clause: str
    snippet: str = ""                       # text that really occurs on that page (checked by verify_snippets)


class EvalQuestion(BaseModel):
    id: str
    category: Literal["factual", "multihop", "unanswerable"]
    answerable: bool
    question: str = Field(min_length=3)
    expected_answer: str
    key_facts: list[str] = Field(default_factory=list)      # what a correct answer must contain (automatic proxy)
    expected_evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "EvalQuestion":
        if self.answerable != (self.category != "unanswerable"):
            raise ValueError(f"{self.id}: 'answerable' must be false exactly for the category 'unanswerable'")
        if bool(self.expected_evidence) != self.answerable:
            raise ValueError(f"{self.id}: answerable questions need expected_evidence, unanswerable ones must have none")
        if self.answerable and not self.key_facts:
            raise ValueError(f"{self.id}: an answerable question needs key_facts")
        if self.category == "multihop" and len(self.expected_evidence) < 2:
            raise ValueError(f"{self.id}: a multi-hop question needs at least two evidence items")
        return self

    def evidence_dicts(self) -> list[dict]:
        return [e.model_dump() for e in self.expected_evidence]


class Dataset(BaseModel):
    meta: dict = Field(default_factory=dict)
    questions: list[EvalQuestion]

    @field_validator("questions")
    @classmethod
    def _unique_ids(cls, qs: list[EvalQuestion]) -> list[EvalQuestion]:
        ids = [q.id for q in qs]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate question ids: {dupes}")
        return qs

    @property
    def answerable(self) -> list[EvalQuestion]:
        return [q for q in self.questions if q.answerable]


def load_dataset(path: str | Path = TEST_SET) -> Dataset:
    """Load and validate a question set (an object with "questions", or a bare list)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Dataset(**data) if isinstance(data, dict) else Dataset(questions=data)


def overlap(test: Dataset, dev: Dataset) -> list[str]:
    """Questions or ids that appear in both sets. Must be empty: the dev set must not leak into the test set."""
    t_text, d_text = {q.question.strip().lower() for q in test.questions}, {q.question.strip().lower() for q in dev.questions}
    t_ids, d_ids = {q.id for q in test.questions}, {q.id for q in dev.questions}
    return sorted(t_text & d_text) + sorted(t_ids & d_ids)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'").replace("‘", "'").replace(" ", " ")).lower()


def verify_snippets(dataset: Dataset, pdf_dir: str | Path) -> list[str]:
    """Check that every labelled snippet really occurs on its labelled page. Returns a list of problems."""
    problems, pages = [], {}
    for q in dataset.questions:
        for e in q.expected_evidence:
            pdf = Path(pdf_dir) / e.document
            if not pdf.is_file():
                problems.append(f"{q.id}: {e.document} not found in {pdf_dir}")
                continue
            if e.document not in pages:
                with pymupdf.open(pdf) as doc:
                    pages[e.document] = [_norm(p.get_text()) for p in doc]
            texts = pages[e.document]
            if not 1 <= e.page <= len(texts):
                problems.append(f"{q.id}: page {e.page} does not exist in {e.document} ({len(texts)} pages)")
            elif e.snippet and _norm(e.snippet) not in texts[e.page - 1]:
                problems.append(f"{q.id}: snippet {e.snippet!r} is not on page {e.page} of {e.document}")
    return problems
