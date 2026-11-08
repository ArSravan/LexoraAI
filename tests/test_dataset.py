import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from config import ROOT
from evaluation.dataset import DEV_SET, TEST_SET, Dataset, EvalQuestion, load_dataset, overlap, verify_snippets
from tests.conftest import make_pdf

EV = {"document": "A.pdf", "page": 1, "clause": "1", "snippet": "fines"}


def q(**kw):
    base = dict(id="F1", category="factual", answerable=True, question="What are the fines?", expected_answer="EUR 20 million",
                key_facts=["20 000 000"], expected_evidence=[EV])
    return {**base, **kw}


# ------------------------------------------------------------------------------ validation rules
def test_a_valid_question_loads():
    assert EvalQuestion(**q()).evidence_dicts() == [EV]


@pytest.mark.parametrize("bad", [
    q(answerable=False),                                             # factual but marked unanswerable
    q(category="unanswerable"),                                      # unanswerable but marked answerable
    q(expected_evidence=[]),                                         # answerable without evidence
    q(key_facts=[]),                                                 # answerable without key facts
    q(category="multihop"),                                          # multi-hop with a single evidence item
    q(category="unanswerable", answerable=False, key_facts=[]),      # unanswerable that still lists evidence
    q(category="trivia"), q(question="?"), q(expected_evidence=[{**EV, "page": 0}]),
])
def test_inconsistent_questions_are_rejected(bad):
    with pytest.raises(ValidationError):
        EvalQuestion(**bad)


def test_unanswerable_and_multihop_questions_are_accepted():
    EvalQuestion(**q(id="U1", category="unanswerable", answerable=False, key_facts=[], expected_evidence=[]))
    EvalQuestion(**q(id="M1", category="multihop", expected_evidence=[EV, {**EV, "clause": "2"}]))


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValidationError, match="duplicate question ids"):
        Dataset(questions=[q(), q(question="Another question?")])


def test_loading_accepts_an_object_or_a_bare_list(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"meta": {"n": 1}, "questions": [q()]}))
    (tmp_path / "b.json").write_text(json.dumps([q()]))
    assert load_dataset(tmp_path / "a.json").meta == {"n": 1} and len(load_dataset(tmp_path / "b.json").questions) == 1
    assert load_dataset(tmp_path / "a.json").answerable[0].id == "F1"


def test_overlap_between_test_and_dev_is_detected():
    a, b = Dataset(questions=[q()]), Dataset(questions=[q(id="D1"), q(id="D2", question="what are the FINES? ")])
    assert overlap(a, b) == ["what are the fines?"]
    assert overlap(a, Dataset(questions=[q(question="Something else entirely?")])) == ["F1"]
    assert overlap(a, Dataset(questions=[q(id="D1", question="Something else entirely?")])) == []


# ------------------------------------------------------------------------------ snippet proof against real PDFs
def test_verify_snippets_finds_wrong_pages_missing_files_and_bad_snippets(tmp_path):
    make_pdf(tmp_path / "A.pdf", [[(60, 700, "Page one has no sanctions.")], [(60, 700, "Infringements bring administrative fines up to 20 000 000 EUR.")]])
    good = Dataset(questions=[q(expected_evidence=[{**EV, "page": 2, "snippet": "ADMINISTRATIVE  fines up to 20 000 000 EUR"}])])
    assert verify_snippets(good, tmp_path) == []
    wrong_page = Dataset(questions=[q(expected_evidence=[{**EV, "page": 1, "snippet": "administrative fines"}])])
    assert "is not on page 1" in verify_snippets(wrong_page, tmp_path)[0]
    beyond = Dataset(questions=[q(expected_evidence=[{**EV, "page": 9}])])
    assert "page 9 does not exist" in verify_snippets(beyond, tmp_path)[0]
    missing = Dataset(questions=[q(expected_evidence=[{**EV, "document": "Nope.pdf"}])])
    assert "not found" in verify_snippets(missing, tmp_path)[0]


# ------------------------------------------------------------------------------ the real data files
@pytest.fixture(scope="module")
def test_set():
    return load_dataset(TEST_SET)


@pytest.fixture(scope="module")
def dev_set():
    return load_dataset(DEV_SET)


def test_the_held_out_set_meets_the_assignment(test_set):
    c = Counter(x.category for x in test_set.questions)
    assert len(test_set.questions) >= 15 and c["factual"] >= 6 and c["multihop"] >= 5 and c["unanswerable"] >= 3
    assert "Held-out" in test_set.meta["note"]


def test_the_dev_set_is_separate_from_the_held_out_set(test_set, dev_set):
    assert overlap(test_set, dev_set) == [] and len(dev_set.questions) >= 10
    assert "ONLY to calibrate" in dev_set.meta["note"] and Counter(x.category for x in dev_set.questions)["unanswerable"] >= 3


SAMPLE_DOCS = ROOT / "sample_docs"


@pytest.mark.skipif(not (SAMPLE_DOCS / "GDPR_Regulation_2016_679.pdf").is_file(),
                    reason="run `python -m scripts.download_sample_docs` to fetch the EU legislation")
@pytest.mark.parametrize("which", ["test", "dev"])
def test_every_label_is_really_on_its_page_in_the_real_documents(which, test_set, dev_set):
    assert verify_snippets(test_set if which == "test" else dev_set, SAMPLE_DOCS) == []
