import csv
import json

import pytest

from config import get_settings
from evaluation import evaluate as ev
from evaluation.dataset import Dataset
from generation.answer_generator import AnswerService
from tests.conftest import FakeOllama, grounded_answer

QUESTIONS = [
    {"id": "F1", "category": "factual", "answerable": True,
     "question": "What is the maximum fine for serious infringements, 4 % of annual turnover?",
     "expected_answer": "EUR 20 million or 4 %", "key_facts": ["20 000 000", "4 %"],
     "expected_evidence": [{"document": "Statute.pdf", "page": 2, "clause": "3(2)", "snippet": ""}]},
    {"id": "F2", "category": "factual", "answerable": True,
     "question": "What is the supplier liability cap in EUR under Section 7 of the agreement?",
     "expected_answer": "EUR 10,000,000", "key_facts": ["10,000,000"],
     "expected_evidence": [{"document": "Vendor_Agreement.pdf", "page": 1, "clause": "7.1", "snippet": ""}]},
    {"id": "U1", "category": "unanswerable", "answerable": False, "question": "Who won the football match yesterday?",
     "expected_answer": "INSUFFICIENT EVIDENCE", "key_facts": [], "expected_evidence": []},
]


def script(messages):
    """Grounded when it is given evidence, a plain general answer when it is not (the no-retrieval baseline)."""
    if any("EVIDENCE PASSAGES" in m["content"] for m in messages):
        return grounded_answer(messages)
    return "A general answer from memory."


@pytest.fixture
def env(fake_embedder, fake_reranker, eu_style_pdf, contract_pdf, tmp_path, monkeypatch):
    for k, v in {"ABSTAIN_RERANK_MIN": "0.5", "MEDIUM_CONF_RERANK": "0.5", "HIGH_CONF_RERANK": "3.0"}.items():
        monkeypatch.setenv(f"LEXORA_{k}", v)
    get_settings.cache_clear()
    dataset = tmp_path / "ds.json"
    dataset.write_text(json.dumps({"meta": {"note": "test"}, "questions": QUESTIONS}))
    contract = contract_pdf.with_name("Vendor_Agreement.pdf")
    if not contract.exists():
        contract.write_bytes(contract_pdf.read_bytes())
    eu = eu_style_pdf.with_name("Statute.pdf")
    if not eu.exists():
        eu.write_bytes(eu_style_pdf.read_bytes())
    return {"pdfs": [str(eu), str(contract)], "dataset": str(dataset), "out": tmp_path / "results", "llm": FakeOllama(script)}


def questions():
    return Dataset(questions=QUESTIONS).questions


def run(env, *extra, llm=None):
    service = AnswerService(llm or env["llm"])
    return ev.main([*env["pdfs"], "--dataset", env["dataset"], "--out", str(env["out"]), *extra], service=service)


def case(env):
    from scripts.search_pdf import ensure_indexed
    from pathlib import Path
    return ensure_indexed([Path(p) for p in env["pdfs"]])


# ------------------------------------------------------------------------------ rows and summaries
def test_rows_are_measured_against_the_labels(env):
    rows = ev.run_generation(case(env), questions(), ["hybrid", "no_retrieval"], AnswerService(env["llm"]), ev.GenCache(None))
    by = {(r["config"], r["id"]): r for r in rows}
    f1 = by["hybrid", "F1"]
    assert not f1["abstained"] and f1["correct_proxy"] is True and f1["verified_cites"] == 1 == f1["total_cites"] and f1["gold_cites"] == 1
    assert f1["confidence"] in ("HIGH", "MEDIUM") and f1["prompt_tokens"] == 100 and f1["retries"] == 0 and f1["error"] == ""
    assert by["hybrid", "U1"]["abstained"] is True and by["hybrid", "U1"]["correct_proxy"] is None
    base = by["no_retrieval", "F1"]
    assert base["abstained"] is False and base["correct_proxy"] is False and base["total_cites"] == 0 and base["answer"] == "A general answer from memory."
    assert len(rows) == 6


def test_the_summary_compares_retrieval_with_no_retrieval(env):
    rows = ev.run_generation(case(env), questions(), ["hybrid", "no_retrieval"], AnswerService(env["llm"]), ev.GenCache(None))
    s = ev.summarize_generation(rows)
    assert s["hybrid"]["answer_correct_proxy"] == 1.0 and s["hybrid"]["citations_verified_rate"] == 1.0
    assert s["hybrid"]["abstention_accuracy"] == 1.0 and s["hybrid"]["false_abstention_rate"] == 0.0 and s["hybrid"]["correct_abstention_rate"] == 1.0
    assert s["hybrid"]["answer_correct_proxy_by_category"] == {"factual": 1.0}
    assert s["no_retrieval"]["answer_correct_proxy"] == 0.0 and s["no_retrieval"]["citations_verified_rate"] is None
    assert s["no_retrieval"]["abstention_accuracy"] == pytest.approx(0.6667) and s["no_retrieval"]["correct_abstention_rate"] == 0.0
    assert s["hybrid"]["n"] == 3 and s["hybrid"]["n_answerable"] == 2 and s["hybrid"]["n_errors"] == 0


def test_errors_are_counted_and_do_not_count_as_correct(env):
    from generation.ollama_client import OllamaError
    llm = FakeOllama(lambda m: OllamaError("Ollama is not reachable."))
    rows = ev.run_generation(case(env), questions(), ["hybrid"], AnswerService(llm), ev.GenCache(None))
    s = ev.summarize_generation(rows)["hybrid"]
    assert s["n_errors"] == 2 and s["answer_correct_proxy"] == 0.0 and rows[0]["error"].startswith("Ollama is not reachable")


# ------------------------------------------------------------------------------ the cache
def test_answers_are_cached_and_a_second_run_asks_the_model_nothing(env, tmp_path):
    cache_file = tmp_path / "cache.jsonl"
    first = ev.run_generation(case(env), questions(), ["hybrid"], AnswerService(env["llm"]), ev.GenCache(cache_file))
    calls = len(env["llm"].calls)
    assert calls == 2 and len(cache_file.read_text().splitlines()) == 3            # U1 never reaches the model
    second = ev.run_generation(case(env), questions(), ["hybrid"], AnswerService(env["llm"]), ev.GenCache(cache_file))
    assert len(env["llm"].calls) == calls and second == first
    ev.run_generation(case(env), questions(), ["hybrid"], AnswerService(env["llm"]), ev.GenCache(cache_file), fresh=True)
    assert len(env["llm"].calls) == calls + 2


def test_a_changed_setting_invalidates_the_cache(env, tmp_path, monkeypatch):
    cache_file = tmp_path / "cache.jsonl"
    ev.run_generation(case(env), questions(), ["hybrid"], AnswerService(env["llm"]), ev.GenCache(cache_file))
    calls = len(env["llm"].calls)
    monkeypatch.setenv("LEXORA_FINAL_TOP_K", "3")
    get_settings.cache_clear()
    ev.run_generation(case(env), questions(), ["hybrid"], AnswerService(env["llm"]), ev.GenCache(cache_file))
    assert len(env["llm"].calls) == calls + 2


def test_an_interrupted_run_resumes_where_it_stopped(env, tmp_path):
    cache_file = tmp_path / "cache.jsonl"
    state = {"n": 0}

    def crashing(messages):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("laptop went to sleep")
        return script(messages)
    with pytest.raises(RuntimeError, match="went to sleep"):
        ev.run_generation(case(env), questions(), ["hybrid"], AnswerService(FakeOllama(crashing)), ev.GenCache(cache_file))
    assert len(cache_file.read_text().splitlines()) == 1                             # F1 was saved before the crash
    resumed = FakeOllama(script)
    rows = ev.run_generation(case(env), questions(), ["hybrid"], AnswerService(resumed), ev.GenCache(cache_file))
    assert len(resumed.calls) == 1 and [r["id"] for r in rows] == ["F1", "F2", "U1"]    # only F2 had to be asked again


# ------------------------------------------------------------------------------ the command line and its files
def test_the_command_writes_csv_json_and_a_readme_ready_report(env, capsys):
    assert run(env, "--gen-configs", "no_retrieval", "hybrid") == 0
    out = env["out"]
    assert {p.name for p in out.iterdir()} == {"retrieval_by_question.csv", "generation_by_question.csv", "summary.json",
                                              "summary.md", "meta.json", "gen_cache.jsonl"}
    gen = list(csv.DictReader(open(out / "generation_by_question.csv", encoding="utf-8")))
    assert len(gen) == 6 and {"id", "config", "abstained", "correct_proxy", "total_s", "answer"} <= set(gen[0])
    ret = list(csv.DictReader(open(out / "retrieval_by_question.csv", encoding="utf-8")))
    assert len(ret) == 5 * 2 and {r["id"] for r in ret} == {"F1", "F2"}              # 5 configurations, answerable questions only
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary) == {"meta", "retrieval", "by_category", "generation", "region"} and set(summary["generation"]) == {"no_retrieval", "hybrid"}
    assert summary["meta"]["n_questions"] == 3 and summary["meta"]["n_answerable"] == 2 and len(summary["meta"]["documents"][0]["sha256"]) == 64
    assert summary["meta"]["seed"] == 42 and summary["meta"]["ollama_model"] == "qwen3.5:9b"
    report = capsys.readouterr().out
    assert "### Retrieval" in report and "### Generation" in report and "| BM25 only |" in report and "AUTOMATIC proxy" in report
    assert report.strip() == (out / "summary.md").read_text().strip()


def test_a_retrieval_only_run_keeps_the_earlier_answer_results(env):
    run(env, "--gen-configs", "hybrid")
    before = json.loads((env["out"] / "summary.json").read_text())["generation"]
    calls = len(env["llm"].calls)
    run(env, "--retrieval-only")
    after = json.loads((env["out"] / "summary.json").read_text())["generation"]
    assert after == before and before and len(env["llm"].calls) == calls


def test_limit_and_a_repeat_run_use_the_cache(env):
    run(env, "--gen-configs", "hybrid", "--limit", "1")
    summary = json.loads((env["out"] / "summary.json").read_text())
    assert summary["meta"]["n_questions"] == 1 and summary["generation"]["hybrid"]["n"] == 1
    calls = len(env["llm"].calls)
    run(env, "--gen-configs", "hybrid", "--limit", "1")
    assert len(env["llm"].calls) == calls


def test_an_invalid_dataset_stops_the_run_before_anything_is_computed(env, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"questions": [{**QUESTIONS[0], "key_facts": []}]}))
    with pytest.raises(Exception, match="key_facts"):
        ev.main([*env["pdfs"], "--dataset", str(bad), "--out", str(env["out"]), "--retrieval-only"])
    assert not env["out"].exists()


# ------------------------------------------------------------------------------ human grading
from evaluation import grading  # noqa: E402

ROWS = [{"id": "F1", "config": "hybrid", "category": "factual", "answerable": True, "answer": "A", "abstained": False},
        {"id": "F2", "config": "hybrid", "category": "factual", "answerable": True, "answer": "B", "abstained": False},
        {"id": "U1", "config": "hybrid", "category": "unanswerable", "answerable": False, "answer": "INSUFFICIENT", "abstained": True},
        {"id": "F1", "config": "no_retrieval", "category": "factual", "answerable": True, "answer": "C", "abstained": False},
        {"id": "U1", "config": "no_retrieval", "category": "unanswerable", "answerable": False, "answer": "D", "abstained": False}]
EXPECTED = {"F1": "e1", "F2": "e2", "U1": "e3"}
TEXT = {"F1": "q1?", "F2": "q2?", "U1": "q3?"}


def test_the_grading_sheet_has_one_row_per_answer_and_never_loses_grades(tmp_path):
    sheet = tmp_path / "g.csv"
    assert grading.export_template(ROWS, EXPECTED, TEXT, sheet) == grading.Exported(rows=5, kept=0, dropped=0)
    rows = list(csv.DictReader(open(sheet, encoding="utf-8")))
    assert [r["grade"] for r in rows] == [""] * 5 and rows[0]["question"] == "q1?" and rows[0]["expected_answer"] == "e1"
    rows[1]["grade"], rows[1]["notes"] = "0.5", "missed the second tier"
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=grading.FIELDS)
        w.writeheader()
        w.writerows(rows)
    assert grading.export_template(ROWS, EXPECTED, TEXT, sheet) == grading.Exported(5, 1, 0)   # exporting again keeps what was graded
    again = list(csv.DictReader(open(sheet, encoding="utf-8")))
    assert again[1]["grade"] == "0.5" and again[1]["notes"] == "missed the second tier" and again[0]["grade"] == ""


def write_grades(path, values):
    grading.export_template(ROWS, EXPECTED, TEXT, path)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r, v in zip(rows, values):
        r["grade"] = v
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=grading.FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_grades_are_parsed_and_invalid_ones_are_reported_with_their_line(tmp_path):
    sheet = tmp_path / "g.csv"
    write_grades(sheet, ["1", "0,5", "", "0", "1"])
    g = {k: v.value for k, v in grading.load_grades(sheet).items()}
    assert g == {("F1", "hybrid"): 1.0, ("F2", "hybrid"): 0.5, ("U1", "hybrid"): None, ("F1", "no_retrieval"): 0.0, ("U1", "no_retrieval"): 1.0}
    write_grades(sheet, ["1", "good", "2", "0", "1"])
    with pytest.raises(ValueError) as e:
        grading.load_grades(sheet)
    assert "line 3 (F2, hybrid)" in str(e.value) and "line 4 (U1, hybrid)" in str(e.value) and "line 2" not in str(e.value)


def test_manual_metrics(tmp_path):
    sheet = tmp_path / "g.csv"
    write_grades(sheet, ["1", "0.5", "1", "0", "0"])
    s = grading.summarize_grades(ROWS, grading.load_grades(sheet))
    assert s["hybrid"] == {"n_graded": 3, "n_total": 3, "n_stale": 0, "answerable_mean_score": 0.75, "answerable_strictly_correct": 0.5,
                           "unanswerable_handled_correctly": 1.0, "behaviour_correct_all": 0.6667}
    assert s["no_retrieval"]["answerable_mean_score"] == 0.0 and s["no_retrieval"]["unanswerable_handled_correctly"] == 0.0
    write_grades(sheet, ["1", "", "", "", ""])
    part = grading.summarize_grades(ROWS, grading.load_grades(sheet))
    assert part["hybrid"]["n_graded"] == 1 and part["hybrid"]["unanswerable_handled_correctly"] is None and part["no_retrieval"]["n_graded"] == 0
    assert part["no_retrieval"]["behaviour_correct_all"] is None


def test_the_command_exports_a_sheet_and_later_adds_the_graders_numbers_without_asking_the_model_again(env):
    run(env, "--gen-configs", "no_retrieval", "hybrid", "--export-grading")
    sheet = env["out"] / "manual_grades.csv"
    rows = list(csv.DictReader(open(sheet, encoding="utf-8")))
    assert len(rows) == 6 and all(r["grade"] == "" for r in rows)
    for r in rows:                                                       # the "human" grades the answers
        r["grade"] = "1" if r["config"] == "hybrid" else "0"
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=grading.FIELDS)
        w.writeheader()
        w.writerows(rows)
    calls = len(env["llm"].calls)
    run(env, "--gen-configs", "no_retrieval", "hybrid", "--apply-grades", str(sheet), "--grader", "A. Student")
    assert len(env["llm"].calls) == calls                                # everything came from the cache
    manual = json.loads((env["out"] / "summary.json").read_text())["manual"]
    assert manual["grader"] == "A. Student" and manual["configs"]["hybrid"]["answerable_strictly_correct"] == 1.0
    assert manual["configs"]["no_retrieval"]["behaviour_correct_all"] == 0.0
    report = (env["out"] / "summary.md").read_text()
    assert "### Manual grades" in report and "Graded by: **A. Student**" in report and "INCOMPLETE" not in report


def test_an_incomplete_grading_sheet_is_flagged_in_the_report(env):
    run(env, "--gen-configs", "hybrid", "--export-grading")
    sheet = env["out"] / "manual_grades.csv"
    rows = list(csv.DictReader(open(sheet, encoding="utf-8")))
    rows[0]["grade"] = "1"
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=grading.FIELDS)
        w.writeheader()
        w.writerows(rows)
    run(env, "--gen-configs", "hybrid", "--apply-grades", str(sheet))
    report = (env["out"] / "summary.md").read_text()
    assert "1 / 3" in report and "Grading is INCOMPLETE" in report and "Graded by: **unspecified**" in report


# ------------------------------------------------------------------------------ a grade belongs to the answer it was given to
def test_a_grade_is_dropped_when_the_answer_changed_and_kept_when_it_did_not(tmp_path):
    sheet = tmp_path / "g.csv"
    write_grades(sheet, ["1", "0.5", "1", "0", "1"])
    new_rows = [dict(r) for r in ROWS]
    new_rows[1]["answer"] = "A DIFFERENT ANSWER FOR F2"                      # the model answered F2 differently this time
    assert grading.export_template(new_rows, EXPECTED, TEXT, sheet) == grading.Exported(rows=5, kept=4, dropped=1)
    after = {(r["id"], r["config"]): r for r in csv.DictReader(open(sheet, encoding="utf-8"))}
    assert after["F2", "hybrid"]["grade"] == "" and after["F2", "hybrid"]["notes"] == "" and after["F1", "hybrid"]["grade"] == "1"


def test_a_stale_grade_is_ignored_and_counted_not_averaged_in(tmp_path):
    sheet = tmp_path / "g.csv"
    write_grades(sheet, ["1", "0.5", "1", "0", "1"])
    changed = [dict(r) for r in ROWS]
    changed[0]["answer"] = "the answer changed after grading"
    s = grading.summarize_grades(changed, grading.load_grades(sheet))["hybrid"]
    assert s["n_stale"] == 1 and s["n_graded"] == 2 and s["answerable_mean_score"] == 0.5     # only F2 (0.5) still counts


def test_an_old_sheet_without_changes_behaves_exactly_as_before(tmp_path):
    sheet = tmp_path / "g.csv"
    write_grades(sheet, ["1", "0.5", "1", "0", "1"])
    s = grading.summarize_grades(ROWS, grading.load_grades(sheet))
    assert all(c["n_stale"] == 0 and c["n_graded"] == c["n_total"] for c in s.values())


# ------------------------------------------------------------------------------ failures are not answers; refusals explain themselves
def test_a_failed_answer_is_never_cached_and_a_cached_failure_is_asked_again(env, tmp_path):
    from generation.ollama_client import OllamaError
    cache_file = tmp_path / "cache.jsonl"
    down = FakeOllama(lambda m: OllamaError("Ollama is not reachable."))
    rows = ev.run_generation(case(env), questions(), ["hybrid"], AnswerService(down), ev.GenCache(cache_file))
    assert [r["error"] != "" for r in rows] == [True, True, False]           # F1, F2 failed; U1 was refused by the gate
    assert [r["refusal_stage"] for r in rows] == ["error", "error", "gate"]
    assert len(cache_file.read_text().splitlines()) == 1                      # only the genuine (gate) refusal was saved
    up = FakeOllama(script)
    again = ev.run_generation(case(env), questions(), ["hybrid"], AnswerService(up), ev.GenCache(cache_file))
    assert len(up.calls) == 2 and all(r["error"] == "" for r in again)       # F1 and F2 are asked again, U1 is cached
    # a failure written to the cache by an OLDER version of the program is ignored as well
    key = ev.GenCache.key(case(env), "hybrid", questions()[0].question)
    poisoned = tmp_path / "poisoned.jsonl"
    poisoned.write_text(json.dumps({"key": key, "row": {**again[0], "error": "old failure"}}) + "\n")
    fresh_llm = FakeOllama(script)
    ev.run_generation(case(env), questions()[:1], ["hybrid"], AnswerService(fresh_llm), ev.GenCache(poisoned))
    assert len(fresh_llm.calls) == 1


def test_retrieval_answers_are_tied_to_the_index_but_the_baseline_is_not(env):
    from services import indexer
    c = case(env)
    q = questions()[0].question
    hybrid_key, base_key = ev.GenCache.key(c, "hybrid", q), ev.GenCache.key(c, "no_retrieval", q)
    assert ev.GenCache.key(c, "hybrid", q) == hybrid_key
    for f in (get_settings().storage_dir / "cases" / c / "docs").glob("*/meta.json"):
        f.unlink()                                                            # the index changes (here: becomes stale)
    assert ev.GenCache.key(c, "hybrid", q) != hybrid_key                      # old retrieval answers can no longer be reused
    assert ev.GenCache.key(c, "no_retrieval", q) == base_key                  # the baseline never looked at the index


def test_every_row_says_who_refused_and_why(env):
    from tests.conftest import fabricated_answer

    def declined(messages):
        return json.dumps({"answer": "INSUFFICIENT EVIDENCE", "claims": [], "abstained": True})

    def first_question(script_fn):
        return ev.run_generation(case(env), questions()[:1], ["hybrid"], AnswerService(FakeOllama(script_fn)), ev.GenCache(None))[0]

    answered, by_verifier, by_model = first_question(script), first_question(fabricated_answer), first_question(declined)
    assert (answered["refusal_stage"], by_verifier["refusal_stage"], by_model["refusal_stage"]) == ("", "verifier", "model")
    assert answered["abstained"] is False and answered["refusal_reason"] == ""
    assert "none of its citations passed verification" in by_verifier["refusal_reason"]
    assert "not part of this case" in by_verifier["verifier_note"] and by_verifier["rejected_draft"] == "The cap is EUR 999 million."
    assert "do not answer the question" in by_model["refusal_reason"]
    gate = ev.run_generation(case(env), questions()[2:], ["hybrid"], AnswerService(FakeOllama(script)), ev.GenCache(None))[0]
    assert gate["refusal_stage"] == "gate" and "football" in gate["answer"]


def test_the_summary_and_report_flag_answerable_questions_refused_after_the_model_answered(env):
    from tests.conftest import fabricated_answer
    rows = ev.run_generation(case(env), questions(), ["hybrid"], AnswerService(FakeOllama(fabricated_answer)), ev.GenCache(None))
    s = ev.summarize_generation(rows)
    assert s["hybrid"]["answerable_refusals_by_stage"] == {"gate": 0, "model": 0, "verifier": 2, "error": 0}
    meta = {"date": "d", "ollama_model": "m", "seed": 42, "n_questions": 3, "n_answerable": 2}
    report = ev.markdown_report(meta, {}, {}, s)
    assert "**WARNING (hybrid):** 2 answerable question(s) were refused AFTER the model answered" in report
    assert "verifier_note" in report
    healthy = ev.run_generation(case(env), questions(), ["hybrid"], AnswerService(FakeOllama(script)), ev.GenCache(None))
    assert "WARNING" not in ev.markdown_report(meta, {}, {}, ev.summarize_generation(healthy))


def test_the_command_repairs_a_stale_index_instead_of_refusing_every_question(env, capsys):
    from services import indexer
    run(env, "--retrieval-only")
    c = case(env)
    for f in (get_settings().storage_dir / "cases" / c / "docs").glob("*/meta.json"):
        f.unlink()                                                            # an index from before the metadata existed
    assert not indexer.index_is_current(c)
    llm = FakeOllama(script)
    assert run(env, "--gen-configs", "hybrid", llm=llm) == 0
    assert indexer.index_is_current(c) and "out of date: rebuilding" in capsys.readouterr().err
    summary = json.loads((env["out"] / "summary.json").read_text())["generation"]["hybrid"]
    assert summary["answerable_refusals_by_stage"] == {"gate": 0, "model": 0, "verifier": 0, "error": 0}
    assert summary["answer_correct_proxy"] == 1.0 and summary["citations_verified_rate"] == 1.0
