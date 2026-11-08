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
    assert set(summary) == {"meta", "retrieval", "by_category", "generation"} and set(summary["generation"]) == {"no_retrieval", "hybrid"}
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
