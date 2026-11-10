"""The verifier ablation (hybrid_noverify), per-citation detail, and citation precision at region level."""
import csv
import json
from pathlib import Path

import pytest

from api.routes import AskIn
from evaluation import evaluate as ev
from evaluation import region_eval as rg
from generation.answer_generator import AnswerService
from services import indexer
from tests.conftest import FakeOllama, fabricated_answer, grounded_answer
from tests.test_evaluate import QUESTIONS, case, env, questions, run  # noqa: F401  (fixtures and helpers)


def rows_for(env, configs, llm=None, **kw):
    return ev.run_generation(case(env), questions(), configs, AnswerService(llm or env["llm"]), ev.GenCache(None), **kw)


# ------------------------------------------------------------------------------ the verifier ablation
def test_without_the_verifier_a_fabricated_draft_is_shown_unchecked_and_with_it_the_draft_is_refused(env):
    liar = FakeOllama(lambda m: fabricated_answer(m))
    rows = rows_for(env, ["hybrid", "hybrid_noverify"], llm=liar)
    by = {(r["config"], r["id"]): r for r in rows}
    on, off = by["hybrid", "F1"], by["hybrid_noverify", "F1"]
    assert on["abstained"] and on["refusal_stage"] == "verifier" and on["retries"] == 1 and on["claims_shown"] == 0
    assert not off["abstained"] and off["retries"] == 0 and off["claims_shown"] == 1
    assert off["unsupported_shown"] == 1 and off["verified_cites"] == 0                   # the checks ran, nothing was done
    assert "999 million" in off["answer"]
    s = ev.summarize_generation(rows)
    assert s["hybrid_noverify"]["unsupported_claim_rate"] == 1.0 and s["hybrid_noverify"]["answers_with_unsupported_claim"] == 1.0
    assert s["hybrid"]["unsupported_claim_rate"] is None                                  # nothing shown, nothing unsupported
    assert s["hybrid"]["false_abstention_rate"] == 1.0 and s["hybrid_noverify"]["false_abstention_rate"] == 0.0


def test_the_ablation_asks_the_model_once_per_question_where_the_verifier_asks_twice(env):
    liar = FakeOllama(lambda m: fabricated_answer(m))
    rows_for(env, ["hybrid_noverify"], llm=liar)
    assert len(liar.calls) == 2                                                          # F1, F2 (U1 is stopped by the gate)
    liar2 = FakeOllama(lambda m: fabricated_answer(m))
    rows_for(env, ["hybrid"], llm=liar2)
    assert len(liar2.calls) == 4


def test_a_good_answer_is_the_same_with_and_without_the_verifier(env):
    rows = rows_for(env, ["hybrid", "hybrid_noverify"])
    by = {(r["config"], r["id"]): r for r in rows}
    assert by["hybrid", "F1"]["answer"] == by["hybrid_noverify", "F1"]["answer"]
    assert by["hybrid_noverify", "F1"]["unsupported_shown"] == 0


def test_the_api_and_the_service_default_to_verifying():
    assert AskIn(question="abc def").verify is True and AskIn(question="abc def", verify=False).verify is False


def test_the_ablation_is_not_part_of_the_hand_graded_sheet(env):
    assert run(env, "--gen-configs", "hybrid", "hybrid_noverify", "--export-grading") == 0
    sheet = list(csv.DictReader(open(env["out"] / "manual_grades.csv", encoding="utf-8")))
    assert {r["config"] for r in sheet} == {"hybrid"} and len(sheet) == 3
    summary = json.loads((env["out"] / "summary.json").read_text())
    assert set(summary["generation"]) == {"hybrid", "hybrid_noverify"}


# ------------------------------------------------------------------------------ citation detail
def test_every_row_stores_its_citations_but_the_csv_does_not(env):
    assert run(env, "--gen-configs", "hybrid") == 0
    cache = [json.loads(l)["row"] for l in (env["out"] / "gen_cache.jsonl").read_text().splitlines()]
    f1 = next(r for r in cache if r["id"] == "F1")
    c = f1["citations"][0]
    assert c["document"] == "Statute.pdf" and c["page"] == 2 and c["status"] == "VERIFIED" and len(c["quote"]) > 20
    header = open(env["out"] / "generation_by_question.csv", encoding="utf-8").readline()
    assert "citations" not in header and "unsupported_shown" in header and "claims_shown" in header


def test_the_document_name_comes_from_the_matched_passage_not_from_what_the_model_wrote(env):
    def sloppy(messages):
        out = json.loads(grounded_answer(messages))
        out["claims"][0]["evidence"][0]["document"] = "statute"                          # the model drops ".pdf"
        return json.dumps(out)
    rows = rows_for(env, ["hybrid"], llm=FakeOllama(sloppy))
    f1 = next(r for r in rows if r["id"] == "F1")
    assert f1["citations"][0]["document"] == "Statute.pdf"


def test_rows_cached_without_citation_detail_are_asked_again_only_on_request(env, tmp_path):
    cache_file = tmp_path / "cache.jsonl"
    ev.run_generation(case(env), questions(), ["hybrid"], AnswerService(env["llm"]), ev.GenCache(cache_file))
    old = [json.loads(l) for l in cache_file.read_text().splitlines()]
    for item in old:
        item["row"].pop("citations")                                                     # a cache from the previous version
    cache_file.write_text("\n".join(json.dumps(i) for i in old) + "\n", encoding="utf-8")
    calls = len(env["llm"].calls)
    rows = ev.run_generation(case(env), questions(), ["hybrid"], AnswerService(env["llm"]), ev.GenCache(cache_file))
    assert len(env["llm"].calls) == calls and all("citations" not in r for r in rows)
    rows = ev.run_generation(case(env), questions(), ["hybrid"], AnswerService(env["llm"]), ev.GenCache(cache_file),
                             refresh_citations=True)
    assert len(env["llm"].calls) == calls + 2 and all("citations" in r for r in rows)     # F1 and F2 (U1 has no retrieval answer to refresh... it is refused)


# ------------------------------------------------------------------------------ region-level precision
def pdfs_of(env):
    return {Path(p).name: Path(p) for p in env["pdfs"]}


def test_a_grounded_citation_is_a_region_hit(env):
    rows = rows_for(env, ["hybrid"])
    report = rg.region_report(indexer.case_chunks(case(env)), questions(), rows, pdfs_of(env))
    assert report["n_citations"] == 2 and report["rows_without_detail"] == 0
    assert report["located_rate"] == 1.0 and report["page_precision"] == 1.0 and report["region_precision"] == 1.0
    assert report["page_hits_outside_the_region"] == 0


def test_a_citation_on_the_right_page_but_in_another_clause_is_a_page_hit_not_a_region_hit(env):
    chunks = indexer.case_chunks(case(env))
    pdfs = pdfs_of(env)
    evidence = [q for q in questions() if q.id == "F1"][0].evidence_dicts()               # Statute.pdf, page 2, clause 3(2)
    gold = rg.gold_regions(chunks, evidence, pdfs.get)
    right = {"document": "Statute.pdf", "page": 2, "clause": "3(2)", "quote": "Serious infringements shall be subject to fines up to 20 000 000 EUR", "status": "VERIFIED"}
    other = {"document": "Statute.pdf", "page": 2, "clause": "2", "quote": "'processor' means a body which processes data on behalf of the controller", "status": "VERIFIED"}
    wrong_page = {"document": "Statute.pdf", "page": 1, "clause": "1", "quote": "This Regulation lays down rules relating to the protection of data", "status": "VERIFIED"}
    res = rg.score_citations([right, other, wrong_page], evidence, gold, pdfs.get)
    assert [(r["page_hit"], r["region_hit"]) for r in res] == [(True, True), (True, False), (False, False)]
    s = rg.summarize(res)
    assert s["page_precision"] == pytest.approx(2 / 3, abs=1e-3) and s["region_precision"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["page_hits_outside_the_region"] == 1


def test_unlocatable_quotes_and_unknown_documents_are_misses_not_errors(env):
    chunks = indexer.case_chunks(case(env))
    pdfs = pdfs_of(env)
    evidence = [q for q in questions() if q.id == "F1"][0].evidence_dicts()
    gold = rg.gold_regions(chunks, evidence, pdfs.get)
    res = rg.score_citations([
        {"document": "Statute.pdf", "page": 2, "clause": "3(2)", "quote": "a sentence that is nowhere in the document", "status": "UNVERIFIED"},
        {"document": "Nope.pdf", "page": 2, "clause": "3(2)", "quote": "anything", "status": "FABRICATED"},
        {"document": "Statute.pdf", "page": 99, "clause": "3(2)", "quote": "Fines", "status": "FABRICATED"}], evidence, gold, pdfs.get)
    assert [r["located"] for r in res] == [False, False, False] and not any(r["region_hit"] for r in res)
    assert rg.summarize(res)["region_precision_of_located"] is None
    assert rg.summarize([]) == {"n_citations": 0}


def test_rows_without_detail_are_counted_and_left_out(env):
    rows = rows_for(env, ["hybrid"])
    for r in rows:
        r.pop("citations")
    report = rg.region_report(indexer.case_chunks(case(env)), questions(), rows, pdfs_of(env))
    assert report["rows_without_detail"] == 2 and report["n_citations"] == 0


def test_the_report_and_the_summary_contain_the_region_section(env):
    assert run(env, "--gen-configs", "hybrid") == 0
    summary = json.loads((env["out"] / "summary.json").read_text())
    assert summary["region"]["config"] == "hybrid" and summary["region"]["region_precision"] == 1.0
    md = (env["out"] / "summary.md").read_text()
    assert "### Citation precision at region level" in md and "Region-level precision" in md
    assert "Claims shown that the validator does not verify" in md


def test_overlap_arithmetic():
    assert rg._area((0, 0, 10, 5)) == 50 and rg._area((5, 5, 0, 0)) == 0
    assert rg._intersection((0, 0, 10, 10), (5, 5, 20, 20)) == 25 and rg._intersection((0, 0, 1, 1), (2, 2, 3, 3)) == 0
    assert rg.union([(0, 0, 1, 1), (2, 2, 3, 4)]) == (0, 0, 3, 4) and rg.union([]) is None
