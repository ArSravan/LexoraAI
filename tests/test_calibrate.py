import json
from pathlib import Path

import pytest

from config import Settings, get_settings
from evaluation import calibrate as cal
from schemas import RetrievedChunk
from verification.evidence_checker import assess
from tests.conftest import make_chunk


def row(i, answerable, top, cov):
    return {"id": i, "answerable": answerable, "top_score": top, "coverage": cov}


SEPARABLE = [row("a1", True, 7.0, 1.0), row("a2", True, 5.0, 0.8), row("a3", True, 4.0, 0.6),
             row("u1", False, -3.0, 0.2), row("u2", False, 1.0, 0.9), row("u3", False, -8.0, 0.0)]


def test_separable_data_is_classified_perfectly_with_thresholds_between_the_groups():
    r = cal.choose_thresholds(SEPARABLE)
    assert r["dev_accuracy"] == 1.0 and r["dev_false_abstention_rate"] == 0.0 and r["dev_correct_abstention_rate"] == 1.0
    assert 1.0 < r["abstain_rerank_min"] <= 4.0                     # above the best unanswerable score, below the weakest answerable
    assert (r["n_dev"], r["n_answerable"], r["n_unanswerable"]) == (6, 3, 3)
    assert all(cal.would_answer(x, r["abstain_rerank_min"], r["abstain_coverage_min"]) == x["answerable"] for x in SEPARABLE)


def test_overlapping_data_gets_the_best_compromise_and_prefers_fewer_false_abstentions():
    rows = [row("a1", True, 6.0, 1.0), row("a2", True, 3.0, 1.0), row("a3", True, 0.5, 1.0),       # a3 looks irrelevant
            row("u1", False, 2.0, 1.0), row("u2", False, -4.0, 0.1), row("u3", False, -6.0, 0.0)]
    r = cal.choose_thresholds(rows)
    # two choices reach 5/6: refuse a3 (one false abstention) or wrongly answer u1. The tie-break takes the second.
    assert r["dev_accuracy"] == pytest.approx(5 / 6, abs=1e-4)
    assert r["dev_false_abstention_rate"] == 0.0 and r["dev_correct_abstention_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert all(cal.would_answer(x, r["abstain_rerank_min"], r["abstain_coverage_min"]) for x in rows[:3])
    assert r["equally_good_pairs"] >= 1


def test_a_missing_score_never_answers():
    assert cal.would_answer(row("x", True, None, 1.0), -100, 0) is False
    r = cal.choose_thresholds(SEPARABLE + [row("n1", False, None, 0.0)])
    assert r["dev_accuracy"] == 1.0


def test_thresholds_are_rounded_only_when_that_changes_no_decision():
    r = cal.choose_thresholds(SEPARABLE)
    assert r["abstain_rerank_min"] == round(r["abstain_rerank_min"], 2) and r["abstain_coverage_min"] == round(r["abstain_coverage_min"], 2)
    tight = [row("a", True, 2.501, 1.0), row("u", False, 2.499, 0.0)]
    t = cal.choose_thresholds(tight)
    assert cal.would_answer(tight[0], t["abstain_rerank_min"], t["abstain_coverage_min"]) and not cal.would_answer(tight[1], t["abstain_rerank_min"], t["abstain_coverage_min"])


def test_both_kinds_of_question_are_required():
    with pytest.raises(ValueError, match="both answerable and unanswerable"):
        cal.choose_thresholds([row("a", True, 5, 1)])
    with pytest.raises(ValueError):
        cal.choose_thresholds([row("u", False, 5, 1)])


def test_the_calibration_decision_is_the_same_as_the_real_gate():
    """would_answer() must agree with evidence_checker.assess() or the calibration would tune the wrong thing."""
    s = Settings(abstain_rerank_min=2.0, abstain_coverage_min=0.5)
    for top, text in [(3.0, "administrative fines for infringements"), (1.0, "administrative fines for infringements"),
                      (3.0, "unrelated words only"), (-1.0, "unrelated words only")]:
        hits = [RetrievedChunk(chunk=make_chunk("c", text), rerank_score=top)]
        suff = assess("administrative fines infringements", hits, s)
        assert cal.would_answer(row("q", True, suff.top_score, suff.coverage), 2.0, 0.5) == suff.sufficient


def test_write_env_updates_or_appends_and_keeps_other_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text("LEXORA_SEED=7\nLEXORA_ABSTAIN_RERANK_MIN=9\n# a comment\n")
    cal.write_env(env, {"LEXORA_ABSTAIN_RERANK_MIN": 2.5, "LEXORA_ABSTAIN_COVERAGE_MIN": 0.32})
    assert env.read_text().splitlines() == ["LEXORA_SEED=7", "LEXORA_ABSTAIN_RERANK_MIN=2.5", "# a comment", "LEXORA_ABSTAIN_COVERAGE_MIN=0.32"]
    cal.write_env(tmp_path / "new.env", {"LEXORA_ABSTAIN_RERANK_MIN": 3})
    assert (tmp_path / "new.env").read_text() == "LEXORA_ABSTAIN_RERANK_MIN=3\n"


def test_the_command_calibrates_on_the_dev_questions_and_can_write_the_env_file(fake_embedder, fake_reranker, eu_style_pdf, contract_pdf, tmp_path, monkeypatch, capsys):
    eu, ct = eu_style_pdf.with_name("Statute.pdf"), contract_pdf.with_name("Vendor_Agreement.pdf")
    eu.write_bytes(eu_style_pdf.read_bytes())
    ct.write_bytes(contract_pdf.read_bytes())
    dev = tmp_path / "dev.json"
    mk = lambda i, cat, q, doc="", clause="": {"id": i, "category": cat, "answerable": cat != "unanswerable", "question": q,   # noqa: E731
        "expected_answer": "x", "key_facts": ["x"] if cat != "unanswerable" else [],
        "expected_evidence": [{"document": doc, "page": 1, "clause": clause}] if cat != "unanswerable" else []}
    dev.write_text(json.dumps({"questions": [
        mk("D1", "factual", "What is the maximum fine for serious infringements, 4 % of annual turnover?", "Statute.pdf", "3(2)"),
        mk("D2", "factual", "What is the supplier liability cap in EUR under Section 7 of the agreement?", "Vendor_Agreement.pdf", "7.1"),
        mk("D3", "factual", "Which indemnities does Table 9.1 list for a data breach?", "Vendor_Agreement.pdf", "Table 9.1"),
        mk("D4", "unanswerable", "Who won the football match yesterday?"),
        mk("D5", "unanswerable", "What colour is the sky on Mars during sunrise?"),
        mk("D6", "unanswerable", "How do I bake sourdough bread at home?")]}))
    out, env = tmp_path / "calibration.json", tmp_path / "chosen.env"
    assert cal.main([str(eu), str(ct), "--dataset", str(dev), "--out", str(out), "--write-env", str(env)]) == 0
    result = json.loads(out.read_text())
    assert result["n_dev"] == 6 and result["dev_accuracy"] == 1.0 and len(result["features"]) == 6
    ans_scores = [f["top_score"] for f in result["features"] if f["answerable"]]
    un_scores = [f["top_score"] for f in result["features"] if not f["answerable"]]
    assert min(ans_scores) > max(un_scores)                                        # the cross-encoder separates them
    assert result["current"]["abstain_rerank_min"] == 2.5 and "approximate" in result["note"]
    written = dict(l.split("=") for l in env.read_text().splitlines())
    assert float(written["LEXORA_ABSTAIN_RERANK_MIN"]) == result["abstain_rerank_min"]
    text = capsys.readouterr().out
    assert "Chosen: abstain_rerank_min=" in text and f"Written to {env}" in text
