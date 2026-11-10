from evaluation import report


def test_a_missing_result_says_which_command_to_run(tmp_path):
    page = report.build(tmp_path)
    assert page.startswith("# Results") and page.count("**Not run yet.**") == len(report.SECTIONS)
    assert "python -m evaluation.ocr_eval sample_docs/*.pdf" in page and "--gen-configs no_retrieval hybrid hybrid_noverify" in page


def test_existing_results_are_included_verbatim_with_their_headings_nested(tmp_path):
    (tmp_path / "summary.md").write_text("Run 2026-11-08\n\n### Retrieval\n\n| a | b |\n|---|---|\n| 1 | 2 |\n", encoding="utf-8")
    (tmp_path / "ocr_eval.md").write_text("OCR text\n", encoding="utf-8")
    page = report.build(tmp_path)
    assert "\n### Retrieval\n" in page and "####" not in page                  # the file's headings nest under the section heading
    assert "| 1 | 2 |" in page and "OCR text" in page and page.count("**Not run yet.**") == len(report.SECTIONS) - 2
    assert "\n## Retrieval, answers, citations, manual grades\n" in page


def test_write_then_check(tmp_path):
    results, target, readme = tmp_path / "r", tmp_path / "docs" / "RESULTS.md", tmp_path / "README.md"
    results.mkdir()
    (results / "summary.md").write_text("Run X\n", encoding="utf-8")
    readme.write_text(f"{report.START}\n{report.END}\n", encoding="utf-8")
    args = ["--results", str(results), "--target", str(target), "--readme", str(readme)]
    assert report.main([*args, "--check"]) == 1                                   # nothing written yet
    assert report.main(args) == 0 and target.is_file()
    assert report.main([*args, "--check"]) == 0
    (results / "summary.md").write_text("Run Y\n", encoding="utf-8")              # the measurements changed
    assert report.main([*args, "--check"]) == 1


def test_demote_only_touches_heading_lines():
    assert report.demote("# T\ntext # not a heading\n## U") == "### T\ntext # not a heading\n#### U"


def test_every_section_names_a_command_that_exists_in_the_repository():
    from pathlib import Path
    root = Path(report.__file__).resolve().parents[1]
    for _, _, _, command in report.SECTIONS:
        module = command.split()[2]
        assert (root / (module.replace(".", "/") + ".py")).is_file(), module


# ------------------------------------------------------------------------------ the generated "Reading the numbers"
import json

R = {label: {"recall@1": 0.5, "recall@3": 0.6, "recall@5": 0.7, "full@5": 0.5, "precision@5": 0.2, "mrr": 0.6, "latency_ms": 20.0} for label in report.LABELS}
R["Dense only"].update({"recall@5": 0.55, "mrr": 0.45})
R["Hybrid + rerank"].update({"recall@3": 0.75, "recall@5": 0.65, "mrr": 0.6, "latency_ms": 480.0})
R["Hybrid + rerank + query decomposition"].update({"full@5": 0.8})
SUMMARY_JSON = {
    "meta": {"n_answerable": 11}, "retrieval": R,
    "by_category": {"multihop / Hybrid + rerank": {"full@5": 0.4}, "multihop / Hybrid + rerank + query decomposition": {"full@5": 0.6}},
    "generation": {"no_retrieval": {"mean_total_s": 16.0, "answer_correct_proxy": 0.5},
                   "hybrid": {"mean_total_s": 70.0, "answer_correct_proxy": 0.7, "citations_verified_rate": 0.96, "citation_precision": 0.75,
                              "n_answerable": 11, "answerable_refused_ids": ["M1"], "false_abstention_rate": 0.09, "unsupported_claim_rate": 0.02},
                   "hybrid_noverify": {"mean_total_s": 69.0, "unsupported_claim_rate": 0.10, "false_abstention_rate": 0.0}},
    "manual": {"grader": "Sravan", "configs": {"no_retrieval": {"answerable_mean_score": 0.73, "unanswerable_handled_correctly": 0.5},
                                              "hybrid": {"answerable_mean_score": 0.91, "unanswerable_handled_correctly": 1.0}}},
    "region": {"n_citations": 30, "region_precision": 0.7, "page_precision": 0.9, "page_hits_outside_the_region": 4},
}


def test_the_sentences_follow_the_numbers_not_a_fixed_story():
    text = report.reading(SUMMARY_JSON)
    assert 'The weakest configuration on recall@5 is "Dense only"' in text
    assert "Reranking** raises recall@3 (0.60 to 0.75), lowers recall@5 (0.70 to 0.65) and leaves unchanged MRR (0.60 to 0.60)" in text
    assert "480 ms" in text and "Query decomposition** raises full@5 (0.50 to 0.80)" in text and "multi-hop full@5 (0.40 to 0.60)" in text
    assert "graded by hand (Sravan)" in text and "0.91 with retrieval and 0.73 without" in text and "70 s per question against 16 s" in text
    assert "0.96 of the emitted citations" in text and "25% of the cited passages" in text and "region level the precision is 0.70" in text
    assert "The hybrid configuration refused 1 of 11 answerable questions (M1)." in text
    assert "10% of the claims shown are not verified, against 2% with it" in text


def test_reversed_results_give_reversed_words():
    s = json.loads(json.dumps(SUMMARY_JSON))
    s["retrieval"]["Hybrid + rerank"].update({"recall@3": 0.5, "recall@5": 0.9})
    assert "Reranking** lowers recall@3 (0.60 to 0.50), raises recall@5 (0.70 to 0.90)" in report.reading(s)


def test_missing_parts_are_skipped_without_error():
    assert report.reading({}) == ""
    only_gen = {"generation": {"no_retrieval": {"mean_total_s": 1.0, "answer_correct_proxy": 0.1},
                               "hybrid": {"mean_total_s": 2.0, "answer_correct_proxy": 0.9, "citations_verified_rate": None, "n_answerable": 3,
                                          "false_abstention_rate": 0.0}}}
    text = report.reading(only_gen)
    assert "automatic proxy; not yet graded by hand" in text and "refused 0 of 3 answerable" in text and "Citations" not in text


def test_the_readme_block_is_replaced_between_the_markers_and_nothing_else_changes(tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    (res / "summary.md").write_text("Run X\n\n### Retrieval\n\n| a |\n|---|\n| 1 |\n", encoding="utf-8")
    (res / "summary.json").write_text(json.dumps(SUMMARY_JSON), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(f"# T\n\nbefore\n\n{report.START}\nOLD\n{report.END}\n\nafter\n", encoding="utf-8")
    assert not report.readme_is_current(readme, res)
    assert report.update_readme(readme, res)
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("# T\n\nbefore\n\n") and text.endswith("\n\nafter\n") and "OLD" not in text
    assert "#### Retrieval" in text and "#### Reading the numbers" in text and "| 1 |" in text
    assert report.readme_is_current(readme, res)
    assert report.update_readme(readme, res) and readme.read_text(encoding="utf-8") == text          # idempotent


def test_without_markers_or_summary_the_readme_is_left_alone(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# no markers\n", encoding="utf-8")
    (tmp_path / "summary.md").write_text("Run\n", encoding="utf-8")
    assert not report.update_readme(readme, tmp_path) and readme.read_text(encoding="utf-8") == "# no markers\n"
    readme.write_text(f"{report.START}\n{report.END}\n", encoding="utf-8")
    assert not report.update_readme(readme, tmp_path / "empty")


def test_check_covers_the_readme_block_too(tmp_path):
    res, target, readme = tmp_path / "r", tmp_path / "RESULTS.md", tmp_path / "README.md"
    res.mkdir()
    (res / "summary.md").write_text("Run X\n", encoding="utf-8")
    readme.write_text(f"{report.START}\n{report.END}\n", encoding="utf-8")
    args = ["--results", str(res), "--target", str(target), "--readme", str(readme)]
    assert report.main(args) == 0 and report.main([*args, "--check"]) == 0
    readme.write_text(readme.read_text(encoding="utf-8").replace("Run X", "Run Z"), encoding="utf-8")
    assert report.main([*args, "--check"]) == 1
