from evaluation import table_eval as te


def test_cell_scoring_requires_the_right_text_in_the_right_place():
    truth = [["A", "B"], ["1", ""], ["3", "4"]]
    assert te.score_table(truth, [["A", "B"], ["1", ""], ["3", "4"]]) == (5, 5)
    assert te.score_table(truth, [["A", "B"], ["1", "x"], ["3", "4"]]) == (5, 5)      # an empty truth cell is not scored
    assert te.score_table(truth, [["A", "B"], ["1", ""], ["4", "3"]]) == (3, 5)        # swapped cells are wrong
    assert te.score_table(truth, [["A"]]) == (1, 5)                                    # missing rows and columns
    assert te.score_table(truth, [[" A ", "B"], ["1", ""], ["3", "4"]]) == (5, 5)      # whitespace is normalised


def test_every_case_pdf_is_built_and_has_the_declared_shape(tmp_path):
    import pymupdf
    for case in te.cases():
        pdf = te.build_case_pdf(case, tmp_path / "c.pdf")
        with pymupdf.open(pdf) as d:
            text = d[0].get_text()
        for spec in case.tables:
            assert spec.rows[1][0] in text, case.name                  # the content really is on the page
            assert len(spec.rows) >= 2 and len({len(r) for r in spec.rows}) == 1


def test_the_evaluation_finds_ruled_tables_exactly_and_misses_the_documented_limits():
    result = te.evaluate()
    by_name = {r["case"]: r for r in result["cases"]}
    for name, r in by_name.items():
        if "not expected" in name:
            assert r["tables_found"] == 0                              # the limit is real, and reported as such
        else:
            assert r["tables_found"] == r["tables_expected"] and r["cells_right"] == r["cells_expected"], name
    s = result["summary"]
    assert s["cell_accuracy_ruled_cases"] == 1.0 and s["cell_accuracy_detected"] == 1.0
    assert s["detection_rate"] < 1.0 and s["cell_accuracy_overall"] < s["cell_accuracy_detected"]
    assert (s["captions_ok"], s["captions_expected"]) == (1, 1)


def test_the_report_and_the_real_document_count(tmp_path, simple_pdf):
    result = te.evaluate([simple_pdf])
    assert result["real_documents"] == [{"document": "simple.pdf", "pages": 2, "tables_found": 0}]
    md = te.markdown(result)
    assert "| Case |" in md and "simple.pdf" in md and "Table extraction against" in md


def test_main_writes_json_and_markdown(tmp_path, capsys):
    assert te.main(["--out", str(tmp_path)]) == 0
    assert (tmp_path / "table_eval.json").is_file() and (tmp_path / "table_eval.md").read_text().startswith("Table extraction")
