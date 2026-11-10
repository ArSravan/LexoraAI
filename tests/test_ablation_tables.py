import pymupdf

from evaluation import ablation_tables as ab
from ingestion.pdf_parser import parse_pdf


def test_the_generated_contract_really_contains_every_answer_in_one_table_row(tmp_path):
    pdf = ab.build_contract(tmp_path / "c.pdf")
    parsed = parse_pdf(pdf, "d", ocr_enabled=False, tables_enabled=True)
    assert parsed.page_count >= 3
    cells = [row for p in parsed.pages for t in p.tables for row in t.rows]
    assert {t.caption for p in parsed.pages for t in p.tables} == {"Table 5.2 Service levels", "Table 9.1 Indemnity obligations", "Table 12.1 Fees"}
    for q in ab.QUESTIONS:
        assert any(all(s in " | ".join(row) for s in q["answer"]) for row in cells), q["id"]
    with pymupdf.open(pdf) as d:                                              # the strings are on the pages too
        text = " ".join(p.get_text() for p in d)
    assert all(s in text for q in ab.QUESTIONS for s in q["answer"])


def test_rank_and_scores():
    class C:
        def __init__(self, t):
            self.text = t
    chunks = [C("alpha beta"), C("IP infringement 1,000,000"), C("other")]
    assert ab.rank_of_evidence(chunks, ["IP infringement", "1,000,000"]) == 2
    assert ab.rank_of_evidence(chunks, ["missing"]) is None
    s = ab.scores([1, 2, None, 6])
    assert s == {"recall@1": 0.25, "recall@3": 0.5, "recall@5": 0.5, "mrr": round((1 + 0.5 + 1 / 6) / 4, 4)}


def test_the_ablation_indexes_both_variants_and_reports_every_mode(fake_embedder, fake_reranker, tmp_path):
    r = ab.run()
    assert r["questions"] == 6 and r["table_chunks"]["text only"] == 0 and r["table_chunks"]["text + tables"] >= 3
    assert set(r["modes"]) == {"bm25", "hybrid + rerank"}
    for by in r["modes"].values():
        assert set(by) == {"text only", "text + tables"} and all(len(m["ranks"]) == 6 for m in by.values())
        assert all(0 <= m["recall@5"] <= 1 and 0 <= m["mrr"] <= 1 for m in by.values())
    md = ab.markdown(r)
    assert "| bm25 | text + tables |" in md and "no \"+figures\"" not in md


def test_the_run_restores_the_environment(fake_embedder, fake_reranker, monkeypatch, tmp_path):
    import os
    monkeypatch.setenv("LEXORA_STORAGE_DIR", str(tmp_path / "mine"))
    ab.run()
    assert os.environ["LEXORA_STORAGE_DIR"] == str(tmp_path / "mine")


def test_main_writes_both_files(fake_embedder, fake_reranker, tmp_path):
    assert ab.main(["--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "ablation_tables.json").is_file() and (tmp_path / "o" / "ablation_tables.md").read_text().startswith("Text only")
