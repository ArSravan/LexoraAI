"""PowerShell and cmd do not expand sample_docs/*.pdf; the scripts do it themselves."""
import argparse
import re
from pathlib import Path

import pytest

from evaluation import table_eval
from services.paths import GlobPaths, expand_globs
from tests.conftest import make_pdf

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["evaluation/evaluate.py", "evaluation/calibrate.py", "evaluation/diagnostics.py", "evaluation/ocr_eval.py",
           "evaluation/table_eval.py", "evaluation/page_anomaly_eval.py", "scripts/search_pdf.py", "scripts/eval_retrieval.py",
           "scripts/ask_pdf.py"]


@pytest.fixture
def folder(tmp_path):
    for name in ("b.pdf", "a.pdf", "c.txt", "d1.pdf", "d2.pdf"):
        (tmp_path / name).write_bytes(b"x")
    return tmp_path


def names(paths):
    return [p.name for p in paths]


def test_a_pattern_is_expanded_sorted_and_only_matches_what_it_says(folder):
    assert names(expand_globs([folder / "*.pdf"])) == ["a.pdf", "b.pdf", "d1.pdf", "d2.pdf"]
    assert names(expand_globs([folder / "d?.pdf"])) == ["d1.pdf", "d2.pdf"]
    assert names(expand_globs([folder / "*.txt"])) == ["c.txt"]


def test_explicit_names_keep_their_place_and_duplicates_are_dropped(folder):
    got = expand_globs([folder / "d2.pdf", folder / "*.pdf", folder / "a.pdf"])
    assert names(got) == ["d2.pdf", "a.pdf", "b.pdf", "d1.pdf"]


def test_a_pattern_without_a_match_is_kept_as_typed(folder):
    assert expand_globs([folder / "*.docx"]) == [folder / "*.docx"]
    assert expand_globs([folder / "missing.pdf"]) == [folder / "missing.pdf"]
    assert all(isinstance(p, Path) for p in expand_globs([str(folder / "a.pdf")]))


def test_the_argparse_action_stores_the_expanded_list(folder):
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+", action=GlobPaths)
    ap.add_argument("--flag", action="store_true")
    ns = ap.parse_args([str(folder / "*.pdf"), "--flag"])
    assert names(ns.pdfs) == ["a.pdf", "b.pdf", "d1.pdf", "d2.pdf"] and ns.flag


def test_every_script_that_takes_documents_uses_the_action():
    for rel in SCRIPTS:
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert re.search(r'add_argument\("pdfs", nargs="[+*]", action=GlobPaths', src), rel
        assert "type=Path" not in re.search(r'add_argument\("pdfs".*', src).group(0), rel


def test_a_script_accepts_a_pattern_end_to_end(tmp_path):
    for n in ("one.pdf", "two.pdf"):
        make_pdf(tmp_path / n, [[(60, 700, "Some text on the page")]])
    assert table_eval.main([str(tmp_path / "*.pdf"), "--out", str(tmp_path / "out")]) == 0
    import json
    docs = json.loads((tmp_path / "out" / "table_eval.json").read_text())["real_documents"]
    assert [d["document"] for d in docs] == ["one.pdf", "two.pdf"]
