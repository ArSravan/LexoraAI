"""Table extraction accuracy against a manual ground truth (assignment extended feature 2).

    python -m evaluation.table_eval [sample_docs/*.pdf]            # a few seconds, no LLM

The ground truth is written by hand in this file: each case is a small PDF whose tables have a known content and
layout (plain grids, a header that spans columns, a label that spans rows, wrapped multi-line cells, two tables on a
page, a long table, and tables WITHOUT full ruling lines (horizontal rules only, or none), which the pdfplumber "lines" strategy is
not expected to find).
The PDF is read by the production code (``extract_tables`` and, for the caption, ``parse_pdf``) and compared cell by cell.

  detected        tables found / tables expected
  cell accuracy   ground-truth non-empty cells that are found with exactly the right text in the right row and column,
                  over the DETECTED tables (an undetected table is counted in "overall", not here)
  overall         the same, but an undetected table scores 0 for all its cells
  caption         the "Table N ..." line above the table was attached to it

The case set is synthetic (so the truth is exact) and small. The three sample documents contain no ruled tables, which
is reported too: the real-document row says how many tables the extractor finds in them, not how accurate it is.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ingestion.pdf_parser import parse_pdf
from ingestion.table_extractor import extract_tables
from services.paths import GlobPaths

RESULTS = Path(__file__).resolve().parent / "results"
GRID = ("GRID", (0, 0), (-1, -1), 0.8, colors.black)
_NORMAL = getSampleStyleSheet()["BodyText"]


@dataclass
class TableSpec:
    rows: list[list[str]]                       # the ground truth: spanned cells are "" (only the first cell holds text)
    style: list[tuple] = field(default_factory=lambda: [GRID])
    caption: str = ""
    ruled: bool = True                          # False: no lines are drawn (expected to be missed)
    wrap: bool = False                          # cells are Paragraphs that wrap onto several lines
    col_widths: list[float] | None = None


@dataclass
class Case:
    name: str
    tables: list[TableSpec]
    note: str = ""


def _story(case: Case):
    story = []
    for t in case.tables:
        if t.caption:
            story.append(Paragraph(t.caption, _NORMAL))
        data = [[Paragraph(c, _NORMAL) if t.wrap and c else c for c in row] for row in t.rows]
        tbl = Table(data, colWidths=t.col_widths)
        tbl.setStyle(TableStyle(t.style if t.ruled else [("FONTSIZE", (0, 0), (-1, -1), 10)]))
        story += [tbl, Spacer(1, 40)]
    return story


def build_case_pdf(case: Case, path: Path) -> Path:
    SimpleDocTemplate(str(path), pagesize=A4, topMargin=70, bottomMargin=50).build(_story(case))
    return path


def cases() -> list[Case]:
    num = [["Item", "Amount (EUR)", "Share", "Page"], ["Data breach", "5,000,000", "62.5%", "14"],
           ["IP infringement", "3,000,000", "37.5%", "15"], ["Confidentiality", "1,250,000", "15.6%", "16"],
           ["Total", "8,000,000", "100%", "18"]]
    legal = [["Obligation", "Clause", "Deadline"], ["Notify the authority", "33(1)", "72 hours"],
             ["Notify the data subject", "34(1)", "without undue delay"], ["Respond to a request", "12(3)", "one month"]]
    long_rows = [["No.", "Provision", "Maximum fine", "Basis", "Note"]] + [
        [str(i), f"Article {80 + i}", f"EUR {i * 1_000_000:,}", f"{i}% of turnover", f"see recital {i}"] for i in range(1, 13)]
    return [
        Case("plain grid, numbers", [TableSpec(num)]),
        Case("legal table with caption", [TableSpec(legal, caption="Table 9.1 Notification obligations")],
             "the caption must be attached"),
        Case("header spanning three columns", [TableSpec(
            [["Indemnity obligations", "", ""], ["Item", "Amount", "Page"], ["Breach", "5,000,000", "14"], ["IP", "3,000,000", "15"]],
            style=[GRID, ("SPAN", (0, 0), (2, 0))])]),
        Case("label spanning rows", [TableSpec(
            [["Category", "Item", "Amount"], ["Liability", "Cap", "10,000,000"], ["", "Floor", "1,000,000"], ["", "Excess", "500,000"],
             ["Indemnity", "Breach", "5,000,000"]], style=[GRID, ("SPAN", (0, 1), (0, 3))])]),
        Case("wrapped multi-line cells", [TableSpec(
            [["Clause", "Text"], ["7.1", "The Supplier's total aggregate liability under this Agreement shall not exceed EUR 10,000,000 in any year."],
             ["7.2", "Nothing in this Section limits liability for gross negligence or wilful misconduct of either party."]],
            wrap=True, col_widths=[60, 300])]),
        Case("two tables on one page", [TableSpec(num[:3]), TableSpec(legal[:3])]),
        Case("long table, 12 rows", [TableSpec(long_rows)]),
        Case("horizontal rules only (not expected to work)", [TableSpec(
            num, style=[("LINEABOVE", (0, 0), (-1, 0), 1, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                        ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black)], ruled=True)],
             "a table typeset with horizontal lines only has no column lines to read"),
        Case("no ruling lines (not expected to work)", [TableSpec(num, ruled=False)],
             "the extractor only reads tables drawn with lines; this is the documented limit"),
    ]


# ------------------------------------------------------------------------------ scoring
def _norm(s: str) -> str:
    return " ".join(str(s).split())


def score_table(truth: list[list[str]], found: list[list[str]]) -> tuple[int, int]:
    """(cells right, cells expected): every non-empty ground-truth cell must match at the same row and column."""
    right = expected = 0
    for r, row in enumerate(truth):
        for c, cell in enumerate(row):
            if not cell:
                continue
            expected += 1
            if r < len(found) and c < len(found[r]) and _norm(found[r][c]) == _norm(cell):
                right += 1
    return right, expected


def evaluate_case(case: Case, tmp: Path) -> dict:
    pdf = build_case_pdf(case, tmp / "case.pdf")
    with pymupdf.open(pdf) as d:
        pages = list(range(d.page_count))
    found = [t for pi in pages for t in extract_tables(str(pdf), [pi]).get(pi, [])]
    found.sort(key=lambda t: (t.page_index, t.bbox[1]))
    parsed = parse_pdf(pdf, "tbl")
    captions = [t.caption for p in parsed.pages for t in p.tables]
    right = expected = det_right = det_expected = 0
    per_table = []
    for i, spec in enumerate(case.tables):
        r, e = score_table(spec.rows, found[i].rows) if i < len(found) else (0, sum(1 for row in spec.rows for c in row if c))
        right, expected = right + r, expected + e
        detected = i < len(found)
        if detected:
            det_right, det_expected = det_right + r, det_expected + e
        per_table.append({"detected": detected, "cells_right": r, "cells_expected": e,
                          "caption_ok": (_norm(captions[i]) == _norm(spec.caption)) if detected and spec.caption and i < len(captions) else None})
    return {"case": case.name, "note": case.note, "tables_expected": len(case.tables), "tables_found": len(found),
            "cells_right": right, "cells_expected": expected, "detected_cells_right": det_right,
            "detected_cells_expected": det_expected, "tables": per_table}


def real_documents(pdfs: list[Path]) -> list[dict]:
    out = []
    for pdf in pdfs:
        with pymupdf.open(pdf) as d:
            n = d.page_count
        found = extract_tables(str(pdf), list(range(n)))
        out.append({"document": pdf.name, "pages": n, "tables_found": sum(len(v) for v in found.values())})
    return out


def evaluate(pdfs: list[Path] | None = None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        rows = [evaluate_case(c, Path(tmp)) for c in cases()]
    exp_t = sum(r["tables_expected"] for r in rows)
    ruled = [r for r, c in zip(rows, cases()) if "not expected" not in c.name]
    summary = {
        "cases": len(rows), "tables_expected": exp_t, "tables_found": sum(min(r["tables_found"], r["tables_expected"]) for r in rows),
        "detection_rate": round(sum(min(r["tables_found"], r["tables_expected"]) for r in rows) / exp_t, 4),
        "cell_accuracy_detected": round(sum(r["detected_cells_right"] for r in rows) / max(1, sum(r["detected_cells_expected"] for r in rows)), 4),
        "cell_accuracy_overall": round(sum(r["cells_right"] for r in rows) / max(1, sum(r["cells_expected"] for r in rows)), 4),
        "cell_accuracy_ruled_cases": round(sum(r["cells_right"] for r in ruled) / max(1, sum(r["cells_expected"] for r in ruled)), 4),
        "captions_ok": sum(1 for r in rows for t in r["tables"] if t["caption_ok"]),
        "captions_expected": sum(1 for r in rows for t in r["tables"] if t["caption_ok"] is not None),
    }
    return {"summary": summary, "cases": rows, "real_documents": real_documents(pdfs) if pdfs else []}


def markdown(result: dict) -> str:
    rows = ["| Case | Tables found / expected | Cells right / expected | Accuracy |", "|---|---|---|---|"]
    for r in result["cases"]:
        acc = r["cells_right"] / r["cells_expected"]
        rows.append(f"| {r['case']} | {r['tables_found']} / {r['tables_expected']} | {r['cells_right']} / {r['cells_expected']} | {acc:.0%} |")
    s = result["summary"]
    out = ["Table extraction against a hand-written ground truth (synthetic PDFs, exact truth).", "", *rows, "",
           f"* detection: {s['tables_found']} of {s['tables_expected']} tables ({s['detection_rate']:.0%})",
           f"* cell accuracy over detected tables: {s['cell_accuracy_detected']:.1%}; overall (missed tables count as 0): "
           f"{s['cell_accuracy_overall']:.1%}; over the cases with ruling lines: {s['cell_accuracy_ruled_cases']:.1%}",
           f"* captions attached: {s['captions_ok']} of {s['captions_expected']}"]
    if result["real_documents"]:
        out += ["", "Tables the extractor finds in real documents (no ground truth, so no accuracy):", "",
                "| Document | Pages | Tables found |", "|---|---|---|"]
        out += [f"| {d['document']} | {d['pages']} | {d['tables_found']} |" for d in result["real_documents"]]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="*", action=GlobPaths, help="real documents to count tables in (optional)")
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    result = evaluate(a.pdfs)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "table_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = markdown(result)
    (a.out / "table_eval.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
