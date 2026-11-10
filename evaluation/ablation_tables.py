"""Ablation: text only versus text + tables (assignment extended feature 7).

    python -m evaluation.ablation_tables                       # about 1 minute with the real models, no LLM

The same contract (generated here, so the truth is known) is indexed twice: once with table extraction switched off
(a table is then ordinary page text) and once with it on (a table becomes a structured chunk with its caption and
"Header: value" pairs). Six questions can only be answered from table cells. A retrieved chunk counts as the evidence
when it contains ALL the strings of the answer (for example "IP infringement" and "1,000,000"), so both indexes are
judged by the same content test, not by clause labels that only one of them has.

Reported per search mode: recall@1, @3, @5 and MRR over the six questions. The question set is small and written by
the author of the system, so it shows a direction, not a significant difference. The other half of the ablation, with
and without the verifier, is the configuration ``hybrid_noverify`` of evaluation.evaluate. Figures (a vision model) were
not implemented, so there is no "+figures" row.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

RESULTS = Path(__file__).resolve().parent / "results"
DOC_NAME = "Services_Agreement.pdf"

QUESTIONS = [
    {"id": "T1", "question": "What is the indemnity amount for IP infringement according to Table 9.1?", "answer": ["IP infringement", "1,000,000"]},
    {"id": "T2", "question": "What is the total of the indemnities listed in Table 9.1?", "answer": ["Total", "3,500,000"]},
    {"id": "T3", "question": "What availability is required for the Support service?", "answer": ["Support", "99.0%"]},
    {"id": "T4", "question": "What response time applies to Backup under the service levels?", "answer": ["Backup", "24 hours"]},
    {"id": "T5", "question": "What is the monthly fee for the Enterprise plan?", "answer": ["Enterprise", "9,800"]},
    {"id": "T6", "question": "How many users does the Standard plan include?", "answer": ["Standard", "50"]},
]
FILLER = [
    "The Supplier shall perform the Services with reasonable skill and care and in accordance with good industry practice.",
    "Each party shall keep the other party's Confidential Information secret and shall use it only for the purposes of this Agreement.",
    "The Customer shall pay each undisputed invoice within thirty days of receipt, and late payments bear interest at the statutory rate.",
    "Neither party excludes liability for death or personal injury caused by negligence, or for fraud or fraudulent misrepresentation.",
    "The Supplier's total aggregate liability under this Agreement shall not exceed EUR 2,000,000 in any contract year.",
    "This Agreement may be terminated by either party on ninety days' written notice, or immediately for a material breach not remedied in thirty days.",
    "The Supplier shall indemnify the Customer against third-party claims arising from the matters listed in Section 9, up to the amounts stated there.",
    "Personal data shall be processed only on the documented instructions of the Customer and in accordance with the applicable data protection law.",
    "A notice under this Agreement must be in writing and delivered by hand or by registered post to the address stated above.",
    "This Agreement is governed by the laws of Germany and the courts of Berlin have exclusive jurisdiction over any dispute.",
]


def build_contract(path: Path) -> Path:
    st = getSampleStyleSheet()["BodyText"]
    grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.8, colors.black), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)])

    def table(rows):
        t = Table(rows)
        t.setStyle(grid)
        return t

    story = [Paragraph("SERVICES AGREEMENT", getSampleStyleSheet()["Title"]), Spacer(1, 8)]
    for n in range(1, 13):
        story.append(Paragraph(f"Section {n}. {['Definitions', 'Services', 'Fees', 'Term', 'Service Levels', 'Confidentiality', 'Liability', 'Data Protection', 'Indemnities', 'Notices', 'Assignment', 'Fees Schedule'][n - 1]}", st))
        for i, text in enumerate(FILLER[: 4 + n % 4], 1):
            story.append(Paragraph(f"{n}.{i} {text}", st))
        if n == 5:
            story += [Paragraph("Table 5.2 Service levels", st), table([["Service", "Availability", "Response time"],
                      ["Platform", "99.9%", "4 hours"], ["Support", "99.0%", "8 hours"], ["Backup", "99.5%", "24 hours"]]), Spacer(1, 10)]
        if n == 9:
            story += [Paragraph("Table 9.1 Indemnity obligations", st), table([["Item", "Amount (EUR)", "Basis"],
                      ["Data breach", "2,000,000", "per incident"], ["IP infringement", "1,000,000", "per claim"],
                      ["Confidentiality", "500,000", "per claim"], ["Total", "3,500,000", "all heads"]]), Spacer(1, 10)]
        if n == 12:
            story += [Paragraph("Table 12.1 Fees", st), table([["Plan", "Monthly fee (EUR)", "Users"],
                      ["Basic", "1,200", "10"], ["Standard", "3,400", "50"], ["Enterprise", "9,800", "250"]]), Spacer(1, 10)]
        if n in (4, 8):
            story.append(PageBreak())
    SimpleDocTemplate(str(path), pagesize=A4).build(story)
    return path


def rank_of_evidence(chunks, answer: list[str]) -> int | None:
    """1-based rank of the first chunk that contains every answer string, or None."""
    return next((i for i, c in enumerate(chunks, 1) if all(s in c.text for s in answer)), None)


def scores(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {f"recall@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 4) for k in (1, 3, 5)}
    out["mrr"] = round(sum(1 / r for r in ranks if r) / n, 4)
    return out


def run(workdir: Path | None = None) -> dict:
    previous = os.environ.get("LEXORA_STORAGE_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["LEXORA_STORAGE_DIR"] = str(Path(tmp) / "storage")
        try:
            from config import get_settings
            get_settings.cache_clear()
            from ingestion.pdf_parser import parse_pdf
            from retrieval import hybrid
            from services import indexer
            pdf = build_contract(Path(tmp) / DOC_NAME)
            n_table_chunks, result = {}, {"questions": len(QUESTIONS), "document": DOC_NAME, "modes": {}}
            for label, tables in (("text only", False), ("text + tables", True)):
                case = "abl-" + label.replace(" ", "").replace("+", "")
                parsed = parse_pdf(pdf, "doc-" + case, DOC_NAME, ocr_enabled=False, tables_enabled=tables)
                res = indexer.index_parsed(case, parsed)
                n_table_chunks[label] = sum(1 for c in res.chunks if c.chunk_type == "table")
                for mode, rerank in (("bm25", False), ("hybrid + rerank", True)):
                    ranks = []
                    for q in QUESTIONS:
                        hits, _ = hybrid.retrieve(case, q["question"], "hybrid" if rerank else "bm25", final_k=5, rerank=rerank)
                        ranks.append(rank_of_evidence([h.chunk for h in hits], q["answer"]))
                    result["modes"].setdefault(mode, {})[label] = {**scores(ranks), "ranks": ranks}
            result["table_chunks"] = n_table_chunks
            return result
        finally:
            from retrieval import vector_store
            vector_store.close_clients()             # chroma holds its files open; the temp folder is deleted next
            get_settings.cache_clear()
            if previous is None:
                os.environ.pop("LEXORA_STORAGE_DIR", None)
            else:
                os.environ["LEXORA_STORAGE_DIR"] = previous
            get_settings.cache_clear()


def markdown(r: dict) -> str:
    rows = ["| Search | Index | recall@1 | recall@3 | recall@5 | MRR |", "|---|---|---|---|---|---|"]
    for mode, by in r["modes"].items():
        for label, m in by.items():
            rows.append(f"| {mode} | {label} | {m['recall@1']:.2f} | {m['recall@3']:.2f} | {m['recall@5']:.2f} | {m['mrr']:.2f} |")
    return (f"Text only versus text + tables on a generated contract ({r['document']}), {r['questions']} questions answerable only from table cells. "
            f"Table chunks in the index: {r['table_chunks']}.\n\n" + "\n".join(rows) +
            "\n\nA chunk counts as the evidence when it contains every string of the answer. Six questions: a direction, not a significance test.\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    r = run()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "ablation_tables.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    md = markdown(r)
    (a.out / "ablation_tables.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
