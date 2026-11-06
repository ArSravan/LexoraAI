"""Show how a PDF is understood by the ingestion pipeline (no search, no LLM).

    python -m scripts.inspect_pdf path/to/file.pdf            # summary + first 10 chunks
    python -m scripts.inspect_pdf path/to/file.pdf --chunks 40
    python -m scripts.inspect_pdf path/to/file.pdf --clause "83(5)"   # full text of matching chunks
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from config import get_settings
from ingestion.chunker import chunk_document
from ingestion.pdf_parser import parse_pdf


def report(path: Path, n_chunks: int = 10, clause: str | None = None) -> str:
    s = get_settings()
    doc = parse_pdf(path, "inspect", ocr_enabled=s.ocr_enabled, ocr_lang=s.ocr_lang, ocr_dpi=s.ocr_dpi,
                    tesseract_cmd=s.tesseract_cmd or None, tables_enabled=s.tables_enabled,
                    scanned_min_chars=s.scanned_min_chars, max_mb=s.max_upload_mb)
    result = chunk_document(doc)
    out = [f"{doc.filename}: {doc.page_count} pages, sha256 {doc.sha256[:12]}...",
           *(f"warning: {w}" for w in doc.warnings)]
    scans = [p for p in doc.pages if p.needs_ocr]
    out.append(f"scanned pages: {[p.page_number for p in scans] or 'none'}"
               + "".join(f"\n  page {p.page_number}: {p.method}, {p.warning}" for p in scans))
    st = result.structure
    out.append(f"structure mode: {st['mode']} | articles {st['n_articles']} | clauses {st['n_clauses']} "
               f"| tables {st['n_tables']}")
    out.append("chunks: " + str(len(result.chunks)) + " " + str(dict(collections.Counter(c.chunk_type for c in result.chunks))))
    if clause:
        for c in result.chunks:
            if c.clause == clause:
                out.append(f"\n[{c.chunk_id}] p.{c.page_number} {c.section} | clause {c.clause} ({c.chunk_type})\n{c.text}")
    else:
        out.append("")
        for c in result.chunks[:n_chunks]:
            out.append(f"[{c.chunk_id}] p.{c.page_number:<3} {c.chunk_type:<8} {c.clause or '-':<14} "
                       f"{len(c.text):>4} chars | {c.text[:70]}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--chunks", type=int, default=10, help="how many chunks to list")
    ap.add_argument("--clause", help="print the full text of chunks with this clause label")
    args = ap.parse_args(argv)
    print(report(args.pdf, args.chunks, args.clause))
    return 0


if __name__ == "__main__":
    sys.exit(main())
