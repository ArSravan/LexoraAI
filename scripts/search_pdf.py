"""Index one or more PDFs and run a question through the retrieval stage (no LLM).

    python -m scripts.search_pdf a.pdf b.pdf -q "What is the maximum fine?"
    python -m scripts.search_pdf a.pdf -q "..." --mode bm25 --no-rerank --k 8
    python -m scripts.search_pdf a.pdf -q "..." --compare     # bm25 / dense / hybrid / +rerank / +decomposition

The PDFs are indexed once (under storage/, keyed by the files' hashes) and reused on later runs.
The first run downloads the two small Hugging Face models (about 90 MB each).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from ingestion.pdf_parser import sha256_file
from retrieval.hybrid import retrieve
from services import indexer
from services.paths import GlobPaths, chunks_file

# (label, mode, rerank, decomposition)
CONFIGS = [("BM25 only", "bm25", False, False), ("Dense only", "dense", False, False),
           ("Hybrid (BM25 + dense, RRF)", "hybrid", False, False),
           ("Hybrid + rerank", "hybrid", True, False),
           ("Hybrid + rerank + query decomposition", "hybrid", True, True)]


def ensure_indexed(pdfs: list[Path]) -> str:
    """Index the PDFs into a case whose id is derived from their content, unless already done."""
    digests = sorted(sha256_file(p) for p in pdfs)
    case_id = "cli-" + hashlib.sha1("".join(digests).encode()).hexdigest()[:10]
    if not indexer.index_is_current(case_id):
        if chunks_file(case_id).exists():                  # saved by an older version: it cannot be used for answers
            print("the saved index of these documents is out of date: rebuilding it", file=sys.stderr)
            indexer.remove_case(case_id)
        for i, p in enumerate(pdfs):
            print(f"indexing {p.name} ...", file=sys.stderr)
            indexer.index_pdf(case_id, f"d{i}", p, p.name)
    return case_id


def show(label: str, hits, timings: dict, preview: int = 80) -> str:
    out = [f"\n== {label}   (search {timings['search_s']}s, rerank {timings['rerank_s']}s)"]
    if timings.get("sub_queries"):
        out.append("   sub-queries: " + " | ".join(timings["sub_queries"]))
    for h in hits:
        c = h.chunk
        ranks = f"bm25#{h.bm25_rank or '-'} dense#{h.dense_rank or '-'}"
        extra = f" rrf {h.rrf_score:.4f}" if h.rrf_score is not None else ""
        extra += f" rerank {h.rerank_score:+.2f}" if h.rerank_score is not None else ""
        out.append(f"{h.final_rank:>2}. {c.document_name} p.{c.page_number} clause {c.clause or '-'} "
                   f"[{c.chunk_type}] {ranks}{extra}\n      {c.text[:preview]}")
    if not hits:
        out.append("   (nothing found)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="+", action=GlobPaths)
    ap.add_argument("-q", "--question", required=True)
    ap.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-decompose", action="store_true")
    ap.add_argument("--k", type=int, default=5, help="passages to show")
    ap.add_argument("--compare", action="store_true", help="show every retrieval configuration")
    args = ap.parse_args(argv)
    case_id = ensure_indexed(args.pdfs)
    print(f"question: {args.question}")
    if args.compare:
        for label, mode, rerank, decomp in CONFIGS:
            hits, tm = retrieve(case_id, args.question, mode, final_k=args.k, rerank=rerank, decompose_query=decomp)
            print(show(label, hits, tm))
    else:
        hits, tm = retrieve(case_id, args.question, args.mode, final_k=args.k, rerank=not args.no_rerank,
                            decompose_query=not args.no_decompose)
        print(show(f"{args.mode}" + ("" if args.no_rerank else " + rerank"), hits, tm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
