"""Measured optimisation: the embedding cache (assignment extended feature 11).

    python -m evaluation.bench_indexing sample_docs/ePrivacy_Directive_2002_58.pdf       # about a minute with the real model

Embedding is the slowest part of indexing. Every (model, text) pair is stored in SQLite, so indexing the same text
again costs a lookup. The benchmark runs in a throw-away storage folder: it extracts and chunks the PDF once, then
embeds all chunk texts twice (cold, then warm) and indexes the whole document twice, and reports the times.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def bench(pdf: Path) -> dict:
    previous = os.environ.get("LEXORA_STORAGE_DIR")
    try:
        return _bench(pdf)
    finally:                                                  # leave the environment as it was found
        if previous is None:
            os.environ.pop("LEXORA_STORAGE_DIR", None)
        else:
            os.environ["LEXORA_STORAGE_DIR"] = previous
        from config import get_settings
        get_settings.cache_clear()


def _bench(pdf: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["LEXORA_STORAGE_DIR"] = tmp
        from config import get_settings
        get_settings.cache_clear()
        from ingestion.chunker import chunk_document
        from ingestion.pdf_parser import parse_pdf
        from retrieval.embeddings import embed_texts
        from services import indexer
        s = get_settings()
        t = time.perf_counter()
        parsed = parse_pdf(pdf, "bench", ocr_enabled=False, tables_enabled=s.tables_enabled)
        parse_s = time.perf_counter() - t
        t = time.perf_counter()
        chunks = chunk_document(parsed, s.chunk_max_chars, s.chunk_overlap).chunks
        chunk_s = time.perf_counter() - t
        texts = [c.index_text for c in chunks]
        t = time.perf_counter()
        embed_texts(texts)
        cold = time.perf_counter() - t
        t = time.perf_counter()
        embed_texts(texts)
        warm = time.perf_counter() - t
        # whole-document indexing: the cache is already warm for these texts, so this is the "re-index" case
        t = time.perf_counter()
        indexer.index_parsed("bench-case", parsed)
        reindex = time.perf_counter() - t
        from retrieval import vector_store
        vector_store.close_clients()                          # chroma holds its files open; the temp folder is deleted next
        get_settings.cache_clear()
    return {"document": pdf.name, "pages": parsed.page_count, "chunks": len(chunks), "parse_s": round(parse_s, 2),
            "chunk_s": round(chunk_s, 2), "embed_cold_s": round(cold, 3), "embed_warm_s": round(warm, 3),
            "embed_speedup": round(cold / max(warm, 1e-6), 1), "index_with_warm_cache_s": round(reindex, 2),
            "embedding_model": s.embedding_model}


def markdown(r: dict) -> str:
    return (f"Embedding cache on {r['document']} ({r['pages']} pages, {r['chunks']} chunks, model `{r['embedding_model']}`).\n\n"
            "| Step | Seconds |\n|---|---|\n"
            f"| Extract the PDF | {r['parse_s']} |\n| Chunk | {r['chunk_s']} |\n"
            f"| Embed all chunks, cache empty | {r['embed_cold_s']} |\n| Embed the same chunks again, cache warm | {r['embed_warm_s']} |\n"
            f"| Index the document again (chunk, embed from cache, write both stores) | {r['index_with_warm_cache_s']} |\n\n"
            f"Speed-up of the embedding step: {r['embed_speedup']}x.\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    r = bench(a.pdf)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "bench_indexing.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    md = markdown(r)
    (a.out / "bench_indexing.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
