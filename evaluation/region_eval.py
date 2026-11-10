"""Citation precision at REGION level (assignment extended feature 4: bounding-box citations).

A citation names a document, a page and a quote. The evidence viewer draws the quote on the page image, so what the
user sees is a REGION. This module checks, for every emitted citation, whether that region is the right one:

  located         the quote can be found on the cited page of the PDF (so it can be drawn at all)
  page hit        the cited page is a page of the labelled evidence
  region hit      the located quote lies inside the labelled evidence region on that page: at least 50 % of the quote's
                  rectangle overlaps the bounding box of a labelled clause (the regions of the indexed chunks whose
                  document and clause match the label) or the rectangle of the labelled snippet

  region precision = region hits / all emitted citations      (page precision is the same with page hits)

A citation to another clause on the right page is a page hit but not a region hit; that gap is what page-level
precision cannot see. Only emitted citations of answers are counted, and a citation that is not located counts as a
miss. Needs the per-citation detail that ``evaluate`` stores in each row (``citations``).
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

from evaluation.metrics import chunk_matches
from schemas import Chunk
from services.render import _find_quote

Rect = tuple[float, float, float, float]
MIN_OVERLAP = 0.5


def _area(r: Rect) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def _intersection(a: Rect, b: Rect) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def union(rects: list[Rect]) -> Rect | None:
    return (min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects), max(r[3] for r in rects)) if rects else None


def quote_rect(pdf: Path, page_number: int, quote: str) -> Rect | None:
    """The rectangle that the evidence viewer would highlight for this quote, or None if it is not on the page."""
    with pymupdf.open(pdf) as doc:
        if not 1 <= page_number <= doc.page_count:
            return None
        return union([tuple(r) for r in _find_quote(doc[page_number - 1], quote)])


def gold_regions(chunks: list[Chunk], evidence: list[dict], pdf_for) -> dict[tuple[str, int], list[Rect]]:
    """{(document, page number): regions} of the labelled evidence: the chunk regions of the labelled clause plus
    the rectangle of the labelled snippet. ``pdf_for(name)`` returns the PDF path of a document (or None)."""
    out: dict[tuple[str, int], list[Rect]] = {}
    for e in evidence:
        for c in chunks:
            if chunk_matches(c, e):
                for r in c.regions:
                    out.setdefault((e["document"], r.page_index + 1), []).append(tuple(r.bbox))
        pdf = pdf_for(e["document"])
        if pdf and e.get("snippet"):
            rect = quote_rect(pdf, e["page"], e["snippet"])
            if rect:
                out.setdefault((e["document"], e["page"]), []).append(rect)
    return out


def score_citations(citations: list[dict], evidence: list[dict], gold: dict, pdf_for) -> list[dict]:
    """One result per citation: located / page hit / region hit."""
    gold_pages = {(e["document"], e["page"]) for e in evidence}
    out = []
    for c in citations:
        key = (c["document"], c.get("page"))
        pdf = pdf_for(c["document"])
        rect = quote_rect(pdf, c["page"], c.get("quote", "")) if pdf and c.get("page") else None
        overlap = max((_intersection(rect, g) / _area(rect) for g in gold.get(key, []) if _area(rect) > 0), default=0.0) if rect else 0.0
        out.append({"document": c["document"], "page": c.get("page"), "status": c.get("status"), "located": rect is not None,
                    "page_hit": key in gold_pages, "region_hit": rect is not None and overlap >= MIN_OVERLAP,
                    "overlap": round(overlap, 3)})
    return out


def summarize(results: list[dict]) -> dict:
    n = len(results)
    if not n:
        return {"n_citations": 0}
    located = [r for r in results if r["located"]]
    return {"n_citations": n, "located_rate": round(len(located) / n, 4),
            "page_precision": round(sum(r["page_hit"] for r in results) / n, 4),
            "region_precision": round(sum(r["region_hit"] for r in results) / n, 4),
            "region_precision_of_located": round(sum(r["region_hit"] for r in located) / len(located), 4) if located else None,
            "page_hits_outside_the_region": sum(r["page_hit"] and not r["region_hit"] for r in results)}


def region_report(case_chunks: list[Chunk], questions, rows: list[dict], pdfs: dict[str, Path], config: str = "hybrid") -> dict:
    """Region-level precision of one configuration. Rows without citation detail (cached by an older version) are
    counted in ``rows_without_detail`` and left out."""
    pdf_for = pdfs.get
    by_id = {q.id: q for q in questions if q.answerable}
    results, missing = [], 0
    for row in rows:
        if row["config"] != config or row["id"] not in by_id:
            continue
        if "citations" not in row:
            missing += 1
            continue
        q = by_id[row["id"]]
        evidence = q.evidence_dicts()
        results += score_citations(row["citations"], evidence, gold_regions(case_chunks, evidence, pdf_for), pdf_for)
    return {"config": config, "rows_without_detail": missing, **summarize(results)}
