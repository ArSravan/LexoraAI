"""Pure helpers that turn API data into things worth showing. No Streamlit calls here, so they are testable."""
from __future__ import annotations

CHECK_LABELS = [("document_exists", "document"), ("page_exists", "page"), ("clause_exists", "clause"),
                ("text_in_passage", "quote"), ("passage_retrieved", "retrieved"), ("supports_claim", "supports claim")]

_COLORS = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red", "INSUFFICIENT EVIDENCE": "red",
           "VERIFIED": "green", "PARTIAL": "orange", "UNVERIFIED": "red", "FABRICATED": "red",
           "indexed": "green", "indexing": "blue", "queued": "gray", "failed": "red"}


def badge(text: str) -> str:
    """Streamlit colour markdown, e.g. ':green[HIGH]'."""
    return f":{_COLORS.get(text, 'gray')}[**{text}**]"


def citation_label(ref: dict) -> str:
    """'Contract.pdf, p.3, clause 7.2' - what a lawyer would write."""
    page = f", p.{ref['page']}" if ref.get("page") else ""
    clause = f", clause {ref['clause']}" if ref.get("clause") else ""
    return f"{ref.get('document') or 'unknown document'}{page}{clause}"


def check_list(checks: dict) -> list[tuple[str, bool]]:
    """The six verification checks in a fixed order, as (label, passed)."""
    return [(label, bool(checks.get(key))) for key, label in CHECK_LABELS]


def check_line(checks: dict) -> str:
    return "  ".join(f"{'✅' if ok else '❌'} {label}" for label, ok in check_list(checks))


def document_row(doc: dict) -> dict:
    """One row of the documents table."""
    return {"File": doc["filename"], "Status": doc["status"], "Pages": doc["page_count"], "Chunks": doc["n_chunks"],
            "Clauses": doc["n_clauses"], "Tables": doc["n_tables"], "Scanned pages": doc["n_scanned"],
            "OCR": doc["ocr_status"]}


def page_rows(pages: list[dict]) -> list[dict]:
    """One row per page for the explorer: characters, how it was read, OCR confidence, warning."""
    return [{"Page": p["page_number"], "Characters": p["char_count"],
             "Read by": "OCR" if p["method"] == "ocr" else ("scan, not read" if p["is_scanned"] else "text layer"),
             "OCR confidence": f"{p['ocr_confidence']:.0f}%" if p.get("ocr_confidence") is not None else "",
             "Tables": p["n_tables"], "Note": p["warning"]} for p in pages]


def evidence_requests(result: dict) -> list[dict]:
    """Everything the evidence viewer needs for each cited passage of an answer: document id, 1-based page,
    quote, chunk id. Evidence that cannot be located on a page is left out."""
    out = []
    for ci, claim in enumerate(result.get("claims", []), 1):
        for ei, ev in enumerate(claim["evidence"], 1):
            if ev.get("document_id") and ev.get("page_index") is not None:
                out.append({"key": f"c{ci}e{ei}", "document_id": ev["document_id"], "page_number": ev["page_index"] + 1,
                            "quote": ev["ref"].get("supporting_text", ""), "chunk_id": ev.get("matched_chunk_id") or "",
                            "label": citation_label(ev["ref"]), "status": ev["status"]})
    return out


def timing_caption(result: dict) -> str:
    t, tok = result.get("timings", {}), result.get("tokens", {})
    parts = [f"{k.replace('_s', '')} {v}s" for k, v in t.items()]
    if tok:
        parts.append(f"tokens {tok.get('prompt', 0)} in / {tok.get('completion', 0)} out")
    return " | ".join(parts)
