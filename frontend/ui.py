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
             "Tables": p["n_tables"], "Unusual page": "⚠ yes" if p.get("anomalous") else "",
             "Note": p["warning"]} for p in pages]


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


# ------------------------------------------------------------------------------ dashboard, evaluation and settings
def _f(v, digits: int = 2) -> str:
    return "-" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def stat_rows(stats: dict) -> list[tuple[str, int]]:
    return [("Cases", stats["cases"]), ("Documents indexed", stats["documents"]), ("Indexed chunks", stats["chunks"]),
            ("Questions asked", stats["questions"]), ("Abstentions", stats["abstentions"])]


def recent_rows(recent: list[dict]) -> list[dict]:
    import time
    return [{"When": time.strftime("%d %b %H:%M", time.localtime(r["created_at"])), "Question": r["question"][:110],
             "Answer": r["confidence"], "Case": r.get("case_title") or "", "Mode": r["mode"], "Model": r["model"],
             "Seconds": round(r["latency"], 1)} for r in recent]


def settings_groups(rows: list[dict]) -> dict[str, list[dict]]:
    """Group the flat settings list for display."""
    groups = {"Storage and upload": ("storage_dir", "max_upload_mb", "max_docs_per_case", "seed"),
              "Extraction, OCR, tables": ("scanned_min_chars", "ocr_enabled", "ocr_lang", "ocr_dpi", "tesseract_cmd", "tables_enabled"),
              "Local language model": tuple(r["name"] for r in rows if r["name"].startswith("ollama_")),
              "Retrieval": ("embedding_model", "embedding_batch_size", "reranker_model", "rerank_enabled", "retrieval_mode",
                            "initial_top_k", "final_top_k", "rrf_k", "dense_min_similarity", "query_decomposition",
                            "chunk_max_chars", "chunk_overlap"),
              "Verification and abstention": ("quote_match_min", "abstain_rerank_min", "abstain_coverage_min", "high_conf_rerank",
                                              "medium_conf_rerank", "verify_retries")}
    by = {r["name"]: r for r in rows}
    out = {g: [{"Setting": n, "Value": by[n]["value"], "Environment variable": by[n]["env"]} for n in names if n in by]
           for g, names in groups.items()}
    placed = {n for names in groups.values() for n in names}
    rest = [r for r in rows if r["name"] not in placed]
    if rest:
        out["Other"] = [{"Setting": r["name"], "Value": r["value"], "Environment variable": r["env"]} for r in rest]
    return {g: v for g, v in out.items() if v}


def retrieval_rows(retrieval: dict) -> list[dict]:
    cols = ["recall@1", "recall@3", "recall@5", "full@5", "precision@5", "mrr"]
    return [{"Configuration": label, **{c: _f(m.get(c)) for c in cols}, "latency ms": f"{m.get('latency_ms', 0):.0f}"}
            for label, m in retrieval.items()]


GENERATION_LABELS = [("Answer correct, automatic key-fact proxy", "answer_correct_proxy"),
                     ("Citations verified by the validator", "citations_verified_rate"),
                     ("Citation precision vs labelled evidence", "citation_precision"),
                     ("Claims shown that are not verified", "unsupported_claim_rate"),
                     ("Abstention accuracy", "abstention_accuracy"), ("False-abstention rate", "false_abstention_rate"),
                     ("Correct-abstention rate", "correct_abstention_rate"), ("Mean total time (s)", "mean_total_s"),
                     ("Mean completion tokens", "mean_completion_tokens")]


def generation_rows(generation: dict) -> list[dict]:
    return [{"Measure": label, **{cfg: _f(g.get(key)) for cfg, g in generation.items()}} for label, key in GENERATION_LABELS]


def manual_rows(manual: dict | None) -> list[dict]:
    if not manual:
        return []
    keys = [("Answers graded", None), ("Answerable: mean score", "answerable_mean_score"),
            ("Answerable: strictly correct", "answerable_strictly_correct"),
            ("Unanswerable: handled correctly", "unanswerable_handled_correctly"), ("All questions: behaviour correct", "behaviour_correct_all")]
    return [{"Measure": label, **{cfg: (f"{c['n_graded']} / {c['n_total']}" if key is None else _f(c[key]))
                                  for cfg, c in manual["configs"].items()}} for label, key in keys]


def region_rows(region: dict | None) -> list[dict]:
    if not region or not region.get("n_citations"):
        return []
    return [{"Measure": m, "Value": _f(region.get(k))} for m, k in
            [("Citations", "n_citations"), ("Quote found on the cited page", "located_rate"), ("Page-level precision", "page_precision"),
             ("Region-level precision", "region_precision"), ("Region-level precision of located quotes", "region_precision_of_located")]]


def ocr_rows(ocr: dict | None) -> list[dict]:
    return [] if not ocr else [{"Condition": k, "Pages": v["pages"], "CER": _f(v["cer"], 3), "WER": _f(v["wer"], 3),
                                "Word recall": _f(v["word_recall"], 3), "OCR confidence %": _f(v["ocr_confidence"], 1)}
                               for k, v in ocr["summary"].items()]


def table_eval_rows(tbl: dict | None) -> list[dict]:
    return [] if not tbl else [{"Case": c["case"], "Tables found / expected": f"{c['tables_found']} / {c['tables_expected']}",
                                "Cells right / expected": f"{c['cells_right']} / {c['cells_expected']}"} for c in tbl["cases"]]


def anomaly_rows(ae: dict | None) -> list[dict]:
    if not ae:
        return []
    k = str(ae["chosen_k"])
    return [{"Defect": d["defect"], "AUROC autoencoder": _f(d["auroc_autoencoder"], 3), "AUROC baseline": _f(d["auroc_zscore"], 3),
             "Detected": f"{d['detected'][k]:.0%}", "False alarms on intact pages": f"{d['false_alarms_on_intact_pages'][k]:.1%}"}
            for d in ae["defects"]]


def ablation_rows(abl: dict | None) -> list[dict]:
    if not abl:
        return []
    return [{"Search": mode, "Index": label, **{c: _f(m[c]) for c in ("recall@1", "recall@3", "recall@5", "mrr")}}
            for mode, by in abl["modes"].items() for label, m in by.items()]


def latency_rows(lat: dict | None) -> list[dict]:
    return [] if not lat else [{"Configuration": c, "Questions": v["n"], "Mean s": v["total_s_mean"], "Median s": v["total_s_median"],
                                "95th pct s": v["total_s_p95"], "Generation s": v["generation_s_mean"],
                                "Completion tokens": v["completion_tokens_mean"]} for c, v in lat["configs"].items()]
