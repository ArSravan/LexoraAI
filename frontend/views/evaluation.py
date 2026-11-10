"""Evaluation: the measured results of the project, read from evaluation/results/ (nothing is computed here)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend import api, ui


def _table(rows: list[dict], missing: str) -> bool:
    if not rows:
        st.info(missing)
        return False
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    return True


def render() -> None:
    st.title("Evaluation")
    try:
        res = api.get("/evaluation")
    except api.ApiError as e:
        st.error(str(e))
        return
    summary = res.get("summary") or {}
    meta = summary.get("meta", {})
    if meta:
        st.caption(f"Run {meta.get('date', '?')} - model {meta.get('ollama_model', '?')}, seed {meta.get('seed', '?')}, "
                   f"{meta.get('n_questions', '?')} held-out questions ({meta.get('n_answerable', '?')} answerable), "
                   f"commit {meta.get('git_commit') or '?'}.")
    tab_a, tab_b, tab_c = st.tabs(["Retrieval and answers", "Extended features", "Latency and cost"])
    with tab_a:
        st.subheader("Retrieval (answerable questions)")
        _table(ui.retrieval_rows(summary.get("retrieval", {})),
               "No evaluation run yet: `python -m evaluation.evaluate sample_docs/*.pdf --retrieval-only`.")
        st.subheader("Answers")
        _table(ui.generation_rows(summary.get("generation", {})) if summary.get("generation") else [],
               "No answer evaluation yet: `python -m evaluation.evaluate sample_docs/*.pdf --gen-configs no_retrieval hybrid`.")
        st.caption("The correctness row is an automatic key-fact proxy, not a human judgement.")
        manual = ui.manual_rows(summary.get("manual"))
        if manual:
            st.subheader(f"Manual grades (graded by {summary['manual']['grader']})")
            _table(manual, "")
        region = ui.region_rows(summary.get("region"))
        if region:
            st.subheader("Citation precision at region level")
            _table(region, "")
    with tab_b:
        st.subheader("OCR accuracy (extended feature: OCR)")
        _table(ui.ocr_rows(res.get("ocr_eval")), "Not run yet: `python -m evaluation.ocr_eval sample_docs/*.pdf`.")
        st.subheader("Table extraction against a ground truth")
        _table(ui.table_eval_rows(res.get("table_eval")), "Not run yet: `python -m evaluation.table_eval sample_docs/*.pdf`.")
        st.subheader("Autoencoder page-anomaly detector")
        _table(ui.anomaly_rows(res.get("page_anomaly_eval")), "Not run yet: `python -m evaluation.page_anomaly_eval sample_docs/*.pdf`.")
        st.subheader("Ablation: text only versus text + tables")
        _table(ui.ablation_rows(res.get("ablation_tables")), "Not run yet: `python -m evaluation.ablation_tables`.")
    with tab_c:
        _table(ui.latency_rows(res.get("latency_report")), "Not run yet: `python -m evaluation.latency_report`.")
        bench = res.get("bench_indexing")
        if bench:
            st.caption(f"Embedding cache: {bench['embed_cold_s']} s cold, {bench['embed_warm_s']} s warm "
                       f"({bench['embed_speedup']}x) on {bench['document']}.")
