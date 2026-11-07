"""Page 1: create cases, upload PDFs, watch indexing, inspect what was extracted."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend import api, ui


def _flash() -> None:
    """Messages that must survive the st.rerun() that refreshes the page after an action."""
    for kind, text in st.session_state.pop("flash", []):
        getattr(st, kind)(text)


def _new_case_form() -> None:
    with st.expander("Create a new case", expanded=not st.session_state.get("case_id")):
        with st.form("new_case", clear_on_submit=True):
            title = st.text_input("Title", placeholder="e.g. Vendor agreement review")
            description = st.text_area("Description (optional)", height=68)
            if st.form_submit_button("Create case", type="primary"):
                try:
                    case = api.post("/cases", {"title": title, "description": description})
                except api.ApiError as e:
                    st.error(str(e))
                else:
                    st.session_state["case_id"] = case["id"]
                    st.session_state["flash"] = [("success", f"Case '{case['title']}' created. Now upload its PDFs.")]
                    st.rerun()


def _upload(case_id: str) -> None:
    n = st.session_state.get("uploader_n", 0)
    files = st.file_uploader("Upload PDFs (drag several files at once)", type=["pdf"], accept_multiple_files=True,
                             key=f"uploader-{case_id}-{n}")
    if not (files and st.button("Upload and index", type="primary")):
        return
    jobs, flash = [], []
    for f in files:
        try:
            jobs.append(api.post(f"/cases/{case_id}/documents", files={"file": (f.name, f.getvalue(), "application/pdf")})["job_id"])
        except api.ApiError as e:
            flash.append(("error", f"{f.name}: {e}"))
    if jobs:
        bar = st.progress(0.0, "Indexing ...")
        for job in api.wait_for_jobs(jobs, lambda frac, msg: bar.progress(frac, msg)):
            if job["status"] == "failed":
                flash.append(("error", f"Indexing failed: {job['error']}"))
        flash.append(("success", f"{len(jobs)} document(s) indexed."))
    st.session_state.update(flash=flash, uploader_n=n + 1)          # a new key empties the uploader
    st.rerun()


def _document(case_id: str, doc: dict) -> None:
    with st.expander(f"{doc['filename']}  -  {doc['status']}", expanded=False):
        st.markdown(f"Status: {ui.badge(doc['status'])}   |   {doc['page_count']} pages, {doc['n_chunks']} chunks, "
                    f"{doc['n_clauses']} clauses, {doc['n_tables']} tables, OCR: {doc['ocr_status']}")
        if doc["error"]:
            st.error(doc["error"])
        for w in doc["warnings"]:
            st.warning(w)
        if doc["status"] == "indexed":
            info = api.get(f"/documents/{doc['id']}/info")
            st.caption(f"Structure detection: {info['structure'].get('mode', '?')}, "
                       f"{info['structure'].get('n_articles', 0)} articles/sections found")
            st.dataframe(pd.DataFrame(ui.page_rows(info["pages"])), hide_index=True, use_container_width=True)
            page = st.number_input("Preview page", 1, max(doc["page_count"], 1), 1, key=f"pg-{doc['id']}")
            png, _ = api.image(f"/documents/{doc['id']}/pages/{page}/image", zoom=1.2)
            left, right = st.columns(2)
            if png:
                left.image(png, caption=f"{doc['filename']} - page {page}")
            text = api.get(f"/documents/{doc['id']}/pages/{page}")
            right.text_area("Extracted text", text["text"] or "(nothing extracted)", height=380, key=f"tx-{doc['id']}-{page}")
        c1, c2, c3 = st.columns([1, 1, 3])
        if c1.button("Re-index", key=f"re-{doc['id']}"):
            api.wait_for_jobs([api.post(f"/cases/{case_id}/documents/{doc['id']}/reindex", force_reparse=True)["job_id"]])
            st.rerun()
        if c2.button("Delete", key=f"del-{doc['id']}"):
            api.delete(f"/cases/{case_id}/documents/{doc['id']}")
            st.rerun()


def render() -> None:
    st.title("Cases and documents")
    _flash()
    try:
        cases = api.get("/cases")
    except api.ApiError as e:
        st.error(str(e))
        return
    _new_case_form()
    if not cases:
        st.info("No cases yet: create one above.")
        return
    ids = [c["id"] for c in cases]
    labels = {c["id"]: f"{c['title']}  ({c['n_docs']} documents)" for c in cases}
    current = st.session_state.get("case_id")
    case_id = st.selectbox("Case", ids, index=ids.index(current) if current in ids else 0, format_func=labels.get)
    st.session_state["case_id"] = case_id
    docs = api.get(f"/cases/{case_id}/documents")
    st.subheader("Documents")
    if docs:
        st.dataframe(pd.DataFrame([ui.document_row(d) for d in docs]), hide_index=True, use_container_width=True)
        for d in docs:
            _document(case_id, d)
    else:
        st.caption("No documents yet.")
    _upload(case_id)
    with st.expander("Danger zone"):
        if st.checkbox("I understand that deleting the case removes its documents and indexes", key=f"ok-{case_id}") \
                and st.button("Delete this case", key=f"delcase-{case_id}"):
            api.delete(f"/cases/{case_id}")
            st.session_state.pop("case_id", None)
            st.rerun()
