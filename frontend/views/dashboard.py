"""Dashboard: how much has been indexed and asked, the latest questions, and the status of the local system."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend import api, ui


def render() -> None:
    st.title("Dashboard")
    try:
        stats, status = api.get("/stats"), api.get("/status")
    except api.ApiError as e:
        st.error(str(e))
        return
    cols = st.columns(5)
    for col, (label, value) in zip(cols, ui.stat_rows(stats)):
        col.metric(label, value)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Recent questions")
        if stats["recent"]:
            st.dataframe(pd.DataFrame(ui.recent_rows(stats["recent"])), hide_index=True, use_container_width=True)
        else:
            st.info("No questions asked yet. Create a case, upload 2-5 PDFs and open Ask.")
    with right:
        st.subheader("System")
        o, s = status["ollama"], status["settings"]
        st.markdown("\n\n".join([
            "🟢 Lexora API",
            f"{'🟢' if o['connected'] else '🔴'} Ollama {o.get('version') or ''} ({o['base_url']})",
            f"{'🟢' if o['model_available'] else '🔴'} Model {o['model']}" + ("" if o["model_available"] else " (not pulled)"),
            f"{'🟢' if status['ocr_available'] else '🟠'} Tesseract OCR",
            f"🟢 Embeddings {s['embedding_model'].split('/')[-1]}",
            f"{'🟢' if s['rerank_enabled'] else '🟠'} Reranker {s['reranker_model'].split('/')[-1]}"
            + ("" if s["rerank_enabled"] else " (disabled)")]))
        if not o["connected"]:
            st.warning("Ollama is not running. Start it with `ollama serve`.")
        elif not o["model_available"]:
            st.warning(f"Model not found. Run `ollama pull {o['model']}`.")
        if o["models"]:
            st.caption("Installed models: " + ", ".join(o["models"]))
