"""Lexora AI - Streamlit front end.   Run:  streamlit run app.py   (the API must be running: python main.py)"""
import streamlit as st

st.set_page_config(page_title="Lexora AI", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")

import _bootstrap  # noqa: E402,F401  (makes `import LexoraAI` work in either folder layout)
from frontend import api  # noqa: E402
from frontend.views import cases, chat  # noqa: E402

with st.sidebar:
    st.markdown("## ⚖️ Lexora AI")
    st.caption("Every answer has evidence.")

nav = st.navigation([
    st.Page(cases.render, title="Cases and documents", icon="🗂️", url_path="cases", default=True),
    st.Page(chat.render, title="Ask", icon="💬", url_path="ask"),
])

with st.sidebar:
    st.divider()
    try:
        status = api.get("/status")
    except api.ApiError:
        st.markdown("🔴 API offline")
        st.caption("Start it with `python main.py`")
    else:
        ollama = status["ollama"]
        st.markdown("🟢 API online")
        st.markdown(f"{'🟢' if ollama['model_available'] else '🔴'} Ollama · {ollama['model']}"
                    if ollama["connected"] else "🔴 Ollama offline")
        if ollama["connected"] and not ollama["model_available"]:
            st.caption(f"Run: ollama pull {ollama['model']}")
        st.markdown(f"{'🟢' if status['ocr_available'] else '🟠'} OCR (Tesseract)")
    st.caption("Local-first: documents never leave this machine.")

nav.run()
