"""Settings: what the application is running with and which environment variable changes each value (read-only)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend import api, ui


def render() -> None:
    st.title("Settings")
    st.caption("Read-only. Change a value with an environment variable or in the `.env` file (see `.env.example`) and restart. "
               "Changing the chunk size, the embedding model or the OCR settings needs a re-index of the documents.")
    try:
        rows = api.get("/settings")
    except api.ApiError as e:
        st.error(str(e))
        return
    for group, items in ui.settings_groups(rows).items():
        st.subheader(group)
        st.dataframe(pd.DataFrame(items), hide_index=True, use_container_width=True)
    if st.button("Test the Ollama connection"):
        o = api.get("/status")["ollama"]
        if o["connected"]:
            st.success(f"Connected to Ollama {o['version']}; models: {', '.join(o['models']) or 'none'}")
            if not o["model_available"]:
                st.warning(f"The selected model '{o['model']}' is not pulled. Run `ollama pull {o['model']}`.")
        else:
            st.error("Cannot reach Ollama. Start it with `ollama serve`.")
    st.info("Privacy: documents, embeddings and prompts stay on this machine. Text inside an uploaded PDF is treated as untrusted "
            "data and is never followed as an instruction.")
