# UI verification

What was actually looked at in a real browser, and what is left for you to check. The automated tests (`tests/test_frontend_views.py`,
`tests/test_ui_pages.py`) render every page with Streamlit's headless runner and check the facts on the page, but they do not look
at layout.

## Checked by the author of this document, in a Linux sandbox

A headless Chromium (Playwright) loaded the running application (`python run.py`) with two tiny stand-in models and a stand-in server
that speaks Ollama's protocol, so no real model produced any text:

| Page | Result |
|---|---|
| Navigation | the sidebar lists Cases and documents, Ask, Dashboard, Evaluation, Settings; the status block shows API, Ollama and OCR |
| Dashboard | five counters, the "no questions yet" hint, the system block (API, Ollama and its version, model, OCR, embeddings, reranker) |
| Evaluation | the three tabs; retrieval and answer tables filled from `evaluation/results/` (screenshot inspected) |
| Settings | the setting groups with their environment variables (screenshot inspected) |
