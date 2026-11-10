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

## Not checked with the real models: your browser pass

Run `python run.py` (or `docker compose up`) with Ollama and the real models, and tick each item.

| # | Check | Result |
|---|---|---|
| 1 | Create a case, upload 2-3 PDFs: the progress bar moves and every document ends "indexed" | |
| 2 | A document's page table shows characters, how each page was read, tables, and the unusual-page column | |
| 3 | Preview a page: the page image and its extracted text appear side by side | |
| 4 | Upload a scanned PDF: its pages are flagged and recovered with OCR (needs Tesseract) | |
| 5 | Ask a factual question: confidence badge, claims, citations with the six checks, the cited page with the quote highlighted | |
| 6 | Ask a multi-hop question: citations on two different pages | |
| 7 | Ask a question the documents cannot answer: INSUFFICIENT EVIDENCE with the reason and the documents searched | |
| 8 | Switch retrieval off: the answer is marked ungrounded | |
| 9 | Dashboard counters and "recent questions" change after asking | |
| 10 | Evaluation page shows every table (run `make extended` and `make report` first) | |
| 11 | Settings page matches the environment | |
| 12 | Stop Ollama: the sidebar shows "Ollama offline" and Ask explains it instead of failing | |
