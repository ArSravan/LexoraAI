# Lexora AI

A local-first, evidence-grounded analyst for legal PDFs: upload documents, ask questions,
and get answers whose citations (document, page, clause, quote) are independently verified
against the source - or an honest "INSUFFICIENT EVIDENCE".

**Status: work in progress.** This repository was started on 2026-11-04. 

## What exists so far

- [x] Project skeleton, pinned dependencies
- [x] Settings
- [ ] Data models
- [ ] PDF validation and text extraction (PyMuPDF)
- [ ] OCR for scanned pages (Tesseract), tables, legal structure detection, chunking
- [ ] Retrieval: embeddings, ChromaDB, BM25, RRF, cross-encoder reranking
- [ ] Local LLM answers (Ollama), citation validation, abstention
- [ ] API, UI, one-command run, Docker
- [ ] Evaluation and tests

This list is updated as each part is actually committed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```