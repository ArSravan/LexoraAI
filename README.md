# Lexora AI

A local-first, evidence-grounded analyst for legal PDFs: upload documents, ask questions,
and get answers whose citations (document, page, clause, quote) are independently verified
against the source - or an honest "INSUFFICIENT EVIDENCE".

**Status: work in progress.** This repository was started on 2026-11-04.

## What exists so far

- [x] Project skeleton, pinned dependencies
- [x] Settings (defaults + `LEXORA_*` environment variables)
- [x] Data models for parsed documents (page index + bounding box on every line)
- [x] PDF validation and text extraction (PyMuPDF): page numbers, bounding boxes, row merging, image-only page flag
- [x] Reading order for two columns, header/footer removal, footnote detection
- [x] OCR for scanned pages (Tesseract)
- [x] Table extraction (pdfplumber)
- [x] Legal structure detection and clause-aware chunking
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

OCR needs the Tesseract program (the tests skip the OCR check if it is missing).
Windows: `winget install --id UB-Mannheim.TesseractOCR -e`. Set `LEXORA_TESSERACT_CMD` if it is
not in the default location (see `.env.example`).

## Try it on a PDF

```bash
python -m scripts.inspect_pdf path/to/file.pdf            # pages, scans, structure, chunk list
python -m scripts.inspect_pdf path/to/file.pdf --clause "83(5)"
```
