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
- [x] Retrieval: cached local embeddings (MiniLM), ChromaDB, BM25 with a clause-number-aware tokenizer,
      Reciprocal Rank Fusion, cross-encoder reranking, query decomposition
- [x] Local LLM answers (Ollama) as structured JSON, independent citation validation (six checks),
      evidence-sufficiency gate with abstention, verifier retry loop, computed confidence,
      numeric comparison and cross-document conflict detection
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

CPU-only PyTorch is much smaller than the default build; install it first:

```bash
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The two small Hugging Face models (embeddings and reranker, about 90 MB each) download on first use and are
cached; after that the project works offline. The two tests in `tests/test_real_models.py` use them and are
skipped when they cannot be loaded.

OCR needs the Tesseract program (the tests skip the OCR check if it is missing).
Windows: `winget install --id UB-Mannheim.TesseractOCR -e`. Set `LEXORA_TESSERACT_CMD` if it is
not in the default location (see `.env.example`).

## Try it on a PDF

```bash
python -m scripts.inspect_pdf path/to/file.pdf            # pages, scans, structure, chunk list
python -m scripts.inspect_pdf path/to/file.pdf --clause "83(5)"
```

```bash
python -m scripts.search_pdf a.pdf b.pdf -q "What is the maximum fine?" --compare
```
`--compare` shows BM25 only, dense only, hybrid, hybrid + rerank and hybrid + rerank + decomposition side by side.

To measure the retrieval configurations on a labelled question set (answerable questions only):

```bash
python -m scripts.eval_retrieval a.pdf b.pdf --dataset path/to/dataset.json --misses
```

## Ask a question (needs Ollama)

Install [Ollama](https://ollama.com/download) and pull the model once:

```bash
ollama pull qwen3.5:9b          # any other model works: set LEXORA_OLLAMA_MODEL or use --model
ollama list                     # verify
python -m scripts.ask_pdf contract.pdf report.pdf -q "Does the liability cap in Section 7 exceed Section 9?"
python -m scripts.ask_pdf contract.pdf -q "..." --no-retrieval       # the same model without documents
```

How an answer is produced and checked:

1. **Retrieve** the best passages (BM25 + dense + RRF + cross-encoder rerank, sub-queries for compound questions).
2. **Gate**: if the best relevance score or the question-term coverage is too low, answer
   `INSUFFICIENT EVIDENCE` without calling the model (the model never fills gaps from general knowledge).
3. **Generate** a JSON answer from the numbered passages `[E1] ...` (document text is untrusted data).
4. **Verify** every citation independently: document exists, page exists, clause exists, the quote occurs in the
   cited passage, the passage was retrieved for this question, and the passage supports the claim's numbers and
   key terms. If the draft fails, the model is told why and may rewrite it once.
5. **Confidence** (HIGH / MEDIUM / LOW) is computed from the verification results, not taken from the model.

Known limits: the validator checks lexical and numeric support, not logical entailment; an answer with some
unverified claims is still shown (flagged, with lower confidence); conflict detection is heuristic.