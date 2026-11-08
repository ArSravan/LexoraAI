# Architecture

Everything runs on one machine: the browser talks to a Streamlit UI, the UI talks to a FastAPI backend, and the backend
uses local models (embeddings, reranker, Ollama). No document or question leaves the machine.

Pictures: [overview.png](overview.png), [indexing.png](indexing.png), [question.png](question.png) (rendered from the
`.mmd` sources next to them; the diagrams below are the same text and are drawn by GitHub).

## 1. Overview

```mermaid
flowchart TB
    user(["User"]) --> ui["<b>Streamlit UI</b>  frontend/<br/>Cases and documents  |  Ask"]
    ui -- "HTTP" --> api["<b>FastAPI</b>  api/<br/>upload validation  |  ask and ask/stream (NDJSON)  |  evidence page images"]
    api --> indexing["<b>Indexing pipeline</b>  ingestion/  services/<br/>extract, OCR, tables, structure, chunks, embeddings"]
    api --> qa["<b>Question pipeline</b>  retrieval/  generation/  verification/<br/>retrieve, gate, generate, verify"]
    indexing --> store[("<b>Local storage</b>  storage/<br/>ChromaDB vectors  |  chunks.json for BM25<br/>SQLite: cases, documents, qa_log, embedding cache<br/>uploaded PDFs + meta.json")]
    qa <--> store
    qa <--> llm["<b>Ollama</b> qwen3.5:9b<br/>runs locally, nothing leaves the machine"]
```

## 2. Indexing: from a PDF to searchable, citable chunks

```mermaid
flowchart TB
    pdf["PDF upload"] --> val["Validate<br/>.pdf name, %PDF header, size,<br/>not encrypted, not a duplicate"]
    val --> ext["Extract text - PyMuPDF<br/>every line keeps page_index and bbox"]
    ext --> clean["Clean<br/>two-column reading order,<br/>header and footer removal, footnotes"]
    clean --> ocrq{"image-only<br/>page?"}
    ocrq -- yes --> ocr["OCR - Tesseract"]
    ocrq -- no --> tab
    ocr --> tab["Tables - pdfplumber"]
    tab --> struct["Legal structure detection<br/>chapter, article, clause, point<br/>state machine with sequence checks"]
    struct --> chunk["Clause-aware chunking<br/>parent and child chunks,<br/>lead-in text, index_text with clause path"]
    chunk --> emb["Embeddings - MiniLM<br/>cached in SQLite by text hash"]
    emb --> chroma[("ChromaDB")]
    chunk --> cj[("chunks.json")]
    ext --> meta[("meta.json<br/>pages, clauses, hash")]
```

Every line keeps its page index and bounding box from the first step on. That is what lets an answer cite a page and
clause and lets the UI draw the quote on the page image.

## 3. Answering: retrieve, gate, generate, verify

```mermaid
flowchart TB
    q(["Question"]) --> dec["Query decomposition<br/>compound question to sub-queries"]
    dec --> bm["BM25<br/>clause-number-aware"]
    dec --> de["Dense search<br/>MiniLM + ChromaDB"]
    bm --> rrf["RRF fusion, k = 60"]
    de --> rrf
    rrf --> rr["Cross-encoder rerank<br/>sub-query results interleaved"]
    rr --> gate{"Sufficiency gate<br/>rerank score >= 2.5 and<br/>term coverage >= 0.32"}
    gate -- no --> abst(["INSUFFICIENT EVIDENCE<br/>reason shown<br/>from the gate: no LLM call"])
    gate -- yes --> llm["Ollama qwen3.5:9b<br/>numbered passages as untrusted data<br/>JSON answer: claims with quotes"]
    llm --> val["Pydantic check, then six checks per citation<br/>document, page, clause, quote,<br/>was retrieved, supports the claim"]
    val --> okq{"at least one<br/>verified claim?"}
    okq -- "no, first try" --> retry["send the rejection reasons<br/>one retry"]
    retry --> llm
    okq -- "no, after retry" --> abst
    okq -- yes --> post["Conflict detection, numeric comparison<br/>Confidence computed from the checks"]
    post --> out(["Answer with verified citations<br/>and highlighted evidence page"])
```

## Where each step lives

| Step | Module | Notes |
|---|---|---|
| Upload checks | `api/routes.py`, `ingestion/pdf_parser.py` | name, `%PDF` header, size, encryption, duplicates |
| Text extraction, reading order, headers and footers | `ingestion/pdf_parser.py`, `ingestion/cleaning.py` | PyMuPDF; two columns; footnotes |
| OCR, tables | `ingestion/ocr.py`, `ingestion/table_extractor.py` | Tesseract for image-only pages; pdfplumber |
| Legal structure, chunking | `ingestion/structure_detector.py`, `ingestion/chunker.py` | article / clause / point path stored on every chunk |
| Embeddings, vector store | `retrieval/embeddings.py`, `retrieval/vector_store.py` | MiniLM, SQLite cache, ChromaDB (cosine) |
| BM25, fusion, rerank, decomposition | `retrieval/bm25.py`, `retrieval/hybrid.py`, `retrieval/reranker.py` | RRF k = 60; cross-encoder rerank |
| Sufficiency gate | `verification/evidence_checker.py`, `verification/abstention.py` | thresholds in `config.py` / `.env` |
| Generation | `generation/answer_generator.py`, `ollama_client.py`, `prompts.py` | JSON output, one verifier retry |
| Citation validation | `verification/citation_validator.py` | six checks per citation |
| Conflicts, numeric comparison | `verification/conflict_detector.py` | heuristic |
| Background jobs, storage | `services/jobs.py`, `services/indexer.py`, `services/db.py` | progress bar in the UI |
| Evaluation | `evaluation/` | held-out set, dev set, metrics, grading, calibration |

## Design decisions and why

| Decision | Reason |
|---|---|
| BM25 and dense retrieval, fused with RRF | measured: dense alone is the weakest configuration on legal text (see README, Results) |
| Chunks follow the document's own structure | an answer must cite a clause, not a 500-character window |
| A gate before the LLM | an unanswerable question is refused in about a second instead of after 70 s, and the model never fills gaps from general knowledge |
| The model must quote, and each quote is checked in code | a wrong or invented citation is caught by the six checks instead of being trusted |
| One retry with the rejection reasons | gives the model a chance to fix a bad citation without looping |
| Confidence computed from the checks | the model's own confidence is not evidence |
| Passages marked as untrusted data | text inside a PDF must not be able to give instructions to the model |
| Everything local | legal documents are confidential; also works offline |
