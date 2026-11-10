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
- [x] SQLite store, background indexing jobs with progress, FastAPI backend (upload validation, streaming answers,
      evidence page images), Streamlit UI, one-command start (`python run.py`, Makefile)
- [ ] Docker: `Dockerfile` and `docker-compose.yml` are written but NOT yet built or tested
- [x] Evaluation: validated held-out and dev question sets, retrieval and answer metrics, cached resumable runs,
      human grading, threshold calibration on the dev set, sample-document downloader
- [x] Architecture diagram ([docs/architecture.md](docs/architecture.md))
- [x] Presentation outline, live-demo script and dry-run log ([docs/](docs/))

This list is updated as each part is actually committed.

## Architecture

![Architecture overview](docs/overview.png)

The browser talks to a Streamlit UI, the UI to a FastAPI backend, and the backend to local models only. Details, the indexing
diagram, the question-answering diagram and the reasons behind each design decision are in
[docs/architecture.md](docs/architecture.md).

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

## Run the application

```bash
ollama pull qwen3.5:9b          # once; Ollama must be running (ollama serve or the tray app)
python run.py                   # starts the API and the UI and opens http://127.0.0.1:8501
```
or `make run`. `python main.py` starts only the API (interactive docs at http://127.0.0.1:8000/docs) and
`streamlit run app.py` only the UI. Change ports with `API_PORT` / `UI_PORT`.

1. **Cases and documents**: create a case, drag in the PDFs. A progress bar follows extraction, OCR, chunking and
   embedding; each document shows its pages, chunks, tables and scanned pages, and its pages can be previewed.
2. **Ask**: choose search mode, reranking, number of passages, or switch retrieval off for the baseline. Every claim
   shows its citations with the six verification checks; a toggle shows the cited page with the quote highlighted
   (yellow) and the clause region outlined (blue).

| Endpoint | Purpose |
|---|---|
| `GET /api/status`, `/api/stats` | Ollama / OCR availability, question log |
| `POST /api/cases`, `GET/DELETE /api/cases/{id}` | cases |
| `POST /api/cases/{id}/documents` | upload (validated: `.pdf` name, `%PDF` header, size, encryption, duplicates) |
| `GET /api/jobs/{id}` | indexing progress |
| `GET /api/documents/{id}/info`, `/pages/{n}`, `/pages/{n}/image` | explorer and evidence viewer (images only, never the raw PDF) |
| `POST /api/cases/{id}/ask`, `/ask/stream` | answer (complete JSON, or newline-delimited events) |

Docker (`docker compose up --build`) is written but has not been built or run yet.

## Project layout

The project folder holds the modules directly (`config.py`, `api/`, `services/`, `retrieval/`, ... next to `app.py`
and `main.py`) and every module imports the others by plain name (`from config import get_settings`). Run everything from
the project folder (`python run.py`, `python -m scripts.<name>`, `python -m evaluation.<name>`, `python -m pytest`); the folder
can have any name. `storage/` (uploaded files, indexes, database) and `.env` live inside the project folder.

## Evaluation

```bash
python -m scripts.download_sample_docs                         # GDPR, ePrivacy Directive, Data Act -> sample_docs/
python -m evaluation.calibrate sample_docs/*.pdf --write-env     # thresholds from the DEV set -> .env
python -m evaluation.evaluate sample_docs/*.pdf --retrieval-only # about 1 minute, no LLM
python -m evaluation.evaluate sample_docs/*.pdf --gen-configs no_retrieval hybrid --export-grading
```

* **Provenance.** Both question sets were written for the earlier private prototype of this project and are reused here unchanged; the labels are re-checked against the PDFs by the tests below.
* **Held-out set** `evaluation/dataset.json`: 15 questions (6 factual, 5 multi-hop, 4 unanswerable), each with an
  expected answer, key facts and the labelled evidence (document, page, clause, a snippet). `tests/test_dataset.py` proves
  every snippet is on its labelled page. **Dev set** `evaluation/calibration_set.json`: 14 different questions used ONLY by
  `evaluation.calibrate`; a test checks that the two sets do not overlap.
* **Mandatory comparisons**: dense vs hybrid retrieval (`recall@k`, `full@k`, `precision@5`, MRR, latency) and
  no-retrieval vs retrieval-enabled answers (key-fact correctness, citations verified, citation precision, abstention
  accuracy, false-abstention rate, latency, tokens). Fixed seed 42.
* LLM answers take about 80 s each on a CPU. Each one is cached in `evaluation/results/gen_cache.jsonl` (not committed)
  as soon as it exists, so an interrupted run resumes; changing any setting that affects answers invalidates the cache.
* **Human grading.** The "answer correct" number from the program is an automatic key-fact PROXY: it misses paraphrases and
  cannot judge a refusal. `--export-grading` writes `evaluation/results/manual_grades.csv`; read each answer, enter 1 / 0.5 / 0
  (scale in the file `evaluation/grading.py`), then run the same command with `--apply-grades <file> --grader "<your name>"`
  (answers come from the cache, the model is not asked again). The report names who graded and flags incomplete grading.
* A failed answer is never cached; cached answers that used retrieval are tied to the exact index, so indexing again makes them
  unusable (the no-retrieval baseline is unaffected). A manual grade is kept only while the answer it was given to is unchanged.
  If answerable questions are refused after the model answered, the report prints a WARNING: read `refusal_stage`, `refusal_reason` and
  `verifier_note` in `generation_by_question.csv` before trusting the numbers.
* Outputs in `evaluation/results/`: `summary.md` (tables for this README), `summary.json`, `retrieval_by_question.csv`,
  `generation_by_question.csv`, `meta.json` (models, library versions, thresholds, document hashes, git commit),
  `calibration.json`.

### Results

Run 2026-11-08 22:31:38 UTC - model `qwen3.5:9b`, seed 42, 15 held-out questions (11 answerable).

#### Retrieval (answerable questions)

| Configuration | recall@1 | recall@3 | recall@5 | full@5 | precision@5 | mrr | latency_ms |
|---|---|---|---|---|---|---|---|
| BM25 only | 0.55 | 0.73 | 0.85 | 0.73 | 0.27 | 0.75 | 22 |
| Dense only | 0.36 | 0.55 | 0.59 | 0.45 | 0.18 | 0.51 | 816 |
| Hybrid (BM25 + dense, RRF) | 0.59 | 0.64 | 0.88 | 0.73 | 0.27 | 0.82 | 16 |
| Hybrid + rerank | 0.59 | 0.76 | 0.83 | 0.73 | 0.27 | 0.76 | 481 |
| Hybrid + rerank + query decomposition | 0.59 | 0.85 | 0.89 | 0.82 | 0.31 | 0.82 | 1131 |

`recall@k` = share of the required evidence items (document + clause) in the top k; `full@k` = share of questions whose evidence is ALL in the top k (strict for multi-hop).

| Category / configuration | recall@1 | recall@5 | full@5 |
|---|---|---|---|
| factual / BM25 only | 0.83 | 1.00 | 1.00 |
| multihop / BM25 only | 0.20 | 0.67 | 0.40 |
| factual / Dense only | 0.67 | 0.83 | 0.83 |
| multihop / Dense only | 0.00 | 0.30 | 0.00 |
| factual / Hybrid (BM25 + dense, RRF) | 0.83 | 1.00 | 1.00 |
| multihop / Hybrid (BM25 + dense, RRF) | 0.30 | 0.73 | 0.40 |
| factual / Hybrid + rerank | 1.00 | 1.00 | 1.00 |
| multihop / Hybrid + rerank | 0.10 | 0.63 | 0.40 |
| factual / Hybrid + rerank + query decomposition | 1.00 | 1.00 | 1.00 |
| multihop / Hybrid + rerank + query decomposition | 0.10 | 0.77 | 0.60 |

#### Generation

| | no_retrieval | hybrid |
|---|---|---|
| Answer correct, automatic key-fact PROXY (answerable) | 0.55 | 0.73 |
| Citations verified by the validator | - | 0.96 |
| Citation precision vs labelled evidence | - | 0.75 |
| Abstention accuracy (all questions) | 0.73 | 0.93 |
| False-abstention rate (answerable refused) | 0.00 | 0.09 |
| Correct-abstention rate (unanswerable refused) | 0.00 | 1.00 |
| Mean retrieval + rerank time (s) | 0.00 | 0.95 |
| Mean generation time (s) | 16.05 | 69.04 |
| Mean total time (s) | 16.10 | 70.05 |
| Mean completion tokens | 95 | 447 |
| Answers that failed with an error | 0 | 0 |

The correctness row is an AUTOMATIC proxy (every key fact occurs in the answer); it misses paraphrases. It is not a human judgement. Grade the answers yourself with `--export-grading` / `--apply-grades`.

#### Manual grades

Graded by: **Sravan** (file `manual_grades.csv`). Scale: answerable: 1 correct and complete, 0.5 partly correct, 0 wrong/ungrounded/unjustified refusal; unanswerable: 1 correctly says the documents do not contain it, 0 answers from outside knowledge.

| | no_retrieval | hybrid |
|---|---|---|
| Answers graded | 15 / 15 | 15 / 15 |
| Answerable: mean score (1 / 0.5 / 0) | 0.73 | 0.91 |
| Answerable: strictly correct (grade 1) | 0.55 | 0.91 |
| Unanswerable: handled correctly | 0.50 | 1.00 |
| All questions: behaviour correct | 0.53 | 0.93 |

#### Reading the numbers

* **Dense-only retrieval is the weakest** (recall@5 0.59, against 0.85 for BM25 alone). The documents are full of clause numbers
  and defined terms, which BM25 matches exactly. Hybrid is best on recall@5 (0.88) and MRR (0.82), but its gap to BM25 alone is
  inside the noise of 11 answerable questions.
* **Reranking is not a clear win on this set.** It raises recall@3 (0.64 to 0.76) but lowers recall@5 (0.88 to 0.83) and MRR
  (0.82 to 0.76), and costs about 30 times the latency (16 ms to 481 ms). It is kept because the sufficiency gate and the
  confidence use its score.
* **Query decomposition** raises full@5 from 0.73 to 0.82 and multi-hop full@5 from 0.40 to 0.60. It was designed after looking
  at multi-hop failures, so this gain is optimistic.
* **Retrieval changes the answers more than it changes the ranking.** Graded by hand, answerable questions score 0.91 with retrieval and
  0.73 without; unanswerable questions are handled correctly 1.00 against 0.50. The price is time: about 70 s per question against 16 s.
* **Citations.** 0.96 of citations pass the validator. Citation precision against the labelled evidence is 0.75: a quarter of the
  cited passages are not among the labelled evidence items. Whether those are valid alternatives was not examined.
* **One false abstention in 11 answerable questions** (hybrid, question M1, the maximum GDPR fine): the sufficiency gate refused it
  because the best rerank score was 0.66, below the threshold 2.5.

### Limits of the evaluation

* 15 questions: one question moves a rate by 6-9 points, so differences below about 10 points are noise.
* The dev set has 14 questions, so the calibrated thresholds are approximate.
* The automatic correctness proxy is not a human judgement; report the manual grades, and say who graded.
* Query decomposition was added after looking at multi-hop failures of a similar prototype; its gain on this set is
  therefore optimistic.
* The citation validator checks lexical and numeric support, not logical entailment.
* The thresholds of the sufficiency gate were calibrated on EU legal texts. Other kinds of documents are checked in
  [docs/dry_runs.md](docs/dry_runs.md), not assumed.
