# Extended features

The assignment lists eleven optional features. This page says, for each one, whether it is implemented, what shows that it works,
how to reproduce the evidence, and where it stops. The measured numbers are not repeated here: they are generated into
[RESULTS.md](RESULTS.md) from `evaluation/results/`, so this page cannot disagree with them.

`make extended` runs every evaluation below that needs no language model (a few minutes); `make eval` runs the answer evaluation
(needs Ollama).

| # | Feature | Status |
|---|---|---|
| 1 | OCR for scanned pages | implemented, evaluated |
| 2 | Table extraction | implemented, evaluated |
| 3 | Chart and figure reading with a vision model | **not implemented** |
| 4 | Bounding-box citations on the page image | implemented, evaluated at region level |
| 5 | Planner + verifier agent loop | implemented |
| 6 | Fine-tuned reranker or embedder | **not implemented** |
| 7 | Ablation table | implemented for the verifier and for tables; no figures |
| 8 | Autoencoder component | implemented, evaluated against a baseline |
| 9 | Multi-document cross-referencing | implemented, covered by tests |
| 10 | Public deployment | **not implemented** |
| 11 | Latency and cost report | implemented, two optimisations measured |

## 1. OCR for scanned pages

**Built.** `parse_pdf` finds pages with (almost) no text layer and an image, flags them (`needs_ocr`), and recovers them with
Tesseract (`ingestion/ocr.py`), keeping a bounding box and a confidence for every line. Without Tesseract the page stays flagged
("Image-only page: no text layer, OCR needed"), the document gets a warning, and a PDF that is only scans is rejected with a clear
message instead of being indexed empty.
**Evidence.** `python -m evaluation.ocr_eval sample_docs/*.pdf`: real pages of the three sample documents are rasterised, degraded
in four fixed ways (clean, noise, skew + blur, low contrast + JPEG), read by the same code path as an upload, and compared with the
digital text of the same page (character error rate, word error rate, word recall). The same script checks the fallback.
**Limits.** The ground truth is the PDF's own text layer, so the test measures OCR fidelity on clean typography, not on handwriting
or on real scanner artefacts. Past a point Tesseract returns nothing at all; such a page is reported as unreadable, not guessed.

## 2. Table extraction

**Built.** Tables drawn with ruling lines are read with pdfplumber, their caption ("Table 9.1 ...") is attached, and each cell is
indexed together with its column header (`ingestion/table_extractor.py`, `schemas.TableData.to_text`).
**Evidence.** `python -m evaluation.table_eval sample_docs/*.pdf`: eight kinds of table with a hand-written ground truth (plain grid, a
header spanning columns, a label spanning rows, wrapped multi-line cells, two tables on a page, a long table, and two layouts it is
not expected to read), scored cell by cell. The same script counts the tables found in the real sample documents.
**Limits.** The ground truth is synthetic, so the scores for the cases it can read are an upper bound for real documents. Tables
without full ruling lines are not found; their text stays in the page text. The three sample documents contain no ruled tables, so
there is no accuracy figure for a real document.

## 3. Figures read by a vision model: not implemented

No vision model is used, so a question whose answer exists only in a figure is answered with "insufficient evidence" or from the text
around the figure. There is therefore no hallucination rate for figures to report.

## 4. Bounding-box citations

**Built.** Every extracted line keeps its page index and bounding box; every chunk keeps the regions it came from. The evidence
viewer renders the cited page and draws the quote in yellow and the clause region in blue (`services/render.py`); the PDF itself is
never served.
**Evidence.** `evaluation/region_eval.py`, run as part of `python -m evaluation.evaluate`: for each emitted citation it checks whether
the quote can be located on the cited page, whether the page is a labelled evidence page, and whether the quote lies inside the
labelled clause region (at least half of its rectangle overlaps it). Page-level and region-level precision are reported side by
side; the gap is the share of citations that are on the right page but in the wrong place.
**Limits.** The region of a clause is taken from the chunks of the labelled clause, so a correct citation of a neighbouring clause
counts as a miss. Cached answers from before this feature carry no citation detail; `--refresh-citations` asks those questions again.

## 5. Planner + verifier loop

**Built.** The model drafts; the validator checks every citation; if the draft is rejected, the model is told which claims failed and
why and may rewrite once (`verify_retries`). The retry log is part of every answer and is shown in the UI ("the verifier rejected the
first draft").
**Evidence.** `tests/test_answer_pipeline.py` (a fabricated draft is rejected, the rewrite is accepted or the question is refused)
and the ablation below.
**Limits.** One retry, no planner that decomposes the task: the "planner" role is the query decomposition of the retrieval step.

## 6. Fine-tuned reranker or embedder: not implemented

The off-the-shelf MiniLM embedder and `ms-marco-MiniLM` cross-encoder are used. The one false abstention of the held-out set is a
scoring problem of that cross-encoder (see [EVALUATION_ANALYSIS.md](EVALUATION_ANALYSIS.md)); fine-tuning it on legal passages would be the
next step, but there is no training setup or gain to report.

## 7. Ablation

* **With and without the verifier.** The configuration `hybrid_noverify` of `evaluation.evaluate` shows the first draft unchecked (no
  retry, no refusal when no citation verifies). The citations are still checked so that their status can be reported. The table
  compares both on correctness, claims shown that are not verified, false abstentions, time and tokens. It is measured
  automatically and is not part of the hand-graded sheet.
* **Text only against text + tables.** `python -m evaluation.ablation_tables` indexes one generated contract twice, with and without
  table extraction, and asks six questions that can only be answered from table cells. A chunk counts as the evidence when it
  contains every string of the answer, so both indexes are judged by the same test. BM25 and hybrid + rerank are both measured.
* **Figures.** There is no "+figures" row because figures are not read.

**Limits.** Six questions on one generated document show a direction, not a significant difference.

## 8. Autoencoder: page-anomaly detector

**Built.** A small autoencoder (14 features -> 6 -> 2 -> 6 -> 14) is fitted, at indexing time, on the pages of the document itself and
flags the pages it cannot reconstruct: a broken text layer, OCR junk, a page that is only a number, a spreadsheet dump, a repeated
line. The Cases page marks them "unusual page". The threshold is median + K x MAD of the document's own reconstruction errors, and K
is chosen from CLEAN documents only (so no defective page influences it). Nothing is trained in advance and no weights are shipped.
**Evidence.** `python -m evaluation.page_anomaly_eval sample_docs/*.pdf`: defects are injected into 10 % of a real document's pages
(six kinds, three seeds; "bad scan" is the real page degraded and read by the real Tesseract), the detector is fitted on the contaminated
document, and AUROC, detection and false-alarm rates are reported next to a plain z-score baseline. The same script measures the
alternative that was rejected: a model fitted on other documents, which flags many intact pages of a differently laid-out document.
**Limits.** The defects are synthetic. The design (bottleneck size, spread floor) was chosen after looking at them, so the detection
rates are optimistic. The results table shows whether the autoencoder beats the baseline; it is reported, not assumed. Documents of
fewer than 12 pages are not scored. A flag means "look at this page", not "this page is wrong".

## 9. Multi-document cross-referencing

**Built.** Passages from different documents are compared: figures with the same topic are reconciled or flagged as a conflict
(`verification/conflict_detector.py`), a numeric question ("does the cap in Section 7 exceed the indemnities in Section 9?") is answered by
arithmetic on the cited figures and the comparison is shown, and a conflict reported by the model is kept only if it cites retrieved
evidence.
**Evidence.** `tests/test_conflicts.py` and `tests/test_answer_pipeline.py` (a restated figure is classified as an explained difference,
two contradicting figures as a conflict, a comparison is computed from the cited numbers).
**Limits.** Heuristic: topics are matched by word overlap, and conflicts are shown, never resolved. There is no benchmark of
conflict detection in this repository.

## 10. Public deployment: not implemented

The application needs a local Ollama and two local models, which hosted free tiers (Hugging Face Spaces, Streamlit Cloud) do not provide.
`docker compose up --build` is the supported way to run it elsewhere ([DOCKER_VERIFICATION.md](DOCKER_VERIFICATION.md)).

## 11. Latency and cost

**Built.** Every answer carries its timings (retrieval, rerank, generation, total) and token counts; the evaluation records them per question.
**Evidence.** `python -m evaluation.latency_report` (per configuration: mean, median, 95th percentile, tokens per second) and two measured
optimisations:
* **The sufficiency gate** refuses an unanswerable question before the model is called. The report measures how long those questions
  took and estimates the model time they did not use.
* **The embedding cache**: `python -m evaluation.bench_indexing <pdf>` embeds the same chunks cold and warm and indexes the document
  again.

**Limits.** The saving of the gate is an estimate (the model was not called for those questions). The model runs locally, so there
is no per-token fee; cost is reported as seconds and tokens, and no cloud price is invented.
