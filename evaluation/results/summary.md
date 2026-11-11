Run 2026-11-11 17:59:35 UTC - model `qwen3.5:9b`, seed 42, 15 held-out questions (11 answerable).

### Retrieval (answerable questions)

| Configuration | recall@1 | recall@3 | recall@5 | full@5 | precision@5 | mrr | latency_ms |
|---|---|---|---|---|---|---|---|
| BM25 only | 0.55 | 0.73 | 0.85 | 0.73 | 0.27 | 0.75 | 21 |
| Dense only | 0.36 | 0.50 | 0.59 | 0.45 | 0.18 | 0.50 | 1318 |
| Hybrid (BM25 + dense, RRF) | 0.55 | 0.59 | 0.83 | 0.64 | 0.25 | 0.73 | 16 |
| Hybrid + rerank | 0.64 | 0.80 | 0.83 | 0.73 | 0.27 | 0.80 | 1073 |
| Hybrid + rerank + query decomposition | 0.64 | 0.85 | 0.89 | 0.82 | 0.31 | 0.86 | 4084 |

`recall@k` = share of the required evidence items (document + clause) in the top k; `full@k` = share of questions whose evidence is ALL in the top k (strict for multi-hop).

| Category / configuration | recall@1 | recall@5 | full@5 |
|---|---|---|---|
| factual / BM25 only | 0.83 | 1.00 | 1.00 |
| multihop / BM25 only | 0.20 | 0.67 | 0.40 |
| factual / Dense only | 0.67 | 0.83 | 0.83 |
| multihop / Dense only | 0.00 | 0.30 | 0.00 |
| factual / Hybrid (BM25 + dense, RRF) | 0.83 | 1.00 | 1.00 |
| multihop / Hybrid (BM25 + dense, RRF) | 0.20 | 0.63 | 0.20 |
| factual / Hybrid + rerank | 1.00 | 1.00 | 1.00 |
| multihop / Hybrid + rerank | 0.20 | 0.63 | 0.40 |
| factual / Hybrid + rerank + query decomposition | 1.00 | 1.00 | 1.00 |
| multihop / Hybrid + rerank + query decomposition | 0.20 | 0.77 | 0.60 |

### Generation

| | no_retrieval | hybrid | hybrid_noverify |
|---|---|---|---|
| Answer correct, automatic key-fact PROXY (answerable) | 0.55 | 0.82 | 0.82 |
| Citations verified by the validator | - | 0.97 | 0.97 |
| Citation precision vs labelled evidence | - | 0.74 | 0.74 |
| Claims shown that the validator does not verify | - | 0.03 | 0.03 |
| Answers shown with at least one such claim | - | 0.09 | 0.09 |
| Abstention accuracy (all questions) | 0.73 | 1.00 | 1.00 |
| False-abstention rate (answerable refused) | 0.00 | 0.00 | 0.00 |
| Correct-abstention rate (unanswerable refused) | 0.00 | 1.00 | 1.00 |
| Mean retrieval + rerank time (s) | 0.00 | 3.63 | 3.41 |
| Mean generation time (s) | 16.08 | 73.89 | 69.18 |
| Mean total time (s) | 16.15 | 77.58 | 72.66 |
| Mean completion tokens | 95 | 482 | 453 |
| Answers that failed with an error | 0 | 0 | 0 |

The correctness row is an AUTOMATIC proxy (every key fact occurs in the answer); it misses paraphrases. It is not a human judgement. Grade the answers yourself with `--export-grading` / `--apply-grades`.

### Citation precision at region level

Configuration `hybrid`, 34 emitted citations. A citation is a REGION hit when the quote it points at lies inside the labelled clause on the page (see `evaluation/region_eval.py`).

| Measure | Value |
|---|---|
| Quote found on the cited page (can be drawn) | 0.97 |
| Page-level precision | 0.85 |
| Region-level precision | 0.74 |
| Region-level precision, of the located quotes | 0.76 |
| Citations on the right page but outside the labelled region | 4 |

### Manual grades

Graded by: **Sravan** (file `manual_grades.csv`). Scale: answerable: 1 correct and complete, 0.5 partly correct, 0 wrong/ungrounded/unjustified refusal; unanswerable: 1 correctly says the documents do not contain it, 0 answers from outside knowledge.

| | no_retrieval | hybrid |
|---|---|---|
| Answers graded | 15 / 15 | 15 / 15 |
| Answerable: mean score (1 / 0.5 / 0) | 0.73 | 0.95 |
| Answerable: strictly correct (grade 1) | 0.55 | 0.91 |
| Unanswerable: handled correctly | 0.50 | 1.00 |
| All questions: behaviour correct | 0.53 | 0.93 |
