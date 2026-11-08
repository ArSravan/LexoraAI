Run 2026-11-08 20:44:50 UTC - model `qwen3.5:9b`, seed 42, 15 held-out questions (11 answerable).

### Retrieval (answerable questions)

| Configuration | recall@1 | recall@3 | recall@5 | full@5 | precision@5 | mrr | latency_ms |
|---|---|---|---|---|---|---|---|
| BM25 only | 0.55 | 0.73 | 0.85 | 0.73 | 0.27 | 0.75 | 20 |
| Dense only | 0.36 | 0.55 | 0.59 | 0.45 | 0.18 | 0.51 | 1019 |
| Hybrid (BM25 + dense, RRF) | 0.59 | 0.64 | 0.88 | 0.73 | 0.27 | 0.82 | 18 |
| Hybrid + rerank | 0.59 | 0.76 | 0.83 | 0.73 | 0.27 | 0.76 | 459 |
| Hybrid + rerank + query decomposition | 0.59 | 0.85 | 0.89 | 0.82 | 0.31 | 0.82 | 1166 |

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

### Generation

| | no_retrieval | hybrid |
|---|---|---|
| Answer correct, automatic key-fact PROXY (answerable) | 0.55 | 0.00 |
| Citations verified by the validator | - | - |
| Citation precision vs labelled evidence | - | - |
| Abstention accuracy (all questions) | 0.73 | 0.27 |
| False-abstention rate (answerable refused) | 0.00 | 1.00 |
| Correct-abstention rate (unanswerable refused) | 0.00 | 1.00 |
| Mean retrieval + rerank time (s) | 0.00 | 0.97 |
| Mean generation time (s) | 16.05 | 128.79 |
| Mean total time (s) | 16.10 | 129.81 |
| Mean completion tokens | 95 | 838 |
| Answers that failed with an error | 0 | 0 |

The correctness row is an AUTOMATIC proxy (every key fact occurs in the answer); it misses paraphrases. It is not a human judgement. Grade the answers yourself with `--export-grading` / `--apply-grades`.

### Manual grades

Graded by: **Sravan** (file `manual_grades.csv`). Scale: answerable: 1 correct and complete, 0.5 partly correct, 0 wrong/ungrounded/unjustified refusal; unanswerable: 1 correctly says the documents do not contain it, 0 answers from outside knowledge.

| | no_retrieval | hybrid |
|---|---|---|
| Answers graded | 15 / 15 | 15 / 15 |
| Answerable: mean score (1 / 0.5 / 0) | 0.73 | 0.00 |
| Answerable: strictly correct (grade 1) | 0.55 | 0.00 |
| Unanswerable: handled correctly | 0.50 | 1.00 |
| All questions: behaviour correct | 0.53 | 0.27 |
