Latency and cost per question (local model on this machine: no per-token fee, the cost is time and compute).

| Configuration | Questions | Mean s | Median s | 95th pct s | Retrieval + rerank s | Generation s | Prompt tokens | Completion tokens | Tokens/s |
|---|---|---|---|---|---|---|---|---|---|
| no_retrieval | 15 | 16.1 | 14.7 | 23.1 | 0.00 | 16.1 | 75 | 95 | 5.92 |
| hybrid | 15 | 77.6 | 90.1 | 173.5 | 3.63 | 73.9 | 1145 | 482 | 6.53 |
| hybrid_noverify | 15 | 72.7 | 79.7 | 170.9 | 3.41 | 69.2 | 1023 | 453 | 6.55 |

Optimisation 1: the sufficiency gate refuses an unanswerable question before the model is called.

| Configuration | Refused at the gate | Mean s when refused | Mean s when answered | Estimated seconds saved | Share of the run |
|---|---|---|---|---|---|
| hybrid | 4 | 1.0 | 105.5 | 418 | 26% |
| hybrid_noverify | 4 | 0.8 | 98.8 | 392 | 26% |

The saving is an estimate: it assumes a refused question would have taken as long as an answered one.
