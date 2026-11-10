Latency and cost per question (local model on this machine: no per-token fee, the cost is time and compute).

| Configuration | Questions | Mean s | Median s | 95th pct s | Retrieval + rerank s | Generation s | Prompt tokens | Completion tokens | Tokens/s |
|---|---|---|---|---|---|---|---|---|---|
| no_retrieval | 15 | 16.1 | 14.3 | 22.8 | 0.00 | 16.1 | 75 | 95 | 5.93 |
| hybrid | 15 | 70.6 | 67.2 | 178.1 | 0.93 | 69.6 | 1048 | 447 | 6.42 |
| hybrid_noverify | 15 | 68.1 | 69.7 | 175.4 | 0.91 | 67.1 | 1048 | 447 | 6.66 |

Optimisation 1: the sufficiency gate refuses an unanswerable question before the model is called.

| Configuration | Refused at the gate | Mean s when refused | Mean s when answered | Estimated seconds saved | Share of the run |
|---|---|---|---|---|---|
| hybrid | 5 | 0.6 | 105.6 | 525 | 33% |
| hybrid_noverify | 5 | 0.6 | 101.9 | 507 | 33% |

The saving is an estimate: it assumes a refused question would have taken as long as an answered one.
