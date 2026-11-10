| rerank threshold | held-out: accuracy / false abstention / correct abstention | dev: accuracy / false abstention / correct abstention |
|---|---|---|
| -1 | 0.93 / 0.00 / 0.75 | 0.93 / 0.00 / 0.83 |
| 0 | 1.00 / 0.00 / 1.00 | 0.93 / 0.00 / 0.83 |
| 0.25 | 1.00 / 0.00 / 1.00 | 0.93 / 0.00 / 0.83 |
| 0.5 | 1.00 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 |
| 1 | 0.93 / 0.09 / 1.00 | 1.00 / 0.00 / 1.00 |
| 1.5 | 0.93 / 0.09 / 1.00 | 1.00 / 0.00 / 1.00 |
| 2 | 0.93 / 0.09 / 1.00 | 1.00 / 0.00 / 1.00 |
| 2.5 | 0.93 / 0.09 / 1.00 | 1.00 / 0.00 / 1.00 |
| 3 | 0.93 / 0.09 / 1.00 | 1.00 / 0.00 / 1.00 |
| 4 | 0.93 / 0.09 / 1.00 | 1.00 / 0.00 / 1.00 |

Shipped thresholds: rerank >= 2.5, coverage >= 0.32 (chosen on the dev set only).

False abstention M1: best rerank score 0.66, coverage 0.50; labelled evidence at rank(s) [4, 2] with cross-encoder score(s) [0.59, 0.66].
