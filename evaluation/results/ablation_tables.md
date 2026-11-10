Text only versus text + tables on a generated contract (Services_Agreement.pdf), 6 questions answerable only from table cells. Table chunks in the index: {'text only': 0, 'text + tables': 3}.

| Search | Index | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|
| bm25 | text only | 1.00 | 1.00 | 1.00 | 1.00 |
| bm25 | text + tables | 0.83 | 1.00 | 1.00 | 0.92 |
| hybrid + rerank | text only | 1.00 | 1.00 | 1.00 | 1.00 |
| hybrid + rerank | text + tables | 1.00 | 1.00 | 1.00 | 1.00 |

A chunk counts as the evidence when it contains every string of the answer. Six questions: a direction, not a significance test.
