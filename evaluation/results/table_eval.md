Table extraction against a hand-written ground truth (synthetic PDFs, exact truth).

| Case | Tables found / expected | Cells right / expected | Accuracy |
|---|---|---|---|
| plain grid, numbers | 1 / 1 | 20 / 20 | 100% |
| legal table with caption | 1 / 1 | 12 / 12 | 100% |
| header spanning three columns | 1 / 1 | 10 / 10 | 100% |
| label spanning rows | 1 / 1 | 13 / 13 | 100% |
| wrapped multi-line cells | 1 / 1 | 6 / 6 | 100% |
| two tables on one page | 2 / 2 | 21 / 21 | 100% |
| long table, 12 rows | 1 / 1 | 65 / 65 | 100% |
| horizontal rules only (not expected to work) | 0 / 1 | 0 / 20 | 0% |
| no ruling lines (not expected to work) | 0 / 1 | 0 / 20 | 0% |

* detection: 8 of 10 tables (80%)
* cell accuracy over detected tables: 100.0%; overall (missed tables count as 0): 78.6%; over the cases with ruling lines: 100.0%
* captions attached: 1 of 1

Tables the extractor finds in real documents (no ground truth, so no accuracy):

| Document | Pages | Tables found |
|---|---|---|
| Data_Act_2023_2854.pdf | 71 | 0 |
| GDPR_Regulation_2016_679.pdf | 88 | 0 |
| ePrivacy_Directive_2002_58.pdf | 11 | 0 |
