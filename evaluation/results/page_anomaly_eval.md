Autoencoder page-anomaly detector, fitted on each document's own pages. Documents: Data_Act_2023_2854.pdf (71 pages), GDPR_Regulation_2016_679.pdf (88 pages). Seeds [42, 43, 44]; 10% of the pages of a document are replaced by the defective version.

| Defect | Defective pages | AUROC autoencoder | AUROC z-score baseline | Detected at K | False alarms on intact pages |
|---|---|---|---|---|---|
| garbled | 48 | 0.875 | 0.992 | 47% | 5.0% |
| symbol noise | 48 | 0.967 | 0.994 | 91% | 5.3% |
| near empty | 48 | 0.998 | 0.997 | 100% | 7.4% |
| number dump | 48 | 0.987 | 1.000 | 100% | 3.9% |
| repeated line | 48 | 0.943 | 0.908 | 66% | 5.0% |
| bad scan (OCR) | 45 | 0.989 | 1.000 | 100% | 5.4% |

Threshold = median + K x MAD of the document's own reconstruction errors. K = 3 is the smallest value whose false-alarm rate on the CLEAN documents is at most 10%; no defective page was used to choose it.

| K | 3 | 4 | 6 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|
| false alarms on clean documents | 9.0% | 5.2% | 3.2% | 1.1% | 1.1% | 0.6% | 0.6% |

Cross-document (a model fitted on the other documents, same K): share of INTACT pages flagged: Data_Act_2023_2854.pdf: 35%; GDPR_Regulation_2016_679.pdf: 12%.
