# Evaluation analysis

How to read the numbers in [RESULTS.md](RESULTS.md) and in the README, what they do and do not show, and the one false abstention.
Nothing here was tuned on the held-out questions.

## 1. Who graded, and how

The manual grades in `evaluation/results/manual_grades.csv` were entered by the author of this repository, who is also the author of
the system: they are a human judgement, but not an independent one. The report names the grader (`--grader`). The scale is in
`evaluation/grading.py`:

* answerable question: **1** every part of the expected answer is present and nothing materially wrong is stated; **0.5** the main
  point is right but a required part is missing or a secondary statement is wrong; **0** the main point is wrong or missing, or the
  question was refused although the documents answer it;
* unanswerable question: **1** the system declines to answer from the documents, **0** it answers with content that is not in them.

A grade is stored together with a fingerprint of the answer it was given to and is ignored if the answer changes, so a re-generated
answer is never silently matched to an old grade. The expected answers and the labelled evidence were fixed before the run
(`evaluation/dataset.json`; every snippet is proven to be on its page by `tests/test_dataset.py`).

## 2. What the automatic numbers measure

* **Answer correct, automatic proxy.** Every key fact of the question must occur in the answer. It misses paraphrases and cannot
  judge a refusal, so it usually differs from the manual grade; read the manual grades.
* **Citations verified by the validator.** The share of emitted citations whose document, page, clause and quote exist, were
  retrieved, and support the claim's numbers and terms. It says nothing about entailment.
* **Citation precision against the labelled evidence.** Strict: a valid citation of a neighbouring clause or a recital counts as a
  miss, because the labels list one clause per fact. An earlier manual inspection of the private prototype's run found that most of
  its "misses" were valid neighbouring passages and a few were tangential; that inspection has not been repeated for this run.
* **Region-level precision** (`evaluation/region_eval.py`) is stricter than page-level precision and shows how many citations are on
  the right page but in the wrong place.
* **Abstention accuracy** counts a correct decision either way. The no-retrieval baseline never refuses by design of the metric
  (it has no gate), so its score on unanswerable questions needs the manual grade: a baseline that says "no document was provided" refuses
  for the wrong reason, and one that answers from memory does exactly what the system is built to prevent.

## 3. The false abstention

**First run.** The held-out question M1 ("Under the GDPR, what is the maximum fine a company can receive, and which kinds of violations
trigger the higher fine?") was refused by the gate: its best cross-encoder score was 0.66, below the threshold 2.5. The diagnostics
(`python -m evaluation.diagnostics sample_docs/*.pdf`) showed that retrieval was not the problem: the labelled evidence (Article 83(4) and
83(5)) was among the top five passages, but the cross-encoder scored it at 0.59 and 0.66. `ms-marco-MiniLM` is trained on web queries and
answers, and a list-style legal clause ("Infringements of the following provisions shall ... be subject to administrative fines up to
EUR 20 000 000 ...") can receive a near-zero logit for a question phrased with "maximum" and "kinds of violations" although it is the answer.
The threshold was not lowered: choosing it because of what it does to the held-out set would be tuning on the test set.

**Final run.** After the later changes (the query vocabulary, the stricter number check, the table extractor, OCR repair and the prompt rule)
M1 is no longer refused, and no answerable question is. The diagnostics explain how: the best score for M1 is now 6.54, but the labelled
passages still score 0.59 and 0.83 (ranks 5 and 1). So the question passes the gate because **another passage** now scores high, not because the
cross-encoder judges the labelled clause better. The vocabulary adds "General Data Protection Regulation" to every question that says "GDPR",
which is the likely reason, but this was not isolated and the passage that scores 6.54 was not inspected. The scoring weakness of the
cross-encoder on list-style clauses is therefore still there; this run does not show that it is fixed. The sensitivity table in
[RESULTS.md](RESULTS.md) now shows the held-out set classified correctly at every threshold from 0 to 4, which reflects the same change.

**What would fix the weakness properly:** a larger independent dev set (to place the threshold with confidence), or a better-calibrated
relevance signal: a legal-domain reranker, or letting borderline scores through to the model while keeping the citation validator and
abstaining if nothing verifies. None of this is implemented.

## 4. What the evaluation does not show

* **Changes made after looking at other documents.** While testing question answering on documents outside the evaluation set,
  four things were changed: a query vocabulary (`retrieval/synonyms.py`), a stricter number check in the validator (every number of a
  claim must be in the quote cited for it), the table extractor and OCR repair, and one prompt rule. None of them was chosen by looking at
  held-out answers, but the vocabulary contains "GDPR", which occurs in many held-out questions, so the held-out set is no longer
  untouched by it. Retrieval, calibration, diagnostics and the answer pipeline all apply the vocabulary in the same way
  (`retrieve()` does it itself), and the numbers were measured after these changes.
* 15 held-out questions: one question moves a rate by 6-9 points; differences below about 10 points are noise.
* The question sets were written for the earlier prototype and reused unchanged. Query decomposition was added after looking at
  multi-hop failures of a similar prototype, so its gain is optimistic.
* All documents are EU legal texts in English. The thresholds were calibrated on them; other kinds of documents are unmeasured.
* The validator checks words and numbers, not meaning; an answer can contain a sentence that no citation supports.
* Latency was measured on one CPU laptop; other jobs may have shared the machine during a run.
