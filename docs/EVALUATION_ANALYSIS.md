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

The held-out question M1 ("Under the GDPR, what is the maximum fine a company can receive, and which kinds of violations trigger the
higher fine?") is refused by the gate: its best cross-encoder score is 0.66, below the threshold 2.5 (`refusal_reason` in
`evaluation/results/generation_by_question.csv`).

* **Is retrieval the problem?** `python -m evaluation.diagnostics sample_docs/*.pdf` prints the rank and the cross-encoder score of the
  labelled evidence (Article 83(4) and 83(5)) for every question. In the earlier private prototype, with the same models and documents,
  both were in the top five and scored 0.66 and 0.59. Run the diagnostics to see this for the current index: if the labelled passages are
  in the top five with low scores, this is a scoring problem of the cross-encoder, not a retrieval failure.
* **Why would the score be low?** `ms-marco-MiniLM` is trained on web queries and answers. A list-style legal clause ("Infringements of
  the following provisions shall ... be subject to administrative fines up to EUR 20 000 000 ...") for a question phrased with
  "maximum" and "kinds of violations" can receive a near-zero logit although it is exactly the answer.
* **Why was the threshold not lowered?** The diagnostics table shows how the false-abstention and the correct-abstention rates move with
  the threshold on the dev set and on the held-out set. Choosing a threshold because of what it does to the held-out set would be
  tuning on the test set. The shipped value was chosen on the dev set only, where the unanswerable questions score close to the lowest
  usable threshold, so a lower value leaves little margin.
* **What would fix it properly?** A larger independent dev set (to place the threshold with confidence), or a better-calibrated
  relevance signal: a legal-domain reranker, or letting borderline scores through to the model while keeping the citation validator and
  abstaining if nothing verifies. None of this is implemented.

## 4. What the evaluation does not show

* 15 held-out questions: one question moves a rate by 6-9 points; differences below about 10 points are noise.
* The question sets were written for the earlier prototype and reused unchanged. Query decomposition was added after looking at
  multi-hop failures of a similar prototype, so its gain is optimistic.
* All documents are EU legal texts in English. The thresholds were calibrated on them; other kinds of documents are unmeasured.
* The validator checks words and numbers, not meaning; an answer can contain a sentence that no citation supports.
* Latency was measured on one CPU laptop; other jobs may have shared the machine during a run.
