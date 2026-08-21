# OCR human-review classifier

`hitl_pipeline.py` learns two related decisions from `data/hitl/human.csv`:

1. whether a reviewer changed `Captured` into `Published` (the row should be reviewed), and
2. the likely correction class for flagged rows.

It combines OCR confidence, normalized edit similarity, character n-gram cosine similarity
(a local TF-IDF-like spelling signal), token overlap, answer-key structure, text-shape features,
and character confusions learned from reviewed corrections. GloVe is deliberately not required:
short OCR strings, numbers, punctuation, and names are often out-of-vocabulary, while downloading
a large embedding makes production scoring brittle. Existing GloVe experiments can remain useful
as offline comparisons.

The validation split is by `scan_id`, not by row. This prevents answers from the same scan leaking
between training and validation. The saved threshold is selected to reach 95% correction recall by
default; this makes the operational trade-off explicit and measurable.

```bash
python hitl_pipeline.py train
python hitl_pipeline.py score
```

For a quick smoke run:

```bash
python hitl_pipeline.py train --sample 10000 --model-out output/hitl/dev.pkl \
  --report-out output/hitl/dev_report.json
python hitl_pipeline.py score --model output/hitl/dev.pkl \
  --input-csv data/hitl/no_human.csv --output-csv output/hitl/no_human_flagged.csv
```

Inspect `validation_report.json` before deployment. In particular, compare correction recall,
precision, and review rate. Change `--target-recall` if review capacity cannot support the measured
review rate. Retrain whenever OCR engines, question formats, or language mix changes.

The scored CSV retains the principal evidence behind each decision: `edit_similarity`,
`char_ngram_similarity`, `token_similarity`, `answer_exact`, `known_confusion_score`, and
`known_confusion_fraction`, followed by the review probability, flag, risk label, and predicted
correction type.

For a non-MCQ row whose capture is empty or `--blank--`, distance and similarity evidence is left
empty. The model uses the dedicated `is_blank` feature instead of treating the blank marker as text.

Answer-key expansion is bounded and independent of `Captured`. Large grammar-style keys are
expanded deterministically in answer-key order up to `--max-variations`; only after parsing does
feature calculation compare the resulting valid variations with the OCR capture.

Correction-type predictions are structurally constrained: `mcq_correction` can only be emitted
when the answer key is an MCQ key. Risk (`high`/`medium`/`low`) remains a separate review-priority
field and is not an MCQ classification.
