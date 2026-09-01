# OCR human-review classifier

`ocr_review.py` now produces two separate decisions:

1. **OCR review:** whether a reviewer is likely to change `Captured` into `Published`, and
2. **answer-key coverage:** whether a non-MCQ response may be a valid semantic alternative missing
   from the current answer key.

Only the OCR decision is trained from `Captured != Published`. Answer-key coverage is a separate
candidate-generation rule because the current reviewed data does not say whether a new answer-key
variation should be accepted.

It combines OCR confidence, normalized edit similarity, character n-gram cosine similarity
(a local TF-IDF-like spelling signal), token overlap, answer-key structure, text-shape features,
and character confusions learned from reviewed corrections. GloVe is deliberately not used by the
OCR model:
short OCR strings, numbers, punctuation, and names are often out-of-vocabulary, while downloading
a large embedding makes production scoring brittle. Existing GloVe experiments can remain useful
as offline comparisons.

During scoring, the AK-coverage stage uses a pretrained sentence-transformer over each complete
captured string and each complete parsed answer-key variation. It records the best cosine
similarity. The binary `possible_gap_suggestion` is true only when semantic similarity is high,
edit and character n-gram surface similarity are low, the semantic-to-surface gap is at least
`0.15`, and no conflict is detected. MCQ, blank and exact-match rows are not eligible.

Conflict checks prevent semantic relatedness being mistaken for equivalence. They cover different
single-word answers (`cat` versus `dog`), numbers, dates/times, negation, opposing polarity,
measurement units, and short near-duplicates whose key term changes. Blocked rows retain their
semantic evidence but receive `blocked_by_conflict`, with details in `ak_conflict_reasons`.

The validation split is by `scan_id`, not by row. This prevents answers from the same scan leaking
between training and validation. The saved threshold is selected to reach 95% correction recall by
default; this makes the operational trade-off explicit and measurable.

```bash
python ocr_review.py train
python ocr_review.py score
```

The default semantic model is `sentence-transformers/all-MiniLM-L6-v2`. The first semantic run may
need to download it. Install the optional dependency with
`pip install -r requirements-semantic.txt`. A downloaded local model can be supplied instead. Use
`--skip-ak-coverage` to run only the OCR decision:

```bash
python ocr_review.py score --semantic-model /path/to/local/sentence-model
python ocr_review.py score --skip-ak-coverage
```

For a quick smoke run:

```bash
python ocr_review.py train --sample 10000 --model-out output/hitl/dev.pkl \
  --report-out output/hitl/dev_report.json
python ocr_review.py score --model output/hitl/dev.pkl \
  --input-csv data/hitl/no_human.csv --output-csv output/hitl/no_human_flagged.csv
```

Inspect `validation_report.json` before deployment. In particular, compare correction recall,
precision, and review rate. Change `--target-recall` if review capacity cannot support the measured
review rate. Retrain whenever OCR engines, question formats, or language mix changes.

The scored CSV retains the principal evidence behind each decision: `edit_similarity`,
`char_ngram_similarity`, `token_similarity`, `answer_exact`, `known_confusion_score`, and
`known_confusion_fraction`.

OCR outputs are `ocr_review_probability`, `requires_ocr_review`, `ocr_risk_label`, and
`predicted_ocr_correction_type`. AK outputs are `semantic_similarity`,
`best_semantic_variation`, `semantic_surface_gap`, `requires_ak_review`, `ak_coverage_label`, and
`ak_review_suggestion`. The additional AK audit columns are `possible_gap_suggestion`,
`ak_required_semantic_threshold`, `ak_conflict_detected`, and `ak_conflict_reasons`.
`requires_any_human_review` and `review_reasons` route the combined queue
without merging the underlying decisions. Legacy OCR column names remain as aliases for existing
downstream reports.

Every scoring run also writes a JSON report next to the CSV, using the suffix
`.scoring_report.json`. It records input/model/output paths, a model SHA-256 identifier, all OCR and
AK thresholds, semantic model settings, row profiles, flagged counts and rates, label/reason counts,
and probability/similarity summaries. Use `--report-out PATH` to choose another location. Recall
and precision are not calculated for ordinary unreviewed scoring, even if a provisional `Published`
column happens to exist. Use `--evaluate-with-published` only when that column contains verified
human truth.

AK thresholds can be adjusted independently with `--ak-semantic-threshold` (default `0.75`) and
`--ak-surface-threshold` (default `0.60`). The stricter phrase thresholds default to `0.78` when
one side is a single word and `0.80` when both sides are multi-word; the minimum semantic-surface
gap defaults to `0.15`. These are initial operating rules, not validated claims
of correctness. Reviewer labels such as accepted new variation, incorrect response, OCR error and
uncertain are needed to train and validate a future AK-coverage model.

For a non-MCQ row whose capture is empty or `--blank--`, distance and similarity evidence is left
empty. The model uses the dedicated `is_blank` feature instead of treating the blank marker as text.

Answer-key expansion is bounded and independent of `Captured`. Large grammar-style keys are
expanded deterministically in answer-key order up to `--max-variations`; only after parsing does
feature calculation compare the resulting valid variations with the OCR capture.

Correction-type predictions are structurally constrained: `mcq_correction` can only be emitted
when the answer key is an MCQ key. Risk (`high`/`medium`/`low`) remains a separate review-priority
field and is not an MCQ classification.
