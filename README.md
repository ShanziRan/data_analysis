# Digitisation data analysis

This repository contains exploratory OCR analysis tools and the current OCR review pipeline used to decide which captured assessment answers should be reviewed by a human verifier.

The current production-oriented entry point is `ocr_review.py`. It makes two separate decisions:

- **OCR review:** predicts whether the captured OCR text is likely to require correction.
- **Answer-key coverage review:** identifies non-MCQ answers that may be valid alternatives missing from the answer key. This is currently rule-based and uses whole-answer sentence embeddings plus conflict checks.

The system suggests review; it does not automatically correct captured answers or update answer keys.

## Repository contents

- `ocr_review.py` — trains the OCR review model and scores unreviewed rows.
- `OCR_REVIEW.md` — detailed model logic, outputs, thresholds and operational guidance.
- `pipeline.py` — older edit-distance and GloVe experiment; retained for comparison, not the main pipeline.
- `models/` — edit distance, TF-IDF, Word2Vec/GloVe, confidence and error-analysis experiments.
- `tools/` — answer-key parsing, question-type detection, character-change analysis, data filtering and plotting utilities.
- `scripts/` — helper scripts for reports, diagrams and legacy shell/PowerShell execution.
- `tests/` — automated tests for feature extraction and answer-key coverage/conflict rules.
- `data/hitl/human.csv` — human-reviewed training data.
- `data/hitl/no_human.csv` — data to score for review.
- `data/`, `data/ranged/` — source extracts and confidence-band analysis files.
- `error/` — derived OCR error examples and summaries.
- `output/hitl/` — trained models, scored queues, JSON reports and generated workflow assets.
- `output/pipeline/` — outputs from the older experimental pipeline.

Large CSVs, trained models and generated reports may contain sensitive assessment data. Confirm the appropriate storage and access policy before sharing or committing them.

## Expected input columns

The main pipeline expects at least:

- `Captured` — OCR-captured response.
- `ANSWER Key` — accepted answer or encoded answer-key variations.
- `confidence` — OCR confidence value.
- `scan_id` — grouping field used to keep related rows together during validation.

Training also requires:

- `Published` — human-verified response used only to create the correction label.

Additional columns are retained in scored output but are not necessarily model features.

## Setup

Use Python 3.10 or newer and create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Whole-answer semantic scoring additionally needs:

```bash
python -m pip install -r requirements-semantic.txt
```

The default semantic model is `sentence-transformers/all-MiniLM-L6-v2`. Its first use may download model files. In restricted environments, download the model through the approved process and pass its local directory with `--semantic-model`.

## Run the main HITL pipeline

Run commands from the repository root so the default relative paths resolve correctly.

Train and validate the OCR review model:

```bash
python ocr_review.py train
```

Defaults:

- input: `data/hitl/human.csv`
- model: `output/hitl/review_model.pkl`
- validation report: `output/hitl/validation_report.json`
- target correction recall: `0.98`

Score unreviewed data with both OCR and answer-key coverage decisions:

```bash
python ocr_review.py score
```

Defaults:

- input: `data/hitl/no_human.csv`
- model: `output/hitl/review_model.pkl`
- scored CSV: `output/hitl/no_human_flagged.csv`
- scoring report: `output/hitl/no_human_flagged.scoring_report.json`

Run OCR scoring without the optional semantic model:

```bash
python ocr_review.py score --skip-ak-coverage
```

Quick smoke test on smaller samples:

```bash
python ocr_review.py train --sample 10000 \
  --model-out output/hitl/dev.pkl \
  --report-out output/hitl/dev_report.json

python ocr_review.py score \
  --model output/hitl/dev.pkl \
  --input-csv data/hitl/no_human.csv \
  --output-csv output/hitl/dev_flagged.csv \
  --limit 1000
```

See all options with:

```bash
python ocr_review.py train --help
python ocr_review.py score --help
```

## Outputs to review

The scored CSV includes the original input columns plus evidence and decisions. Important fields include:

- OCR evidence: `edit_similarity`, `char_ngram_similarity`, `token_similarity`, `answer_exact`, `known_confusion_score` and `known_confusion_fraction`.
- OCR decision: `ocr_review_probability`, `requires_ocr_review`, `ocr_risk_label` and `predicted_ocr_correction_type`.
- Answer-key evidence: `semantic_similarity`, `best_semantic_variation`, `semantic_surface_gap`, `ak_conflict_detected` and `ak_conflict_reasons`.
- Answer-key decision: `requires_ak_review`, `ak_coverage_label` and `possible_gap_suggestion`.
- Combined routing: `requires_any_human_review` and `review_reasons`.

Review `validation_report.json` before deploying a model. The threshold is chosen on held-out, human-reviewed scans to meet the requested recall target. Raising target recall normally increases the number of rows sent to human review.

The scoring report records the model hash, thresholds, input/output paths, row counts, flag rates and score distributions. Recall and precision are only meaningful when verified `Published` truth is available.

## Legacy and analysis utilities

The older pipeline can still be run for comparison:

```bash
python pipeline.py data/hitl/no_human.csv
```

It uses edit distance and a downloadable GloVe model and writes a timestamped CSV under `output/pipeline/`. New development should normally use `ocr_review.py`.

Most files in `models/` and `tools/` are standalone analysis utilities rather than a single application. Read the script and check its paths/arguments before running it against full data.

## Development notes

- Keep OCR correction and answer-key coverage as separate decisions and labels.
- Do not use `Published` as a scoring feature; it is training/evaluation truth only.
- Parse answer keys independently of the captured answer.
- Non-MCQ blank captures must not receive edit or semantic similarity values.
- MCQ-only correction labels must only be produced for MCQ answer keys.
- Add or update tests when changing parsing, features, conflict checks, thresholds or output columns.
- Avoid committing generated models, large extracts or sensitive outputs unless repository policy explicitly requires it.
