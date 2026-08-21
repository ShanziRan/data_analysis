"""Train and apply a human-in-the-loop OCR review classifier.

The only supervised target is derived from the human-reviewed file: a row needs
review when the normalized Captured value differs from Published.  Published is
never used as a feature and is not required when scoring new data.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

from models.edit_distance import levenshtein_distance
from tools.answer_parsing import expand_answer_variations_bounded, is_mcq_answer_key
from tools.char_error import extract_character_changes


FEATURE_COLUMNS = [
    "ocr_confidence", "edit_similarity", "char_ngram_similarity", "token_similarity",
    "answer_exact", "is_blank", "is_mcq", "captured_length", "best_answer_length",
    "length_difference", "digit_fraction", "alpha_fraction", "punctuation_fraction",
    "variation_count", "known_confusion_score", "known_confusion_fraction",
]


def _text(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _norm(value) -> str:
    return _text(value).casefold()


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError(f"Could not decode {path}")


def correction_type(captured, published, answer_key) -> str:
    raw_c, raw_p = _text(captured), _text(published)
    c, p = raw_c.casefold(), raw_p.casefold()
    if c == p:
        return "no_correction"
    if c in ("", "--blank--"):
        return "blank_to_text"
    if p in ("", "--blank--"):
        return "text_to_blank"
    if raw_c != raw_p and c == p:
        return "case_or_whitespace"
    if is_mcq_answer_key(answer_key):
        return "mcq_correction"
    if len(c) == len(p) == 2 and c == p[::-1]:
        return "transposition"
    changes = extract_character_changes(c, p)
    if len(changes) != 1:
        return "multiple_edits"
    before, after = changes[0].split(" -> ", 1)
    if before == "<ins>":
        return "insertion"
    if after == "<del>":
        return "deletion"
    return "substitution"


def learn_confusions(frame: pd.DataFrame, min_count: int = 2) -> dict[str, float]:
    """Learn smoothed character-change probabilities from reviewed corrections."""
    counts: Counter[str] = Counter()
    total = 0
    for captured, published in zip(frame["Captured"], frame["Published"]):
        if _norm(captured) == _norm(published):
            continue
        changes = extract_character_changes(captured, published)
        counts.update(changes)
        total += len(changes)
    denominator = total + max(len(counts), 1)
    return {k: (v + 1) / denominator for k, v in counts.items() if v >= min_count}


def _ngrams(text: str, n: int = 3) -> Counter[str]:
    padded = f"  {text}  "
    return Counter(padded[i:i + n] for i in range(max(0, len(padded) - n + 1)))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    denom = math.sqrt(sum(v * v for v in left.values()) * sum(v * v for v in right.values()))
    return sum(v * right.get(k, 0) for k, v in left.items()) / denom if denom else 0.0


def _token_similarity(left: str, right: str) -> float:
    a, b = set(re.findall(r"\w+", left)), set(re.findall(r"\w+", right))
    return len(a & b) / len(a | b) if a or b else 1.0


@lru_cache(maxsize=16384)
def _accepted_answers(answer_key, max_variations: int) -> list[str]:
    values = [_norm(v) for v in expand_answer_variations_bounded(
        answer_key, limit=max_variations
    )]
    return list(dict.fromkeys(v for v in values if v))


def row_features(row, confusions: dict[str, float], max_variations: int = 100) -> dict[str, float]:
    captured = _norm(row.get("Captured"))
    is_blank = captured in ("", "--blank--")
    is_mcq = is_mcq_answer_key(row.get("ANSWER Key"))
    confidence = pd.to_numeric(row.get("confidence"), errors="coerce")

    # A missing free-text response is a categorical OCR state, not a string that
    # should be compared with every accepted answer. Keep similarity/distance
    # genuinely missing and let is_blank carry the information into the model.
    if is_blank and not is_mcq:
        return {
            "ocr_confidence": float(confidence) if pd.notna(confidence) else 0.0,
            "min_distance": np.nan,
            "edit_similarity": np.nan,
            "char_ngram_similarity": np.nan,
            "token_similarity": np.nan,
            "answer_exact": 0.0,
            "is_blank": 1.0,
            "is_mcq": 0.0,
            "captured_length": 0,
            "best_answer_length": np.nan,
            "length_difference": np.nan,
            "digit_fraction": 0.0,
            "alpha_fraction": 0.0,
            "punctuation_fraction": 0.0,
            "variation_count": 0,
            "known_confusion_score": np.nan,
            "known_confusion_fraction": np.nan,
        }

    answers = _accepted_answers(row.get("ANSWER Key"), max_variations)
    if not answers:
        answers = [""]
    distances = [levenshtein_distance(captured, answer) for answer in answers]
    best_i = min(range(len(answers)), key=lambda i: distances[i])
    best, distance = answers[best_i], distances[best_i]
    scale = max(len(captured), len(best), 1)
    edit_similarity = 1 - distance / scale
    captured_grams = _ngrams(captured)
    char_similarity = max(_cosine(captured_grams, _ngrams(a)) for a in answers)
    token_similarity = max(_token_similarity(captured, a) for a in answers)
    changes = extract_character_changes(captured, best)
    probabilities = [confusions.get(change, 0.0) for change in changes]
    length = max(len(captured), 1)
    alnum = sum(ch.isalnum() for ch in captured)
    return {
        "ocr_confidence": float(confidence) if pd.notna(confidence) else 0.0,
        "min_distance": distance,
        "edit_similarity": edit_similarity,
        "char_ngram_similarity": char_similarity,
        "token_similarity": token_similarity,
        "answer_exact": float(captured in answers),
        "is_blank": float(is_blank),
        "is_mcq": float(is_mcq),
        "captured_length": len(captured),
        "best_answer_length": len(best),
        "length_difference": abs(len(captured) - len(best)),
        "digit_fraction": sum(ch.isdigit() for ch in captured) / length,
        "alpha_fraction": sum(ch.isalpha() for ch in captured) / length,
        "punctuation_fraction": (len(captured) - alnum) / length,
        "variation_count": len(answers),
        "known_confusion_score": max(probabilities, default=0.0),
        "known_confusion_fraction": sum(p > 0 for p in probabilities) / max(len(probabilities), 1),
    }


def make_features(frame: pd.DataFrame, confusions: dict[str, float], max_variations: int) -> pd.DataFrame:
    records = []
    for number, (_, row) in enumerate(frame.iterrows(), 1):
        records.append(row_features(row, confusions, max_variations))
        if number % 10000 == 0:
            print(f"Feature extraction: {number:,}/{len(frame):,}", flush=True)
    return pd.DataFrame.from_records(records, columns=FEATURE_COLUMNS + ["min_distance"])


def _threshold_for_recall(y_true, probability, target_recall: float) -> float:
    positive = np.asarray(probability)[np.asarray(y_true, dtype=bool)]
    if not len(positive):
        return 0.5
    return float(np.quantile(positive, max(0.0, 1.0 - target_recall), method="lower"))


def _threshold_metrics(y_true, probability, threshold: float) -> dict[str, float]:
    pred = np.asarray(probability) >= threshold
    actual = np.asarray(y_true, dtype=bool)
    tp, fp = int((pred & actual).sum()), int((pred & ~actual).sum())
    fn, tn = int((~pred & actual).sum()), int((~pred & ~actual).sum())
    return {
        "threshold": threshold, "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1), "review_rate": (tp + fp) / max(len(y_true), 1),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def _average_precision(y_true, probability) -> float:
    order = np.argsort(-np.asarray(probability))
    y = np.asarray(y_true, dtype=int)[order]
    positives = y.sum()
    if not positives:
        return 0.0
    return float((np.cumsum(y)[y == 1] / (np.flatnonzero(y == 1) + 1)).sum() / positives)


@dataclass
class LogisticModel:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    coef_: np.ndarray | None = None
    intercept_: float = 0.0

    def fit(self, x, y, sample_weight=None, epochs: int = 80, learning_rate: float = .08):
        values = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        target = np.asarray(y, dtype=float)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        z = np.clip((values - self.mean_) / self.scale_, -10, 10)
        self.coef_ = np.zeros(z.shape[1])
        self.intercept_ = 0.0
        weights = np.ones(len(z)) if sample_weight is None else np.asarray(sample_weight)
        weights = weights / weights.mean()
        # Full-batch weighted gradient descent is deterministic and fast for 16 features.
        for epoch in range(epochs):
            logits = np.clip(z @ self.coef_ + self.intercept_, -30, 30)
            error = (1 / (1 + np.exp(-logits)) - target) * weights
            rate = learning_rate / math.sqrt(1 + epoch / 10)
            self.coef_ -= rate * ((z.T @ error) / len(z) + .001 * self.coef_)
            self.intercept_ -= rate * error.mean()
        return self

    def predict_proba(self, x):
        values = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        z = np.clip((values - self.mean_) / self.scale_, -10, 10)
        p = 1 / (1 + np.exp(-np.clip(z @ self.coef_ + self.intercept_, -30, 30)))
        return np.column_stack([1 - p, p])


@dataclass
class CentroidClassifier:
    classes_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    centroids_: np.ndarray | None = None

    def fit(self, x, y):
        values = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        labels = np.asarray(y)
        self.mean_, self.scale_ = values.mean(axis=0), values.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        z = (values - self.mean_) / self.scale_
        self.classes_ = np.unique(labels)
        self.centroids_ = np.vstack([z[labels == label].mean(axis=0) for label in self.classes_])
        return self

    def predict(self, x):
        values = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        z = (values - self.mean_) / self.scale_
        distance = ((z[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return self.classes_[distance.argmin(axis=1)]

    def predict_constrained(self, x, is_mcq):
        """Prevent structurally impossible MCQ/non-MCQ type predictions."""
        values = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        z = (values - self.mean_) / self.scale_
        distance = ((z[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        output = []
        for row_number, mcq in enumerate(np.asarray(is_mcq, dtype=bool)):
            if mcq and "mcq_correction" in self.classes_:
                output.append("mcq_correction")
                continue
            allowed = self.classes_ != "mcq_correction"
            if not allowed.any():
                output.append("multiple_edits")
                continue
            allowed_indices = np.flatnonzero(allowed)
            nearest = allowed_indices[distance[row_number, allowed].argmin()]
            output.append(self.classes_[nearest])
        return np.asarray(output, dtype=object)


def _columns_expected_by_model(model, declared_columns, available_columns) -> list[str]:
    """Resolve feature columns, including artifacts made during the distance-log transition.

    A short-lived version trained on all generated columns (17, including
    ``min_distance``) but saved the intended 16 feature names. Supporting that
    shape here avoids forcing users to repeat a full training run.
    """
    columns = list(declared_columns)
    model_mean = getattr(model, "mean_", None)
    expected = len(model_mean) if model_mean is not None else 0
    if expected == len(columns):
        return columns
    if expected == len(columns) + 1 and "min_distance" in available_columns:
        return columns + ["min_distance"]
    raise ValueError(
        f"Model expects {expected} features, but its artifact declares {len(columns)}. "
        "Retrain the model with the current hitl_pipeline.py."
    )


@dataclass
class Artifact:
    review_model: object
    type_model: object | None
    type_labels: list[str]
    confusions: dict[str, float]
    review_threshold: float
    max_variations: int
    feature_columns: list[str]


def train(args) -> None:
    data = read_csv(args.human_csv)
    required = {"Captured", "Published", "ANSWER Key", "confidence"}
    if missing := required - set(data.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if args.sample and args.sample < len(data):
        data = data.sample(args.sample, random_state=args.random_state).reset_index(drop=True)

    y = (_norm_series(data["Captured"]) != _norm_series(data["Published"])).astype(int)
    groups = data[args.group_column].astype(str) if args.group_column in data else pd.Series(data.index)
    unique_groups = groups.drop_duplicates().to_numpy()
    rng = np.random.default_rng(args.random_state)
    rng.shuffle(unique_groups)
    valid_groups = set(unique_groups[:max(1, int(len(unique_groups) * args.validation_size))])
    valid_mask = groups.isin(valid_groups).to_numpy()
    train_i, valid_i = np.flatnonzero(~valid_mask), np.flatnonzero(valid_mask)
    train_data, valid_data = data.iloc[train_i], data.iloc[valid_i]
    # Confusions are learned only on the training fold for honest validation.
    confusions = learn_confusions(train_data)
    x_train = make_features(train_data, confusions, args.max_variations)
    x_valid = make_features(valid_data, confusions, args.max_variations)
    y_train, y_valid = y.iloc[train_i], y.iloc[valid_i]
    weights = np.where(y_train == 1, len(y_train) / max(2 * y_train.sum(), 1),
                       len(y_train) / max(2 * (len(y_train) - y_train.sum()), 1))
    review_model = LogisticModel()
    review_model.fit(x_train[FEATURE_COLUMNS], y_train, sample_weight=weights)
    probability = review_model.predict_proba(x_valid[FEATURE_COLUMNS])[:, 1]
    threshold = _threshold_for_recall(y_valid, probability, args.target_recall)

    types = data.apply(lambda r: correction_type(r["Captured"], r["Published"], r["ANSWER Key"]), axis=1)
    corrected_train = y_train.to_numpy(dtype=bool)
    type_model = None
    type_labels: list[str] = []
    if corrected_train.sum() >= 20 and types.iloc[train_i][corrected_train].nunique() > 1:
        type_model = CentroidClassifier()
        type_model.fit(x_train.loc[corrected_train, FEATURE_COLUMNS], types.iloc[train_i][corrected_train])
        type_labels = list(type_model.classes_)

    # Refit confusion statistics and feature-dependent models on all reviewed data.
    final_confusions = learn_confusions(data)
    x_all = make_features(data, final_confusions, args.max_variations)
    all_weights = np.where(y == 1, len(y) / max(2 * y.sum(), 1), len(y) / max(2 * (len(y) - y.sum()), 1))
    review_model.fit(x_all[FEATURE_COLUMNS], y, sample_weight=all_weights)
    if type_model is not None:
        type_model.fit(x_all.loc[y.astype(bool), FEATURE_COLUMNS], types.loc[y.astype(bool)])

    artifact = Artifact(review_model, type_model, type_labels, final_confusions, threshold,
                        args.max_variations, FEATURE_COLUMNS)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    with args.model_out.open("wb") as handle:
        pickle.dump(artifact, handle)
    report = {
        "rows": len(data), "corrections": int(y.sum()), "correction_rate": float(y.mean()),
        "validation_rows": len(valid_i), "validation_average_precision": _average_precision(y_valid, probability),
        "operating_point": _threshold_metrics(y_valid, probability, threshold),
        "target_recall": args.target_recall, "group_column": args.group_column,
        "correction_types": types.value_counts().to_dict(), "feature_columns": FEATURE_COLUMNS,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved model to {args.model_out} and validation report to {args.report_out}")


def _norm_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.casefold()


def score(args) -> None:
    with args.model.open("rb") as handle:
        artifact: Artifact = pickle.load(handle)
    data = read_csv(args.input_csv)
    if args.limit:
        data = data.head(args.limit).copy()
    required = {"Captured", "ANSWER Key", "confidence"}
    if missing := required - set(data.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    features = make_features(data, artifact.confusions, artifact.max_variations)
    review_columns = _columns_expected_by_model(
        artifact.review_model, artifact.feature_columns, features.columns
    )
    probability = artifact.review_model.predict_proba(features[review_columns])[:, 1]
    threshold = args.threshold if args.threshold is not None else artifact.review_threshold
    result = data.copy()
    # Keep the principal evidence used by the classifier in the output so each
    # review decision can be inspected and analysed without recalculating it.
    evidence_columns = [
        "min_distance",
        "edit_similarity",
        "char_ngram_similarity",
        "token_similarity",
        "answer_exact",
        "known_confusion_score",
        "known_confusion_fraction",
    ]
    for column in evidence_columns:
        result[column] = features[column].to_numpy()
    result["review_probability"] = probability
    result["requires_human_review"] = probability >= threshold
    result["risk_label"] = np.where(probability >= max(.8, threshold), "high",
                            np.where(probability >= threshold, "medium", "low"))
    if artifact.type_model is not None:
        type_columns = _columns_expected_by_model(
            artifact.type_model, artifact.feature_columns, features.columns
        )
        if hasattr(artifact.type_model, "predict_constrained"):
            result["predicted_correction_type"] = artifact.type_model.predict_constrained(
                features[type_columns], features["is_mcq"]
            )
        else:
            # Compatibility for any older classifier implementation.
            result["predicted_correction_type"] = artifact.type_model.predict(features[type_columns])
            impossible = (
                features["is_mcq"].eq(0)
                & result["predicted_correction_type"].eq("mcq_correction")
            )
            result.loc[impossible, "predicted_correction_type"] = "multiple_edits"
        result.loc[~result["requires_human_review"], "predicted_correction_type"] = "no_correction"
    result["review_threshold"] = threshold
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"Flagged {result['requires_human_review'].sum():,}/{len(result):,} rows; saved {args.output_csv}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train", help="train and validate on human-reviewed rows")
    fit.add_argument("--human-csv", type=Path, default=Path("data/hitl/human.csv"))
    fit.add_argument("--model-out", type=Path, default=Path("output/hitl/review_model.pkl"))
    fit.add_argument("--report-out", type=Path, default=Path("output/hitl/validation_report.json"))
    fit.add_argument("--group-column", default="scan_id")
    fit.add_argument("--validation-size", type=float, default=.2)
    fit.add_argument("--target-recall", type=float, default=.95)
    fit.add_argument("--max-variations", type=int, default=100)
    fit.add_argument("--sample", type=int, help="optional development sample size")
    fit.add_argument("--random-state", type=int, default=42)
    fit.set_defaults(func=train)
    apply = sub.add_parser("score", help="flag unreviewed rows")
    apply.add_argument("--model", type=Path, default=Path("output/hitl/review_model.pkl"))
    apply.add_argument("--input-csv", type=Path, default=Path("data/hitl/no_human.csv"))
    apply.add_argument("--output-csv", type=Path, default=Path("output/hitl/no_human_flagged.csv"))
    apply.add_argument("--threshold", type=float)
    apply.add_argument("--limit", type=int, help="optional smoke-test row limit")
    apply.set_defaults(func=score)
    return root


if __name__ == "__main__":
    options = parser().parse_args()
    options.func(options)
