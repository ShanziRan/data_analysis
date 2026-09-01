"""Train and apply OCR-review and answer-key-coverage decisions.

The only supervised target is derived from the human-reviewed file: a row needs
review when the normalized Captured value differs from Published.  Published is
never used as a feature and is not required when scoring new data.

Answer-key coverage is deliberately separate.  A pretrained sentence encoder
compares the complete captured answer with each accepted answer-key variation.
It flags semantically close but textually different answers as candidates for
answer-key review; it does not change the OCR review probability or label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
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

DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALE_WORDS = {"hundred": 100, "thousand": 1000}
_TEMPORAL_WORDS = set(
    "monday tuesday wednesday thursday friday saturday sunday "
    "january february march april may june july august september october november december "
    "morning afternoon evening night noon midnight am pm".split()
)
_NEGATIONS = {"no", "not", "never", "neither", "without", "cannot", "can't", "isn't", "wasn't", "don't", "doesn't"}
_POLARITY_GROUPS = [
    ({"yes", "true", "correct"}, {"no", "false", "incorrect"}),
    ({"before", "earlier"}, {"after", "later"}),
    ({"increase", "increased", "rises", "rose", "more", "higher"},
     {"decrease", "decreased", "falls", "fell", "less", "lower"}),
    ({"north"}, {"south"}), ({"east"}, {"west"}),
    ({"male", "man", "boy"}, {"female", "woman", "girl"}),
]
_UNIT_ALIASES = {
    "pound": "gbp", "pounds": "gbp", "gbp": "gbp", "£": "gbp",
    "pence": "pence", "penny": "pence", "pennies": "pence", "p": "pence",
    "dollar": "usd", "dollars": "usd", "usd": "usd", "$": "usd",
    "euro": "eur", "euros": "eur", "eur": "eur", "€": "eur",
    "metre": "metre", "metres": "metre", "meter": "metre", "meters": "metre",
    "centimetre": "cm", "centimetres": "cm", "centimeter": "cm", "centimeters": "cm", "cm": "cm",
    "kilometre": "km", "kilometres": "km", "kilometer": "km", "kilometers": "km", "km": "km",
    "gram": "g", "grams": "g", "g": "g", "kilogram": "kg", "kilograms": "kg", "kg": "kg",
    "second": "second", "seconds": "second", "minute": "minute", "minutes": "minute",
    "hour": "hour", "hours": "hour", "day": "day", "days": "day",
}


def _semantic_tokens(text: str) -> list[str]:
    return re.findall(r"(?u)\b[\w']+\b|[£$€]", _norm(text))


def _number_values(tokens: list[str]) -> set[str]:
    values: set[str] = set(re.findall(r"\d+(?:[.,]\d+)?", " ".join(tokens)))
    current = 0
    active = False
    for token in [*tokens, "<end>"]:
        if token in _NUMBER_WORDS:
            current += _NUMBER_WORDS[token]
            active = True
        elif token == "hundred" and active:
            current = max(current, 1) * 100
        elif token == "thousand" and active:
            current *= 1000
        elif active:
            values.add(str(current))
            current, active = 0, False
    return values


def _ak_conflict_reasons(captured: str, candidate: str) -> list[str]:
    """Conservative contradictions that semantic relatedness must not override."""
    left, right = _semantic_tokens(captured), _semantic_tokens(candidate)
    left_set, right_set = set(left), set(right)
    reasons = []

    if len(left) == len(right) == 1 and left != right:
        reasons.append("single_token_substitution")
    if _number_values(left) != _number_values(right) and (_number_values(left) or _number_values(right)):
        reasons.append("number_mismatch")
    left_time, right_time = left_set & _TEMPORAL_WORDS, right_set & _TEMPORAL_WORDS
    temporal_relation = {"after", "before", "next", "previous", "following", "preceding"}
    if (left_time and right_time and left_time != right_time
            and not ((left_set | right_set) & temporal_relation)):
        reasons.append("date_or_time_mismatch")
    if bool(left_set & _NEGATIONS) != bool(right_set & _NEGATIONS):
        reasons.append("negation_mismatch")
    for positive, negative in _POLARITY_GROUPS:
        if ((left_set & positive and right_set & negative)
                or (left_set & negative and right_set & positive)):
            reasons.append("polarity_mismatch")
            break
    left_units = {_UNIT_ALIASES[t] for t in left if t in _UNIT_ALIASES}
    right_units = {_UNIT_ALIASES[t] for t in right if t in _UNIT_ALIASES}
    if left_units and right_units and left_units != right_units:
        reasons.append("measurement_unit_mismatch")

    # Short near-duplicates often differ in the one entity/value that matters:
    # "red cat" versus "red dog", or "John Smith" versus "James Smith".
    if 2 <= len(left) <= 4 and len(left) == len(right):
        shared = left_set & right_set
        if len(shared) >= len(left_set) - 1 and left_set != right_set:
            reasons.append("key_term_mismatch")
    return list(dict.fromkeys(reasons))


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


class SemanticEncoder:
    """Encode complete strings with a pretrained sentence-transformer model.

    ``sentence-transformers`` is used when available.  A Transformers mean-
    pooling fallback keeps the pipeline usable with an equivalent local or
    Hugging Face model without changing the AK-coverage logic.
    """

    def __init__(self, model_name: str, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._backend = None
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._backend is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except (OSError, ValueError):
                self._model = SentenceTransformer(self.model_name)
            self._backend = "sentence-transformers"
            return
        except ModuleNotFoundError:
            pass

        try:
            from transformers import AutoModel, AutoTokenizer
            import torch  # noqa: F401 - checked here for a clearer error
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Semantic AK coverage requires sentence-transformers, or both "
                "transformers and torch. Install sentence-transformers or run "
                "score with --skip-ak-coverage."
            ) from error
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, local_files_only=True
            )
            self._model = AutoModel.from_pretrained(
                self.model_name, local_files_only=True
            )
        except (OSError, ValueError):
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._backend = "transformers"

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if self._backend == "sentence-transformers":
            return np.asarray(self._model.encode(
                texts, batch_size=self.batch_size, normalize_embeddings=True,
                show_progress_bar=False,
            ), dtype=np.float32)

        import torch

        batches = []
        for start in range(0, len(texts), self.batch_size):
            tokens = self._tokenizer(
                texts[start:start + self.batch_size], padding=True, truncation=True,
                max_length=256, return_tensors="pt",
            )
            with torch.no_grad():
                hidden = self._model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            batches.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(batches)


def make_ak_coverage_decisions(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    encoder,
    max_variations: int,
    semantic_threshold: float = .75,
    surface_threshold: float = .60,
    min_semantic_surface_gap: float = .15,
    single_phrase_threshold: float = .78,
    multiword_threshold: float = .80,
    chunk_size: int = 2000,
) -> pd.DataFrame:
    """Return a distinct, rule-based answer-key coverage decision per row."""
    thresholds = [semantic_threshold, surface_threshold, min_semantic_surface_gap,
                  single_phrase_threshold, multiword_threshold]
    if any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("AK coverage thresholds must be between 0 and 1")

    output = pd.DataFrame({
        "semantic_similarity": np.full(len(frame), np.nan),
        "best_semantic_variation": pd.Series([None] * len(frame), dtype=object),
        "semantic_surface_gap": np.full(len(frame), np.nan),
        "ak_required_semantic_threshold": np.full(len(frame), np.nan),
        "ak_conflict_detected": np.zeros(len(frame), dtype=bool),
        "ak_conflict_reasons": pd.Series([""] * len(frame), dtype=object),
        "possible_gap_suggestion": np.zeros(len(frame), dtype=bool),
        "requires_ak_review": np.zeros(len(frame), dtype=bool),
        "ak_coverage_label": pd.Series(["no_ak_gap_signal"] * len(frame), dtype=object),
        "ak_review_suggestion": pd.Series(["none"] * len(frame), dtype=object),
    })

    eligible = (
        features["is_mcq"].eq(0)
        & features["is_blank"].eq(0)
        & features["answer_exact"].eq(0)
    ).to_numpy()
    eligible_positions = np.flatnonzero(eligible)

    for chunk_start in range(0, len(eligible_positions), chunk_size):
        positions = eligible_positions[chunk_start:chunk_start + chunk_size]
        row_answers = []
        unique_texts = []
        seen = set()
        for position in positions:
            captured = _norm(frame.iloc[position].get("Captured"))
            answers = _accepted_answers(
                frame.iloc[position].get("ANSWER Key"), max_variations
            )
            row_answers.append((captured, answers))
            for text in [captured, *answers]:
                if text and text not in seen:
                    seen.add(text)
                    unique_texts.append(text)

        if not unique_texts:
            continue
        vectors = encoder.encode(unique_texts)
        vector_by_text = dict(zip(unique_texts, vectors))

        for position, (captured, answers) in zip(positions, row_answers):
            if not answers or captured not in vector_by_text:
                continue
            candidates = [a for a in answers if a in vector_by_text]
            if not candidates:
                continue
            scores = np.asarray([
                float(np.dot(vector_by_text[captured], vector_by_text[a]))
                for a in candidates
            ])
            best_index = int(scores.argmax())
            semantic = float(np.clip(scores[best_index], -1.0, 1.0))
            surface = np.nanmax([
                features.iloc[position]["edit_similarity"],
                features.iloc[position]["char_ngram_similarity"],
            ])
            gap = semantic - surface
            captured_tokens = _semantic_tokens(captured)
            candidate_tokens = _semantic_tokens(candidates[best_index])
            if len(captured_tokens) == 1 or len(candidate_tokens) == 1:
                required_semantic = max(semantic_threshold, single_phrase_threshold)
            else:
                required_semantic = max(semantic_threshold, multiword_threshold)
            conflicts = _ak_conflict_reasons(captured, candidates[best_index])
            requires_review = (
                semantic >= required_semantic
                and surface <= surface_threshold
                and gap >= min_semantic_surface_gap
                and not conflicts
            )
            output.at[position, "semantic_similarity"] = semantic
            output.at[position, "best_semantic_variation"] = candidates[best_index]
            output.at[position, "semantic_surface_gap"] = gap
            output.at[position, "ak_required_semantic_threshold"] = required_semantic
            output.at[position, "ak_conflict_detected"] = bool(conflicts)
            output.at[position, "ak_conflict_reasons"] = ";".join(conflicts)
            output.at[position, "possible_gap_suggestion"] = requires_review
            output.at[position, "requires_ak_review"] = requires_review
            if requires_review:
                output.at[position, "ak_coverage_label"] = "possible_answer_key_gap"
                output.at[position, "ak_review_suggestion"] = "review_for_possible_ak_change"
            elif conflicts:
                output.at[position, "ak_coverage_label"] = "blocked_by_conflict"
            elif semantic < required_semantic:
                output.at[position, "ak_coverage_label"] = "semantic_below_threshold"
            elif surface > surface_threshold:
                output.at[position, "ak_coverage_label"] = "surface_match_existing_ak"
            elif gap < min_semantic_surface_gap:
                output.at[position, "ak_coverage_label"] = "insufficient_semantic_surface_gap"

        print(
            f"Semantic AK coverage: {min(chunk_start + len(positions), len(eligible_positions)):,}/"
            f"{len(eligible_positions):,}", flush=True,
        )
    return output


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
        "Retrain the model with the current ocr_review.py."
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


def _numeric_summary(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_scoring_report(args, artifact: Artifact, features: pd.DataFrame,
                         result: pd.DataFrame, threshold: float) -> dict:
    rows = len(result)
    ak_eligible = (
        features["is_mcq"].eq(0)
        & features["is_blank"].eq(0)
        & features["answer_exact"].eq(0)
    )
    ocr_flagged = int(result["requires_ocr_review"].sum())
    ak_flagged = int(result["requires_ak_review"].sum())
    combined_flagged = int(result["requires_any_human_review"].sum())
    correction_counts = (
        _value_counts(result["predicted_ocr_correction_type"])
        if "predicted_ocr_correction_type" in result else {}
    )
    ocr_evaluation = None
    if args.evaluate_with_published:
        if "Published" not in result:
            raise ValueError("--evaluate-with-published requires a Published column")
        actual_correction = (
            _norm_series(result["Captured"]) != _norm_series(result["Published"])
        ).astype(int)
        ocr_evaluation = {
            **_threshold_metrics(
                actual_correction, result["ocr_review_probability"], threshold
            ),
            "average_precision": _average_precision(
                actual_correction, result["ocr_review_probability"]
            ),
        }
    return {
        "report_type": "scoring",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "files": {
            "input_csv": str(args.input_csv),
            "model": str(args.model),
            "model_sha256": _sha256(args.model),
            "output_csv": str(args.output_csv),
        },
        "parameters": {
            "limit": args.limit,
            "max_variations": int(artifact.max_variations),
            "ocr_threshold": float(threshold),
            "artifact_ocr_threshold": float(artifact.review_threshold),
            "ocr_threshold_overridden": args.threshold is not None,
            "ak_coverage_enabled": not args.skip_ak_coverage,
            "semantic_model": None if args.skip_ak_coverage else args.semantic_model,
            "ak_semantic_threshold": float(args.ak_semantic_threshold),
            "ak_surface_threshold": float(args.ak_surface_threshold),
            "ak_min_semantic_surface_gap": float(args.ak_min_semantic_surface_gap),
            "ak_single_phrase_threshold": float(args.ak_single_phrase_threshold),
            "ak_multiword_threshold": float(args.ak_multiword_threshold),
            "semantic_batch_size": int(args.semantic_batch_size),
            "semantic_chunk_size": int(args.semantic_chunk_size),
            "evaluate_with_published": bool(args.evaluate_with_published),
        },
        "data_profile": {
            "mcq_rows": int(features["is_mcq"].sum()),
            "blank_rows": int(features["is_blank"].sum()),
            "exact_answer_rows": int(features["answer_exact"].sum()),
            "ak_eligible_rows": int(ak_eligible.sum()),
        },
        "ocr_review": {
            "flagged_rows": ocr_flagged,
            "flagged_rate": ocr_flagged / max(rows, 1),
            "risk_label_counts": _value_counts(result["ocr_risk_label"]),
            "predicted_correction_type_counts": correction_counts,
            "probability_summary": _numeric_summary(result["ocr_review_probability"]),
            "ground_truth_metrics": ocr_evaluation,
        },
        "answer_key_coverage": {
            "semantically_scored_rows": int(result["semantic_similarity"].notna().sum()),
            "flagged_rows": ak_flagged,
            "flagged_rate_all_rows": ak_flagged / max(rows, 1),
            "flagged_rate_eligible_rows": ak_flagged / max(int(ak_eligible.sum()), 1),
            "label_counts": _value_counts(result["ak_coverage_label"]),
            "conflict_rows": int(result["ak_conflict_detected"].sum()),
            "conflict_reason_counts": _value_counts(
                result.loc[result["ak_conflict_reasons"].ne(""), "ak_conflict_reasons"]
            ),
            "semantic_similarity_summary": _numeric_summary(result["semantic_similarity"]),
            "semantic_surface_gap_summary": _numeric_summary(result["semantic_surface_gap"]),
        },
        "combined_review_queue": {
            "flagged_rows": combined_flagged,
            "flagged_rate": combined_flagged / max(rows, 1),
            "review_reason_counts": _value_counts(result["review_reasons"]),
        },
        "feature_columns": list(artifact.feature_columns),
    }


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
    result["ocr_review_probability"] = probability
    result["requires_ocr_review"] = probability >= threshold
    result["ocr_risk_label"] = np.where(probability >= max(.8, threshold), "high",
                                np.where(probability >= threshold, "medium", "low"))
    if artifact.type_model is not None:
        type_columns = _columns_expected_by_model(
            artifact.type_model, artifact.feature_columns, features.columns
        )
        if hasattr(artifact.type_model, "predict_constrained"):
            result["predicted_ocr_correction_type"] = artifact.type_model.predict_constrained(
                features[type_columns], features["is_mcq"]
            )
        else:
            # Compatibility for any older classifier implementation.
            result["predicted_ocr_correction_type"] = artifact.type_model.predict(features[type_columns])
            impossible = (
                features["is_mcq"].eq(0)
                & result["predicted_ocr_correction_type"].eq("mcq_correction")
            )
            result.loc[impossible, "predicted_ocr_correction_type"] = "multiple_edits"
        result.loc[~result["requires_ocr_review"], "predicted_ocr_correction_type"] = "no_correction"
    result["ocr_review_threshold"] = threshold

    if args.skip_ak_coverage:
        coverage = pd.DataFrame({
            "semantic_similarity": np.full(len(data), np.nan),
            "best_semantic_variation": pd.Series([None] * len(data), dtype=object),
            "semantic_surface_gap": np.full(len(data), np.nan),
            "ak_required_semantic_threshold": np.full(len(data), np.nan),
            "ak_conflict_detected": np.zeros(len(data), dtype=bool),
            "ak_conflict_reasons": pd.Series([""] * len(data), dtype=object),
            "possible_gap_suggestion": np.zeros(len(data), dtype=bool),
            "requires_ak_review": np.zeros(len(data), dtype=bool),
            "ak_coverage_label": pd.Series(["semantic_not_scored"] * len(data), dtype=object),
            "ak_review_suggestion": pd.Series(["run_with_semantic_model"] * len(data), dtype=object),
        })
    else:
        encoder = SemanticEncoder(args.semantic_model, args.semantic_batch_size)
        coverage = make_ak_coverage_decisions(
            data, features, encoder, artifact.max_variations,
            semantic_threshold=args.ak_semantic_threshold,
            surface_threshold=args.ak_surface_threshold,
            min_semantic_surface_gap=args.ak_min_semantic_surface_gap,
            single_phrase_threshold=args.ak_single_phrase_threshold,
            multiword_threshold=args.ak_multiword_threshold,
            chunk_size=args.semantic_chunk_size,
        )
    for column in coverage:
        result[column] = coverage[column].to_numpy()

    result["requires_any_human_review"] = (
        result["requires_ocr_review"] | result["requires_ak_review"]
    )
    result["review_reasons"] = np.select(
        [
            result["requires_ocr_review"] & result["requires_ak_review"],
            result["requires_ocr_review"],
            result["requires_ak_review"],
        ],
        ["ocr_and_answer_key", "ocr_correction", "possible_answer_key_gap"],
        default="none",
    )

    # Backward-compatible aliases for existing downstream reports. New code
    # should use the explicit ocr_* names above.
    result["review_probability"] = result["ocr_review_probability"]
    result["requires_human_review"] = result["requires_ocr_review"]
    result["risk_label"] = result["ocr_risk_label"]
    if "predicted_ocr_correction_type" in result:
        result["predicted_correction_type"] = result["predicted_ocr_correction_type"]
    result["review_threshold"] = result["ocr_review_threshold"]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    report_out = args.report_out or args.output_csv.with_suffix(".scoring_report.json")
    report = build_scoring_report(args, artifact, features, result, threshold)
    report["files"]["report_out"] = str(report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"OCR review: {result['requires_ocr_review'].sum():,}; "
        f"AK review: {result['requires_ak_review'].sum():,}; "
        f"either: {result['requires_any_human_review'].sum():,}/{len(result):,}; "
        f"saved {args.output_csv} and {report_out}"
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train", help="train and validate on human-reviewed rows")
    fit.add_argument("--human-csv", type=Path, default=Path("data/hitl/human.csv"))
    fit.add_argument("--model-out", type=Path, default=Path("output/hitl/review_model.pkl"))
    fit.add_argument("--report-out", type=Path, default=Path("output/hitl/validation_report.json"))
    fit.add_argument("--group-column", default="scan_id")
    fit.add_argument("--validation-size", type=float, default=.2)
    fit.add_argument("--target-recall", type=float, default=.98)
    fit.add_argument("--max-variations", type=int, default=100)
    fit.add_argument("--sample", type=int, help="optional development sample size")
    fit.add_argument("--random-state", type=int, default=42)
    fit.set_defaults(func=train)
    apply = sub.add_parser("score", help="flag unreviewed rows")
    apply.add_argument("--model", type=Path, default=Path("output/hitl/review_model.pkl"))
    apply.add_argument("--input-csv", type=Path, default=Path("data/hitl/no_human.csv"))
    apply.add_argument("--output-csv", type=Path, default=Path("output/hitl/no_human_flagged.csv"))
    apply.add_argument("--report-out", type=Path,
                       help="default: OUTPUT_CSV with .scoring_report.json suffix")
    apply.add_argument("--evaluate-with-published", action="store_true",
                       help="calculate OCR recall/precision only when Published is verified truth")
    apply.add_argument("--threshold", type=float)
    apply.add_argument("--semantic-model", default=DEFAULT_SEMANTIC_MODEL,
                       help="sentence-transformer model name or local path")
    apply.add_argument("--skip-ak-coverage", action="store_true",
                       help="score OCR only and mark semantic coverage as not scored")
    apply.add_argument("--ak-semantic-threshold", type=float, default=.75)
    apply.add_argument("--ak-surface-threshold", type=float, default=.60)
    apply.add_argument("--ak-min-semantic-surface-gap", type=float, default=.15)
    apply.add_argument("--ak-single-phrase-threshold", type=float, default=.78)
    apply.add_argument("--ak-multiword-threshold", type=float, default=.80)
    apply.add_argument("--semantic-batch-size", type=int, default=64)
    apply.add_argument("--semantic-chunk-size", type=int, default=2000)
    apply.add_argument("--limit", type=int, help="optional smoke-test row limit")
    apply.set_defaults(func=score)
    return root


if __name__ == "__main__":
    options = parser().parse_args()
    options.func(options)
