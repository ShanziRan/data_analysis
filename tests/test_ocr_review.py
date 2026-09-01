import unittest

import numpy as np
import pandas as pd

from ocr_review import make_ak_coverage_decisions, row_features


class FakeSemanticEncoder:
    """Small deterministic encoder for testing decision logic without downloads."""

    vectors = {
        "the day after wednesday": [1.0, 0.0, 0.0],
        "thursday": [0.9, 0.1, 0.0],
        "unrelated response": [1.0, 0.0, 0.0],
        "thursday morning": [0.0, 0.9, 0.1],
        "cat": [0.7, 0.7, 0.0],
        "dog": [0.71, 0.69, 0.0],
        "forty pounds": [0.6, 0.6, 0.2],
        "fourteen pounds": [0.61, 0.59, 0.2],
        "the light is on": [0.5, 0.5, 0.5],
        "the light is not on": [0.51, 0.49, 0.5],
    }

    def encode(self, texts):
        values = np.asarray([self.vectors[text] for text in texts], dtype=float)
        return values / np.linalg.norm(values, axis=1, keepdims=True)


class AnswerKeyCoverageTests(unittest.TestCase):
    def _features(self, frame):
        return pd.DataFrame([
            row_features(row, {}, max_variations=20)
            for _, row in frame.iterrows()
        ])

    def test_semantically_close_but_textually_different_answer_is_flagged(self):
        frame = pd.DataFrame([{
            "Captured": "the day after Wednesday",
            "ANSWER Key": "Thursday",
            "confidence": 0.99,
        }])
        decisions = make_ak_coverage_decisions(
            frame, self._features(frame), FakeSemanticEncoder(), 20,
            semantic_threshold=.80, surface_threshold=.60,
        )

        self.assertTrue(bool(decisions.loc[0, "requires_ak_review"]))
        self.assertEqual(decisions.loc[0, "ak_coverage_label"], "possible_answer_key_gap")
        self.assertEqual(decisions.loc[0, "best_semantic_variation"], "thursday")

    def test_low_semantic_similarity_is_not_flagged(self):
        frame = pd.DataFrame([{
            "Captured": "unrelated response",
            "ANSWER Key": "Thursday morning",
            "confidence": 0.99,
        }])
        decisions = make_ak_coverage_decisions(
            frame, self._features(frame), FakeSemanticEncoder(), 20,
            semantic_threshold=.80, surface_threshold=.60,
        )

        self.assertFalse(bool(decisions.loc[0, "requires_ak_review"]))
        self.assertEqual(decisions.loc[0, "ak_coverage_label"], "semantic_below_threshold")

    def test_blank_non_mcq_is_not_semantically_scored(self):
        frame = pd.DataFrame([{
            "Captured": "--blank--",
            "ANSWER Key": "Thursday",
            "confidence": 0.4,
        }])
        decisions = make_ak_coverage_decisions(
            frame, self._features(frame), FakeSemanticEncoder(), 20,
        )

        self.assertFalse(bool(decisions.loc[0, "requires_ak_review"]))
        self.assertTrue(np.isnan(decisions.loc[0, "semantic_similarity"]))

    def test_related_single_words_are_blocked(self):
        frame = pd.DataFrame([{
            "Captured": "cat", "ANSWER Key": "dog", "confidence": 0.99,
        }])
        decisions = make_ak_coverage_decisions(
            frame, self._features(frame), FakeSemanticEncoder(), 20,
        )
        self.assertFalse(bool(decisions.loc[0, "possible_gap_suggestion"]))
        self.assertEqual(decisions.loc[0, "ak_coverage_label"], "blocked_by_conflict")
        self.assertIn("single_token_substitution", decisions.loc[0, "ak_conflict_reasons"])

    def test_number_conflict_is_blocked(self):
        frame = pd.DataFrame([{
            "Captured": "forty pounds", "ANSWER Key": "fourteen pounds", "confidence": 0.99,
        }])
        decisions = make_ak_coverage_decisions(
            frame, self._features(frame), FakeSemanticEncoder(), 20,
        )
        self.assertFalse(bool(decisions.loc[0, "possible_gap_suggestion"]))
        self.assertIn("number_mismatch", decisions.loc[0, "ak_conflict_reasons"])

    def test_negation_conflict_is_blocked(self):
        frame = pd.DataFrame([{
            "Captured": "the light is on",
            "ANSWER Key": "the light is not on",
            "confidence": 0.99,
        }])
        decisions = make_ak_coverage_decisions(
            frame, self._features(frame), FakeSemanticEncoder(), 20,
        )
        self.assertFalse(bool(decisions.loc[0, "possible_gap_suggestion"]))
        self.assertIn("negation_mismatch", decisions.loc[0, "ak_conflict_reasons"])


if __name__ == "__main__":
    unittest.main()
