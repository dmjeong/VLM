from __future__ import annotations

import unittest

from mini_vlm.config import MiniVlmConfig
from mini_vlm.evaluation.evaluate_cli import (
    normalize_answer,
    resolve_split_path,
    score_generation,
    select_generation_indices,
)


class EvaluationHelperTest(unittest.TestCase):
    def test_resolve_split_path_uses_config_test_jsonl(self) -> None:
        config = MiniVlmConfig(test_jsonl="data/test.jsonl")

        self.assertEqual(resolve_split_path(config, split="test"), "data/test.jsonl")

    def test_resolve_split_path_allows_override(self) -> None:
        config = MiniVlmConfig(test_jsonl="")

        self.assertEqual(resolve_split_path(config, split="test", override_jsonl="custom.jsonl"), "custom.jsonl")

    def test_normalize_answer_removes_case_and_punctuation(self) -> None:
        self.assertEqual(normalize_answer(" A Red-Bus! "), "a red bus")

    def test_score_generation_tracks_contains_and_overlap(self) -> None:
        score = score_generation("This image shows a red bus on a road.", "red bus")

        self.assertTrue(score["contains_answer"])
        self.assertFalse(score["exact_match"])
        self.assertEqual(score["token_overlap"], 1.0)

    def test_select_generation_indices_can_sample_evenly(self) -> None:
        self.assertEqual(
            select_generation_indices(dataset_length=1000, max_samples=5, sampling="even"),
            [0, 250, 500, 749, 999],
        )

    def test_select_generation_indices_keeps_first_mode_for_debugging(self) -> None:
        self.assertEqual(
            select_generation_indices(dataset_length=1000, max_samples=5, sampling="first"),
            [0, 1, 2, 3, 4],
        )


if __name__ == "__main__":
    unittest.main()
