from __future__ import annotations

import unittest

from mini_vlm.data.collator import MiniVlmCollator
from mini_vlm.data.dataset import MiniVlmSample


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [max(3, (ord(char) % 50) + 3) for char in text]


class MiniVlmCollatorTest(unittest.TestCase):
    def test_masks_prompt_tokens_and_keeps_answer_labels(self) -> None:
        sample = MiniVlmSample(
            sample_id="sample",
            image="images/sample-grid.ppm",
            question="Q?",
            answer="A",
        )
        collator = MiniVlmCollator(FakeTokenizer(), image_root="data/samples")

        input_ids, labels = collator.build_text_features(sample)

        self.assertEqual(len(input_ids), len(labels))
        self.assertIn(-100, labels)
        self.assertEqual(labels[-1], FakeTokenizer.eos_token_id)

    def test_collates_batch_without_torch_dependency(self) -> None:
        sample = MiniVlmSample(
            sample_id="sample",
            image="images/sample-grid.ppm",
            question="Q?",
            answer="A",
        )
        collator = MiniVlmCollator(FakeTokenizer(), image_root="data/samples")

        batch = collator([sample])

        self.assertEqual(batch.sample_ids, ["sample"])
        self.assertEqual(batch.image_paths, ["data/samples/images/sample-grid.ppm"])
        self.assertEqual(len(batch.input_ids), 1)


if __name__ == "__main__":
    unittest.main()
