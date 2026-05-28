from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mini_vlm.data.dataset import MiniVlmDataset, MiniVlmSample, load_jsonl_samples


class MiniVlmDatasetTest(unittest.TestCase):
    def test_loads_jsonl_samples(self) -> None:
        samples = load_jsonl_samples("data/samples/train.jsonl")

        self.assertGreaterEqual(len(samples), 2)
        self.assertEqual(samples[0].sample_id, "sample-0001")
        self.assertEqual(samples[0].task, "caption")
        self.assertTrue(any(sample.sample_id.startswith("web-apple") for sample in samples))

    def test_dataset_validates_image_paths(self) -> None:
        dataset = MiniVlmDataset("data/samples/train.jsonl", image_root="data/samples")

        self.assertGreaterEqual(len(dataset), 2)
        self.assertTrue(dataset.image_path_for(dataset[0]).exists())

    def test_rejects_missing_required_fields(self) -> None:
        with self.assertRaises(ValueError):
            MiniVlmSample.from_dict({"image": "x.png", "question": "q"}, index=1)

    def test_missing_image_raises_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            annotation = root / "data.jsonl"
            annotation.write_text(
                '{"sample_id":"one","image":"missing.png","question":"q","answer":"a"}\n',
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                MiniVlmDataset(annotation, image_root=root)


if __name__ == "__main__":
    unittest.main()
