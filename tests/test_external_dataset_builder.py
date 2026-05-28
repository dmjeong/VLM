from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from scripts.data.build_external_vlm_10k_dataset import (
    BuildState,
    build_mmbench_prompt,
    clean_question,
    conversation_pairs,
    iter_json_array_objects,
    option_answer,
    qa_questions,
    sample_from_mmbench_row,
    samples_from_pairs,
    fill_from_mmbench,
)


class ExternalDatasetBuilderTest(unittest.TestCase):
    def test_conversation_pairs_extracts_human_gpt_turns(self) -> None:
        pairs = conversation_pairs(
            [
                {"from": "human", "value": "<image>\nWhat is shown?"},
                {"from": "gpt", "value": "A kitchen is shown."},
            ]
        )

        self.assertEqual(pairs, [("What is shown?", "A kitchen is shown.")])

    def test_clean_question_falls_back_when_only_image_token_exists(self) -> None:
        self.assertEqual(clean_question("<image>"), "Describe this image.")

    def test_samples_from_pairs_expands_long_caption_prompts(self) -> None:
        answer = " ".join(["detail"] * 40)
        rows = samples_from_pairs(
            pairs=[("Describe.\n<image>", answer)],
            image="images/coco/train2017/000.jpg",
            source="sharegpt4v",
            source_id="id",
            split="train",
            caption_prompt_variants=3,
            qa_prompt_variants=3,
            remaining=10,
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["metadata"]["source"], "sharegpt4v")

    def test_samples_from_pairs_expands_regular_qa_prompts(self) -> None:
        rows = samples_from_pairs(
            pairs=[("What is shown?", "A kitchen.")],
            image="images/coco/val2017/000.jpg",
            source="lvis-instruct4v",
            source_id="id",
            split="train",
            caption_prompt_variants=1,
            qa_prompt_variants=3,
            remaining=10,
        )

        self.assertEqual(len(rows), 3)
        self.assertIn("Based on the image", rows[1]["question"])

    def test_qa_questions_deduplicates_variants(self) -> None:
        questions = qa_questions("What is shown?", 2)

        self.assertEqual(len(questions), 2)

    def test_qa_questions_avoids_duplicate_based_on_prefix(self) -> None:
        questions = qa_questions("Based on the image, what is shown?", 3)

        self.assertNotIn("based on the image, based on the image", " ".join(questions).lower())

    def test_mmbench_prompt_and_answer_include_options(self) -> None:
        row = {"question": "Pick one.", "hint": "Look carefully.", "A": "Cat", "B": "Dog", "answer": "B"}

        self.assertIn("Options:", build_mmbench_prompt(row))
        self.assertEqual(option_answer(row, "B"), "B. Dog")

    def test_sample_from_mmbench_row_decodes_image(self) -> None:
        tiny_jpeg = base64.b64encode(b"\xff\xd8\xff\xe0fake").decode("ascii")
        row = {
            "index": "1",
            "question": "Pick one.",
            "A": "Cat",
            "B": "Dog",
            "answer": "B",
            "image": tiny_jpeg,
            "category": "reasoning",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            sample = sample_from_mmbench_row(row=row, split="validation", output_root=Path(temp_dir))

            self.assertIsNotNone(sample)
            assert sample is not None
            self.assertEqual(sample["answer"], "B. Dog")
            self.assertTrue((Path(temp_dir) / sample["image"]).exists())

    def test_iter_json_array_objects_streams_large_array_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "records.json"
            path.write_text('[{"id": 1}, {"id": 2}, {"id": 3}]', encoding="utf-8")

            rows = list(iter_json_array_objects(path))

            self.assertEqual([row["id"] for row in rows], [1, 2, 3])

    def test_fill_from_mmbench_quota_is_additive(self) -> None:
        tiny_jpeg = base64.b64encode(b"\xff\xd8\xff\xe0fake").decode("ascii")
        rows = [
            {"index": str(index), "question": "Pick one.", "A": "Cat", "B": "Dog", "answer": "B", "image": tiny_jpeg}
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            target_rows = [{"sample_id": "existing", "image": "existing.jpg", "question": "q", "answer": "a"}]
            state = BuildState(output_root=Path(temp_dir), rng=__import__("random").Random(42))

            fill_from_mmbench(target_rows=target_rows, rows=rows, quota=2, split="validation", state=state)

            self.assertEqual(len(target_rows), 3)


if __name__ == "__main__":
    unittest.main()
