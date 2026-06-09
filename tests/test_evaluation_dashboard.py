from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mini_vlm.evaluation.dashboard import (
    assess_prediction_answer,
    build_image_groups,
    choice_letter_match,
    object_label_hits_generation,
    prediction_success,
    write_evaluation_dashboard,
)
from mini_vlm.utils.checkpoints import write_json


class EvaluationDashboardTest(unittest.TestCase):
    def test_object_label_hit_uses_normalized_text(self) -> None:
        self.assertTrue(object_label_hits_generation("computer-mouse", "A computer mouse is visible."))
        self.assertFalse(object_label_hits_generation("airplane", "A cat."))

    def test_choice_letter_match_reads_option_prefix(self) -> None:
        self.assertTrue(choice_letter_match("A. spring", "A\n\nThe picture was taken in spring."))
        self.assertFalse(choice_letter_match("D. winter", "A\n\nThe picture was taken in spring."))
        self.assertIsNone(choice_letter_match("The image shows a bus.", "A bus."))

    def test_yes_no_paraphrase_counts_as_answer_match(self) -> None:
        prediction = {
            "question": "Are there any humans or animals in the image?",
            "expected_answer": "From what I can see, there aren't any humans or animals in the image.",
            "generated_answer": "No, there are no humans or animals in the image.",
            "exact_match": False,
            "contains_answer": False,
            "token_overlap": 0.4667,
        }

        assessed = assess_prediction_answer(prediction)

        self.assertTrue(assessed["answer_match"])
        self.assertTrue(assessed["yes_no_match"])
        self.assertEqual(assessed["answer_match_reason"], "yes-no")

    def test_yes_no_contradiction_fails_even_with_high_overlap(self) -> None:
        prediction = {
            "question": "Are there any people or animals in the photo?",
            "expected_answer": "No, there are no people or animals visible in the photo.",
            "generated_answer": "Yes, there are people and animals in the photo.",
            "exact_match": False,
            "contains_answer": False,
            "token_overlap": 0.8,
        }

        assessed = assess_prediction_answer(prediction)

        self.assertFalse(assessed["answer_match"])
        self.assertFalse(assessed["yes_no_match"])

    def test_count_mismatch_fails_even_with_high_overlap(self) -> None:
        prediction = {
            "question": "How many magnets are visible?",
            "expected_answer": "There are two magnets visible.",
            "generated_answer": "There are four magnets visible.",
            "exact_match": False,
            "contains_answer": False,
            "token_overlap": 0.75,
        }

        assessed = assess_prediction_answer(prediction)

        self.assertFalse(assessed["answer_match"])
        self.assertFalse(assessed["count_match"])
        self.assertFalse(prediction_success(prediction))

    def test_short_fact_mismatch_does_not_pass_on_overlap_only(self) -> None:
        prediction = {
            "question": "Based on the image, what brand is the bus?",
            "expected_answer": "The bus is a Volvo.",
            "generated_answer": "The bus is a Tesla Model 3.",
            "exact_match": False,
            "contains_answer": False,
            "token_overlap": 0.8,
        }

        assessed = assess_prediction_answer(prediction)

        self.assertFalse(assessed["answer_match"])
        self.assertEqual(assessed["answer_match_reason"], "short-fact")

    def test_short_fact_uses_question_to_find_core_answer(self) -> None:
        prediction = {
            "question": "Based on the image, what are they using to cut the cake?",
            "expected_answer": "They are using a knife to cut the cake.",
            "generated_answer": "Knife.",
            "exact_match": False,
            "contains_answer": False,
            "token_overlap": 0.1111,
        }

        assessed = assess_prediction_answer(prediction)

        self.assertTrue(assessed["answer_match"])
        self.assertEqual(assessed["answer_match_reason"], "short-fact")

    def test_short_fact_accepts_supported_short_synonym(self) -> None:
        prediction = {
            "question": "Look at the image and answer: What about the bride? How is she dressed?",
            "expected_answer": "The bride is dressed in a bridal gown.",
            "generated_answer": "wedding dress",
            "exact_match": False,
            "contains_answer": False,
            "token_overlap": 0.0,
        }

        assessed = assess_prediction_answer(prediction)

        self.assertTrue(assessed["answer_match"])
        self.assertEqual(assessed["answer_match_reason"], "short-fact")

    def test_short_fact_accepts_short_answer_inside_long_position_answer(self) -> None:
        prediction = {
            "question": "Where is the tennis ball located?",
            "expected_answer": "The tennis ball is located in the air, positioned slightly above and to the right of the tennis player.",
            "generated_answer": "in air",
            "exact_match": False,
            "contains_answer": False,
            "token_overlap": 0.1,
        }

        assessed = assess_prediction_answer(prediction)

        self.assertTrue(assessed["answer_match"])
        self.assertEqual(assessed["answer_match_reason"], "short-fact")

    def test_yes_no_accepts_descriptive_positive_answer(self) -> None:
        prediction = {
            "question": "Based on the image, is there any fruit or vegetable that appears to be the largest among the rest?",
            "expected_answer": "The bell peppers appear to be the largest among the rest of the fruits and vegetables in the image.",
            "generated_answer": "yes",
            "exact_match": False,
            "contains_answer": False,
            "token_overlap": 0.0,
        }

        assessed = assess_prediction_answer(prediction)

        self.assertTrue(assessed["answer_match"])
        self.assertTrue(assessed["yes_no_match"])

    def test_yes_no_accepts_not_exactly_as_negative(self) -> None:
        prediction = {
            "question": "Look at the image and answer: Are the candles evenly spread around the cake?",
            "expected_answer": "Not exactly. Some candles are grouped closer together while others are slightly farther apart.",
            "generated_answer": "no",
            "exact_match": False,
            "contains_answer": False,
            "token_overlap": 0.0,
        }

        assessed = assess_prediction_answer(prediction)

        self.assertTrue(assessed["answer_match"])
        self.assertTrue(assessed["yes_no_match"])

    def test_short_fact_accepts_answer_choice_from_question(self) -> None:
        prediction = {
            "question": "Are the bananas placed together or scattered in the image?",
            "expected_answer": "The bananas are scattered in the image.",
            "generated_answer": "scattered",
            "exact_match": False,
            "contains_answer": False,
            "token_overlap": 0.0,
        }

        assessed = assess_prediction_answer(prediction)

        self.assertTrue(assessed["answer_match"])
        self.assertEqual(assessed["answer_match_reason"], "short-fact")

    def test_alternative_question_does_not_accept_plain_yes(self) -> None:
        prediction = {
            "question": "Are the bananas placed together or scattered in the image?",
            "expected_answer": "The bananas are scattered in the image.",
            "generated_answer": "yes",
            "exact_match": False,
            "contains_answer": False,
            "token_overlap": 0.0,
        }

        assessed = assess_prediction_answer(prediction)

        self.assertFalse(assessed["answer_match"])
        self.assertNotEqual(assessed["answer_match_reason"], "yes-no")

    def test_build_image_groups_marks_problematic_image(self) -> None:
        predictions = [
            {
                "sample_id": "one",
                "image": "images/airplane/a.jpg",
                "question": "What object?",
                "expected_answer": "Airplane.",
                "generated_answer": "A cat.",
                "exact_match": False,
                "contains_answer": False,
                "token_overlap": 0.0,
            }
        ]
        samples = {
            "one": {
                "sample_id": "one",
                "image": "images/airplane/a.jpg",
                "metadata": {"object": "airplane", "category": "vehicle", "file_title": "Airplane"},
            }
        }

        groups = build_image_groups(predictions=predictions, samples_by_id=samples, image_root=Path("/tmp"))

        self.assertEqual(groups[0].object_label, "airplane")
        self.assertEqual(groups[0].object_hit_rate, 0.0)
        self.assertEqual(groups[0].severity, "심각")

    def test_build_image_groups_uses_category_when_object_missing(self) -> None:
        predictions = [
            {
                "sample_id": "one",
                "image": "images/mmbench/a.jpg",
                "question": "Which season?",
                "expected_answer": "A. spring",
                "generated_answer": "A\nThe picture was taken in spring.",
                "exact_match": False,
                "contains_answer": False,
                "token_overlap": 1.0,
            }
        ]
        samples = {
            "one": {
                "sample_id": "one",
                "image": "images/mmbench/a.jpg",
                "task": "multiple-choice-vqa",
                "metadata": {"category": "attribute_recognition", "source": "mmbench-dev-en"},
            }
        }

        groups = build_image_groups(predictions=predictions, samples_by_id=samples, image_root=Path("/tmp"))

        self.assertEqual(groups[0].object_label, "attribute_recognition")
        self.assertEqual(groups[0].label_kind, "category")
        self.assertIsNone(groups[0].object_hit_rate)
        self.assertEqual(groups[0].choice_letter_rate, 1.0)
        self.assertEqual(groups[0].answer_match_rate, 1.0)
        self.assertEqual(groups[0].severity, "양호")

    def test_build_image_groups_uses_answer_match_for_semantic_status(self) -> None:
        predictions = [
            {
                "sample_id": "one",
                "image": "images/coco/a.jpg",
                "question": "Based on the image, is there any water source or pond in the image?",
                "expected_answer": "No, there isn't any visible water source or pond in the image.",
                "generated_answer": "No, there is no water source or pond in the image.",
                "exact_match": False,
                "contains_answer": False,
                "token_overlap": 0.6923,
            }
        ]
        samples = {
            "one": {
                "sample_id": "one",
                "image": "images/coco/a.jpg",
                "task": "visual-question-answering",
                "metadata": {"category": "visual-question-answering", "source": "lvis-instruct4v"},
            }
        }

        groups = build_image_groups(predictions=predictions, samples_by_id=samples, image_root=Path("/tmp"))

        self.assertEqual(groups[0].answer_match_rate, 1.0)
        self.assertEqual(groups[0].severity, "양호")
        self.assertNotIn("완전 일치 답변이 없음", groups[0].issues)

    def test_choice_letter_match_with_conflicting_text_is_warning(self) -> None:
        predictions = [
            {
                "sample_id": "one",
                "image": "images/mmbench/a.jpg",
                "question": "Which city is shown?",
                "expected_answer": "A. Xi'an",
                "generated_answer": "A. Beijing appears to be shown.",
                "exact_match": False,
                "contains_answer": False,
                "token_overlap": 0.0,
            }
        ]
        samples = {
            "one": {
                "sample_id": "one",
                "image": "images/mmbench/a.jpg",
                "task": "multiple-choice-vqa",
                "metadata": {"category": "landmark_recognition", "source": "mmbench-dev-en"},
            }
        }

        groups = build_image_groups(predictions=predictions, samples_by_id=samples, image_root=Path("/tmp"))

        self.assertEqual(groups[0].answer_match_rate, 1.0)
        self.assertEqual(groups[0].choice_letter_rate, 1.0)
        self.assertEqual(groups[0].severity, "주의")
        self.assertIn("선택지 letter는 맞았지만 생성 설명이 정답 텍스트와 어긋나는 행이 있음", groups[0].issues)

    def test_write_evaluation_dashboard_creates_html(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            image_path = image_dir / "a.jpg"
            image_path.write_bytes(b"not-a-real-image-but-path-exists")
            summary_path = root / "summary.json"
            predictions_path = root / "predictions.jsonl"
            dataset_path = root / "test.jsonl"
            write_json(
                {
                    "split": "test",
                    "dataset_path": str(dataset_path),
                    "checkpoint": "checkpoint",
                    "sample_count": 1,
                    "generation_sample_count": 1,
                    "avg_loss": 1.0,
                    "exact_match_rate": 0.0,
                    "contains_answer_rate": 0.0,
                    "avg_token_overlap": 0.0,
                },
                summary_path,
            )
            predictions_path.write_text(
                '{"sample_id":"one","image":"images/a.jpg","question":"q","expected_answer":"Airplane.","generated_answer":"A cat.","exact_match":false,"contains_answer":false,"token_overlap":0.0}\n',
                encoding="utf-8",
            )
            dataset_path.write_text(
                '{"sample_id":"one","image":"images/a.jpg","question":"q","answer":"Airplane.","metadata":{"object":"airplane","category":"vehicle","file_title":"Airplane"}}\n',
                encoding="utf-8",
            )

            dashboard_path = write_evaluation_dashboard(
                summary_path=summary_path,
                predictions_path=predictions_path,
                dataset_path=dataset_path,
                image_root=root,
                output_path=root / "dashboard.html",
            )

            html = dashboard_path.read_text(encoding="utf-8")
            self.assertIn("이미지별 평가 대시보드", html)
            self.assertIn("가장 취약한 이미지 Top 5", html)
            self.assertIn("전체 테스트 이미지", html)
            self.assertIn("1 / 1 images", html)
            self.assertIn("airplane", html)
            self.assertIn("A cat.", html)
            self.assertIn("dashboard_assets/images/", html)
            self.assertNotIn("file://", html)
            self.assertNotIn("unknown", html)
            self.assertTrue(any((root / "dashboard_assets" / "images").iterdir()))


if __name__ == "__main__":
    unittest.main()
