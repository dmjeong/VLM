from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mini_vlm.evaluation.dashboard import build_image_groups, object_label_hits_generation, write_evaluation_dashboard
from mini_vlm.utils.checkpoints import write_json


class EvaluationDashboardTest(unittest.TestCase):
    def test_object_label_hit_uses_normalized_text(self) -> None:
        self.assertTrue(object_label_hits_generation("computer-mouse", "A computer mouse is visible."))
        self.assertFalse(object_label_hits_generation("airplane", "A cat."))

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
            self.assertIn("airplane", html)
            self.assertIn("A cat.", html)
            self.assertIn("dashboard_assets/images/", html)
            self.assertNotIn("file://", html)
            self.assertTrue(any((root / "dashboard_assets" / "images").iterdir()))


if __name__ == "__main__":
    unittest.main()
