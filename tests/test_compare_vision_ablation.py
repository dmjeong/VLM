from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mini_vlm.evaluation.compare_vision_ablation import ExperimentSpec, write_comparison
from mini_vlm.utils.checkpoints import write_json


class CompareVisionAblationTest(unittest.TestCase):
    def test_write_comparison_refreshes_dashboards_and_links_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            (image_dir / "sample.jpg").write_bytes(b"fake-image")

            dataset_path = root / "test.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "sample_id": "one",
                        "image": "images/sample.jpg",
                        "question": "Which season?",
                        "answer": "A. spring",
                        "task": "multiple-choice-vqa",
                        "metadata": {"category": "attribute_recognition", "source": "mmbench-dev-en"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            stage0 = root / "stage0"
            stage1 = root / "stage1"
            evaluation = stage1 / "evaluation"
            stage0.mkdir()
            evaluation.mkdir(parents=True)
            write_json({"epochs": [{"validation_loss": 1.2}]}, stage0 / "itc_summary.json")
            write_json({"epochs": [{"avg_loss": 2.0, "validation_loss": 1.5}]}, stage1 / "training_summary.json")
            write_json(
                {
                    "split": "test",
                    "dataset_path": str(dataset_path),
                    "checkpoint": str(stage1),
                    "loaded_files": {"visual_adapter": str(stage1 / "visual_adapter.pt")},
                    "sample_count": 1,
                    "generation_sample_count": 1,
                    "generation_unique_image_count": 1,
                    "avg_loss": 1.4,
                    "exact_match_rate": 0.0,
                    "contains_answer_rate": 0.0,
                    "avg_token_overlap": 1.0,
                },
                evaluation / "test_summary.json",
            )
            (evaluation / "test_predictions.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "one",
                        "image": "images/sample.jpg",
                        "question": "Which season?",
                        "expected_answer": "A. spring",
                        "generated_answer": "A\nThe picture was taken in spring.",
                        "exact_match": False,
                        "contains_answer": False,
                        "token_overlap": 1.0,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            config_path = root / "config.json"
            write_json(
                {
                    "experiment_name": "compare-test",
                    "test_jsonl": str(dataset_path),
                    "image_root": str(root),
                    "output_dir": str(stage1),
                },
                config_path,
            )

            paths = write_comparison(
                experiments=[
                    ExperimentSpec(
                        name="Demo",
                        stage1_dir=str(stage1),
                        stage0_dir=str(stage0),
                        config_path=str(config_path),
                    )
                ],
                output_dir=root / "comparison",
                refresh_dashboards=True,
            )

            report = paths.report_path.read_text(encoding="utf-8")
            comparison_dashboard = paths.dashboard_path.read_text(encoding="utf-8")
            row = json.loads(paths.json_path.read_text(encoding="utf-8"))[0]
            experiment_dashboard = evaluation / "dashboard.html"
            mirrored_dashboard = paths.dashboard_path.parent / "experiments" / "demo" / "dashboard.html"

            self.assertIn("통합 대시보드", report)
            self.assertIn("[열기](experiments/demo/dashboard.html)", report)
            self.assertIn('href="experiments/demo/dashboard.html"', comparison_dashboard)
            self.assertIn("Vision Encoder Ablation 비교", comparison_dashboard)
            self.assertTrue(experiment_dashboard.exists())
            self.assertTrue(mirrored_dashboard.exists())
            self.assertTrue((mirrored_dashboard.parent / "dashboard_assets").exists())
            self.assertIn("이미지별 평가 대시보드", experiment_dashboard.read_text(encoding="utf-8"))
            self.assertEqual(row["dashboard_status"], "업데이트")
            self.assertEqual(row["comparison_dashboard_path"], str(mirrored_dashboard))


if __name__ == "__main__":
    unittest.main()
