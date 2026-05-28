from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mini_vlm.reporting.training_report import build_epoch_rows, write_training_report


class TrainingReportTest(unittest.TestCase):
    def test_build_epoch_rows_uses_one_based_epoch(self) -> None:
        rows = build_epoch_rows({"epochs": [{"epoch": 0, "avg_loss": 2.0, "validation_loss": 3.0}]})

        self.assertEqual(rows[0]["epoch"], 1)
        self.assertEqual(rows[0]["train_avg_loss"], 2.0)
        self.assertEqual(rows[0]["validation_loss"], 3.0)

    def test_write_training_report_creates_csv_svg_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "training_summary.json").write_text(
                json.dumps(
                    {
                        "experiment_name": "unit-test",
                        "epoch_count": 2,
                        "sample_count": 4,
                        "batches_per_epoch": 2,
                        "final_avg_loss": 1.0,
                        "epochs": [
                            {"epoch": 0, "avg_loss": 2.0, "validation_loss": 2.5, "optimizer_step": 2},
                            {"epoch": 1, "avg_loss": 1.0, "validation_loss": 1.5, "optimizer_step": 4},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            paths = write_training_report(output_dir)

            self.assertTrue(paths.csv_path.exists())
            self.assertTrue(paths.svg_path.exists())
            self.assertTrue(paths.markdown_path.exists())
            self.assertIn("Loss curve", paths.svg_path.read_text(encoding="utf-8"))
            self.assertIn("best validation loss", paths.markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
