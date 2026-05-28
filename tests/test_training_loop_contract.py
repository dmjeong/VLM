from __future__ import annotations

import unittest

from mini_vlm.training.train_alignment import (
    format_training_progress,
    has_pending_optimizer_step,
    normalize_gradient_accumulation_steps,
    should_log_training_step,
    should_step_optimizer,
    summarize_epoch_losses,
)
from mini_vlm.training.train_instruction import build_instruction_training_plan
from mini_vlm.config import MiniVlmConfig


class TrainingLoopContractTest(unittest.TestCase):
    def test_tiny_epoch_keeps_pending_optimizer_step(self) -> None:
        self.assertFalse(should_step_optimizer(batch_index=0, accumulation_steps=8))
        self.assertTrue(has_pending_optimizer_step(batch_count=1, accumulation_steps=8))

    def test_accumulation_boundary_steps_optimizer(self) -> None:
        self.assertTrue(should_step_optimizer(batch_index=7, accumulation_steps=8))
        self.assertFalse(has_pending_optimizer_step(batch_count=8, accumulation_steps=8))

    def test_accumulation_steps_are_never_below_one(self) -> None:
        self.assertEqual(normalize_gradient_accumulation_steps(0), 1)
        self.assertEqual(normalize_gradient_accumulation_steps(-3), 1)

    def test_epoch_loss_summary_tracks_change(self) -> None:
        summary = summarize_epoch_losses(epoch=0, losses=[3.0, 2.0, 1.5], optimizer_step=3)

        self.assertEqual(summary["batch_count"], 3)
        self.assertEqual(summary["loss_change"], -1.5)
        self.assertEqual(summary["loss_decrease"], 1.5)
        self.assertEqual(summary["avg_loss"], 2.1666666666666665)

    def test_epoch_loss_summary_rejects_nan(self) -> None:
        with self.assertRaises(FloatingPointError):
            summarize_epoch_losses(epoch=0, losses=[3.0, float("nan")], optimizer_step=1)

    def test_progress_logging_is_dense_for_small_epochs(self) -> None:
        self.assertTrue(should_log_training_step(batch_index=0, batch_count=17))
        self.assertTrue(should_log_training_step(batch_index=16, batch_count=17))

    def test_progress_line_contains_percent_and_loss(self) -> None:
        line = format_training_progress(
            epoch=0,
            epoch_count=10,
            batch_index=4,
            batch_count=20,
            global_step=5,
            optimizer_step=5,
            loss=1.25,
            running_loss=1.5,
        )

        self.assertIn("epoch 1/10", line)
        self.assertIn("25.0%", line)
        self.assertIn("loss=1.2500", line)


class InstructionTrainingPlanTest(unittest.TestCase):
    def test_stage_two_plan_is_gated_after_stage_one(self) -> None:
        plan = build_instruction_training_plan(MiniVlmConfig(use_lora=True))

        self.assertTrue(plan.adapter_checkpoint_required)
        self.assertIn("visual_adapter", plan.trainable_parts)
        self.assertIn("llm_lora", plan.trainable_parts)
        self.assertIn("Stage 1", plan.blocked_reason)


if __name__ == "__main__":
    unittest.main()
