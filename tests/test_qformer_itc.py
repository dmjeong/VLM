from __future__ import annotations

import importlib.util
import unittest

from mini_vlm.data.dataset import MiniVlmSample
from mini_vlm.training.pretrain_qformer_itc import build_itc_text, should_log_itc_step, summarize_itc_epoch


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class QFormerItcHelperTest(unittest.TestCase):
    def test_caption_sample_uses_answer_as_itc_text(self) -> None:
        sample = MiniVlmSample(
            sample_id="caption",
            image="image.jpg",
            question="Describe this image.",
            answer="A red apple on a table.",
            task="caption",
        )

        self.assertEqual(build_itc_text(sample), "A red apple on a table.")

    def test_vqa_sample_combines_question_and_answer_for_itc_text(self) -> None:
        sample = MiniVlmSample(
            sample_id="vqa",
            image="image.jpg",
            question="What fruit is shown?",
            answer="An apple is shown.",
            task="vqa",
        )

        self.assertEqual(build_itc_text(sample), "What fruit is shown? An apple is shown.")

    def test_itc_epoch_summary_tracks_validation(self) -> None:
        summary = summarize_itc_epoch(
            epoch=0,
            losses=[1.5, 1.0],
            skipped_batches=1,
            validation_summary={"avg_loss": 1.25, "batch_count": 2, "skipped_batches": 0},
        )

        self.assertEqual(summary["loss_decrease"], 0.5)
        self.assertEqual(summary["skipped_batches"], 1)
        self.assertEqual(summary["validation_loss"], 1.25)

    def test_itc_progress_logging_is_dense_for_small_epochs(self) -> None:
        self.assertTrue(should_log_itc_step(batch_index=0, batch_count=5))
        self.assertTrue(should_log_itc_step(batch_index=4, batch_count=5))


@unittest.skipUnless(TORCH_AVAILABLE, "torch가 설치된 환경에서만 ITC tensor 테스트를 실행합니다.")
class QFormerItcTensorTest(unittest.TestCase):
    def test_masked_mean_pool_ignores_padding(self) -> None:
        import torch

        from mini_vlm.models.qformer import masked_mean_pool

        hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
        mask = torch.tensor([[1, 1, 0]])

        pooled = masked_mean_pool(hidden, mask)

        self.assertTrue(torch.allclose(pooled, torch.tensor([[2.0, 4.0]])))

    def test_symmetric_itc_loss_rewards_matching_diagonal(self) -> None:
        import torch

        from mini_vlm.models.qformer import symmetric_itc_loss

        good_logits = torch.tensor([[8.0, 0.1], [0.2, 8.0]])
        bad_logits = torch.tensor([[0.1, 8.0], [8.0, 0.2]])

        good_loss = symmetric_itc_loss(logits_per_image=good_logits, logits_per_text=good_logits.t())
        bad_loss = symmetric_itc_loss(logits_per_image=bad_logits, logits_per_text=bad_logits.t())

        self.assertLess(float(good_loss), float(bad_loss))

    def test_initialize_qformer_from_distilbert_copies_self_attention_weights(self) -> None:
        import torch
        from torch import nn

        from mini_vlm.models.qformer import initialize_qformer_from_distilbert
        from mini_vlm.models.visual_adapter import QFormerVisualAdapter

        class FakeAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_lin = nn.Linear(8, 8)
                self.k_lin = nn.Linear(8, 8)
                self.v_lin = nn.Linear(8, 8)
                self.out_lin = nn.Linear(8, 8)

        class FakeFfn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin1 = nn.Linear(8, 32)
                self.lin2 = nn.Linear(32, 8)

        class FakeSourceLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attention = FakeAttention()
                self.sa_layer_norm = nn.LayerNorm(8)
                self.ffn = FakeFfn()
                self.output_layer_norm = nn.LayerNorm(8)

        class FakeTextEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer = nn.Module()
                self.transformer.layer = nn.ModuleList([FakeSourceLayer()])

        adapter = QFormerVisualAdapter(vision_dim=8, llm_dim=12, visual_token_count=2, hidden_dim=8, layer_count=1)
        text_encoder = FakeTextEncoder()
        with torch.no_grad():
            text_encoder.transformer.layer[0].attention.q_lin.weight.fill_(0.25)

        initialized = initialize_qformer_from_distilbert(adapter, text_encoder)

        self.assertEqual(initialized, 1)
        self.assertTrue(torch.allclose(adapter.layers[0].self_attention.in_proj_weight[:8], torch.full((8, 8), 0.25)))


if __name__ == "__main__":
    unittest.main()
