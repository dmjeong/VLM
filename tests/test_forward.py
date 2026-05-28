from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch가 설치된 환경에서만 MiniVLM forward 테스트를 실행합니다.")
class MiniVlmForwardTest(unittest.TestCase):
    def test_forward_prepends_visual_tokens(self) -> None:
        import torch
        from torch import nn

        from mini_vlm.models.mini_vlm import MiniVlmForConditionalGeneration
        from mini_vlm.models.vision_encoder import VisionFeatures
        from mini_vlm.models.visual_adapter import MlpVisualAdapter

        class FakeVisionEncoder(nn.Module):
            def forward(self, pixel_values):
                batch_size = pixel_values.shape[0]
                return VisionFeatures(
                    cls_token=torch.zeros(batch_size, 8),
                    patch_tokens=torch.randn(batch_size, 6, 8),
                )

        class FakeLlm(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(32, 12)
                self.head = nn.Linear(12, 32)

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, *, inputs_embeds, attention_mask, labels=None):
                logits = self.head(inputs_embeds)
                loss = logits.mean()
                return SimpleNamespace(loss=loss, logits=logits)

        model = MiniVlmForConditionalGeneration(
            vision_encoder=FakeVisionEncoder(),
            visual_adapter=MlpVisualAdapter(vision_dim=8, llm_dim=12, visual_token_count=4, hidden_dim=16),
            llm=FakeLlm(),
            visual_token_count=4,
            freeze_llm=False,
        )

        output = model(
            pixel_values=torch.randn(2, 3, 4, 4),
            input_ids=torch.tensor([[1, 2, 3], [1, 4, 0]]),
            attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
            labels=torch.tensor([[-100, 2, 3], [-100, 4, -100]]),
        )

        self.assertEqual(tuple(output.visual_tokens.shape), (2, 4, 12))
        self.assertEqual(tuple(output.inputs_embeds.shape), (2, 7, 12))
        self.assertEqual(tuple(output.attention_mask.shape), (2, 7))
        self.assertEqual(tuple(output.labels.shape), (2, 7))
        self.assertTrue((output.labels[:, :4] == -100).all())

    def test_forward_casts_visual_tokens_to_llm_embedding_dtype(self) -> None:
        import torch
        from torch import nn

        from mini_vlm.models.mini_vlm import MiniVlmForConditionalGeneration
        from mini_vlm.models.vision_encoder import VisionFeatures
        from mini_vlm.models.visual_adapter import MlpVisualAdapter

        class FakeVisionEncoder(nn.Module):
            def forward(self, pixel_values):
                return VisionFeatures(cls_token=torch.zeros(1, 8), patch_tokens=torch.randn(1, 6, 8))

        class FakeLlm(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(32, 12).to(dtype=torch.bfloat16)

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, *, inputs_embeds, attention_mask, labels=None):
                return SimpleNamespace(loss=inputs_embeds.float().mean(), logits=inputs_embeds.float())

        model = MiniVlmForConditionalGeneration(
            vision_encoder=FakeVisionEncoder(),
            visual_adapter=MlpVisualAdapter(vision_dim=8, llm_dim=12, visual_token_count=4, hidden_dim=16),
            llm=FakeLlm(),
            visual_token_count=4,
            freeze_llm=False,
        )

        output = model(
            pixel_values=torch.randn(1, 3, 4, 4),
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        )

        self.assertEqual(output.inputs_embeds.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
