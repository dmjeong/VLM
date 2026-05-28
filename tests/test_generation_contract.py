from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch가 설치된 환경에서만 generation contract 테스트를 실행합니다.")
class GenerationContractTest(unittest.TestCase):
    def test_greedy_generation_returns_text(self) -> None:
        import torch
        from torch import nn

        from mini_vlm.models.generation import greedy_generate_from_visual_prefix
        from mini_vlm.models.vision_encoder import VisionFeatures

        class FakeTokenizer:
            eos_token_id = 2

            def decode(self, token_ids, skip_special_tokens=True):
                return " ".join(str(token_id) for token_id in token_ids)

        class FakeVision(nn.Module):
            def forward(self, pixel_values):
                return VisionFeatures(
                    cls_token=torch.zeros(1, 4),
                    patch_tokens=torch.zeros(1, 3, 4),
                )

        class FakeAdapter(nn.Module):
            def forward(self, patch_tokens, cls_token=None):
                return torch.zeros(1, 2, 6)

        class FakeLlm(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(16, 6)

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, *, inputs_embeds, attention_mask, labels=None):
                logits = torch.zeros(inputs_embeds.shape[0], inputs_embeds.shape[1], 16)
                logits[:, -1, 5] = 1
                return SimpleNamespace(logits=logits)

        model = SimpleNamespace(vision_encoder=FakeVision(), visual_adapter=FakeAdapter(), llm=FakeLlm())
        result = greedy_generate_from_visual_prefix(
            model=model,
            tokenizer=FakeTokenizer(),
            pixel_values=torch.zeros(1, 3, 4, 4),
            prompt_input_ids=torch.tensor([[1, 3]]),
            max_new_tokens=2,
        )

        self.assertEqual(result.token_ids, [5, 5])
        self.assertEqual(result.text, "5 5")

    def test_generation_helpers_block_repeated_ngrams_and_trim_stop_text(self) -> None:
        from mini_vlm.models.generation import get_banned_ngram_next_tokens, trim_stop_strings

        banned_tokens = get_banned_ngram_next_tokens([1, 2, 1, 2], ngram_size=3)
        self.assertEqual(banned_tokens, {1})
        self.assertEqual(trim_stop_strings("apple\nAnswer: repeated", ("\nAnswer:",)), "apple")


if __name__ == "__main__":
    unittest.main()
