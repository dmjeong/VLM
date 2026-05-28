from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch가 설치된 환경에서만 visual adapter shape 테스트를 실행합니다.")
class VisualAdapterTest(unittest.TestCase):
    def test_mlp_visual_adapter_projects_patch_tokens(self) -> None:
        import torch

        from mini_vlm.models.visual_adapter import MlpVisualAdapter

        adapter = MlpVisualAdapter(vision_dim=8, llm_dim=12, visual_token_count=4, hidden_dim=16)
        patch_tokens = torch.randn(2, 9, 8)
        cls_token = torch.randn(2, 8)

        visual_tokens = adapter(patch_tokens, cls_token)

        self.assertEqual(tuple(visual_tokens.shape), (2, 4, 12))

    def test_perceiver_resampler_projects_patch_tokens(self) -> None:
        import torch

        from mini_vlm.models.visual_adapter import PerceiverResamplerAdapter

        adapter = PerceiverResamplerAdapter(vision_dim=8, llm_dim=12, visual_token_count=5, hidden_dim=16)
        patch_tokens = torch.randn(2, 9, 8)
        cls_token = torch.randn(2, 8)

        visual_tokens = adapter(patch_tokens, cls_token)

        self.assertEqual(tuple(visual_tokens.shape), (2, 5, 12))

    def test_attention_head_count_uses_divisor(self) -> None:
        from mini_vlm.models.visual_adapter import choose_attention_head_count

        self.assertEqual(choose_attention_head_count(384), 8)
        self.assertEqual(choose_attention_head_count(7), 7)


if __name__ == "__main__":
    unittest.main()
