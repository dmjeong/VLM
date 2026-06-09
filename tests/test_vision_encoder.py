from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from mini_vlm.models.vision_encoder import (
    extract_hf_last_hidden_state,
    resolve_hf_hidden_size,
    resolve_hf_patch_count,
    split_hf_vision_sequence,
)


class VisionEncoderHelperTest(unittest.TestCase):
    def test_resolve_hf_hidden_size_reads_direct_hidden_size(self) -> None:
        config = SimpleNamespace(hidden_size=384)

        self.assertEqual(resolve_hf_hidden_size(config), 384)

    def test_resolve_hf_hidden_size_reads_nested_vision_config(self) -> None:
        config = SimpleNamespace(vision_config=SimpleNamespace(hidden_size=768))

        self.assertEqual(resolve_hf_hidden_size(config), 768)

    def test_extract_hf_last_hidden_state_reads_direct_output(self) -> None:
        hidden_state = torch.randn(2, 5, 8)
        outputs = SimpleNamespace(last_hidden_state=hidden_state)

        self.assertIs(extract_hf_last_hidden_state(outputs), hidden_state)

    def test_extract_hf_last_hidden_state_reads_clip_style_output(self) -> None:
        hidden_state = torch.randn(2, 5, 8)
        outputs = SimpleNamespace(vision_model_output=SimpleNamespace(last_hidden_state=hidden_state))

        self.assertIs(extract_hf_last_hidden_state(outputs), hidden_state)

    def test_resolve_hf_patch_count_reads_image_and_patch_size(self) -> None:
        config = SimpleNamespace(image_size=224, patch_size=16)

        self.assertEqual(resolve_hf_patch_count(config), 196)

    def test_split_hf_vision_sequence_keeps_siglip_patch_tokens(self) -> None:
        hidden_state = torch.randn(2, 196, 8)
        pooled_output = torch.randn(2, 8)
        config = SimpleNamespace(image_size=224, patch_size=16)

        cls_token, patch_tokens = split_hf_vision_sequence(
            last_hidden_state=hidden_state,
            pooled_output=pooled_output,
            config=config,
        )

        self.assertIs(cls_token, pooled_output)
        self.assertIs(patch_tokens, hidden_state)

    def test_split_hf_vision_sequence_splits_clip_cls_token(self) -> None:
        hidden_state = torch.randn(2, 197, 8)
        config = SimpleNamespace(image_size=224, patch_size=16)

        cls_token, patch_tokens = split_hf_vision_sequence(
            last_hidden_state=hidden_state,
            pooled_output=None,
            config=config,
        )

        self.assertEqual(tuple(cls_token.shape), (2, 8))
        self.assertEqual(tuple(patch_tokens.shape), (2, 196, 8))


if __name__ == "__main__":
    unittest.main()
