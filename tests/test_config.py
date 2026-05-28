from __future__ import annotations

import unittest

from mini_vlm.config import MiniVlmConfig, load_config


class MiniVlmConfigTest(unittest.TestCase):
    def test_loads_default_config(self) -> None:
        config = load_config("configs/dinov3-mini-vlm.json")

        self.assertEqual(config.experiment_name, "dinov3-mini-vlm-mlp-baseline")
        self.assertEqual(config.adapter_type, "mlp")
        self.assertEqual(config.visual_token_count, 32)

    def test_rejects_unknown_adapter(self) -> None:
        with self.assertRaises(ValueError):
            MiniVlmConfig.from_dict({"adapter_type": "unknown"})

    def test_loads_local_dinov3_backend_config(self) -> None:
        config = load_config("configs/dinov3-local-vits16-qwen-smoke.json")

        self.assertEqual(config.vision_backend, "torchhub")
        self.assertEqual(config.vision_model_name, "dinov3_vits16")
        self.assertEqual(config.adapter_type, "perceiver")
        self.assertTrue(config.use_lora)
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.max_grad_norm, 1.0)
        self.assertEqual(config.learning_rate, 0.00002)
        self.assertEqual(config.lora_target_modules, ("q_proj", "k_proj", "v_proj", "o_proj"))
        self.assertEqual(config.no_repeat_ngram_size, 3)
        self.assertTrue(config.vision_weights.endswith(".pth"))

    def test_rejects_unknown_vision_backend(self) -> None:
        with self.assertRaises(ValueError):
            MiniVlmConfig.from_dict({"vision_backend": "unknown"})

    def test_loads_wikimedia_1k_stage1_config(self) -> None:
        config = load_config("configs/dinov3-local-vits16-qwen-wikimedia-1k-adapter-stage1.json")

        self.assertEqual(config.train_jsonl, "data/wikimedia_commons_1k/train.jsonl")
        self.assertEqual(config.validation_jsonl, "data/wikimedia_commons_1k/validation.jsonl")
        self.assertEqual(config.test_jsonl, "data/wikimedia_commons_1k/test.jsonl")
        self.assertFalse(config.use_lora)
        self.assertEqual(config.visual_token_count, 16)
        self.assertEqual(config.gradient_accumulation_steps, 4)

    def test_loads_external_10k_stage1_config(self) -> None:
        config = load_config("configs/dinov3-local-vits16-qwen-external-10k-adapter-stage1.json")

        self.assertEqual(config.train_jsonl, "data/external_vlm_10k/train.jsonl")
        self.assertEqual(config.validation_jsonl, "data/external_vlm_10k/validation.jsonl")
        self.assertEqual(config.test_jsonl, "data/external_vlm_10k/test.jsonl")
        self.assertEqual(config.image_root, "data/external_vlm_10k")
        self.assertFalse(config.use_lora)
        self.assertEqual(config.max_text_length, 192)
        self.assertEqual(config.gradient_accumulation_steps, 8)

    def test_parses_generation_and_lora_fields(self) -> None:
        config = MiniVlmConfig.from_dict(
            {
                "adapter_type": "perceiver",
                "use_lora": True,
                "device": "cpu",
                "seed": 7,
                "max_grad_norm": 0.5,
                "lora_r": 4,
                "lora_target_modules": ["q_proj", "v_proj"],
                "max_new_tokens": 12,
                "repetition_penalty": 1.2,
                "stop_strings": ["STOP"],
            }
        )

        self.assertEqual(config.adapter_type, "perceiver")
        self.assertTrue(config.use_lora)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.max_grad_norm, 0.5)
        self.assertEqual(config.lora_r, 4)
        self.assertEqual(config.lora_target_modules, ("q_proj", "v_proj"))
        self.assertEqual(config.max_new_tokens, 12)
        self.assertEqual(config.repetition_penalty, 1.2)
        self.assertEqual(config.stop_strings, ("STOP",))


if __name__ == "__main__":
    unittest.main()
