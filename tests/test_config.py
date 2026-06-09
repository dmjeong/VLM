from __future__ import annotations

import unittest
from pathlib import Path

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

    def test_loads_distilbert_initialized_qformer_itc_smoke_config(self) -> None:
        config = load_config("configs/dinov3-local-vits16-qwen-qformer-distilbert-itc-smoke.json")

        self.assertEqual(config.adapter_type, "qformer")
        self.assertEqual(config.adapter_hidden_dim, 768)
        self.assertTrue(config.qformer_init_from_text)
        self.assertEqual(config.qformer_text_model_id, "distilbert-base-uncased")

    def test_loads_external_10k_distilbert_qformer_itc_config(self) -> None:
        config = load_config("configs/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-stage0.json")

        self.assertEqual(config.adapter_type, "qformer")
        self.assertEqual(config.adapter_hidden_dim, 768)
        self.assertTrue(config.qformer_init_from_text)
        self.assertEqual(config.train_jsonl, "data/external_vlm_10k/train.jsonl")

    def test_loads_distilbert_qformer_itc_to_llm_smoke_config(self) -> None:
        config = load_config("configs/dinov3-local-vits16-qwen-qformer-distilbert-itc-to-llm-smoke.json")

        self.assertEqual(config.adapter_type, "qformer")
        self.assertEqual(config.adapter_hidden_dim, 768)
        self.assertTrue(config.init_visual_adapter.endswith("visual_adapter.pt"))

    def test_loads_external_10k_distilbert_qformer_itc_to_llm_config(self) -> None:
        config = load_config(
            "configs/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1.json"
        )

        self.assertEqual(config.adapter_type, "qformer")
        self.assertEqual(config.adapter_hidden_dim, 768)
        self.assertEqual(config.train_jsonl, "data/external_vlm_10k/train.jsonl")
        self.assertTrue(config.init_visual_adapter.endswith("visual_adapter.pt"))

    def test_loads_dinov3_stage2_lora_config(self) -> None:
        config = load_config(
            "configs/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage2-lora-epoch1.json"
        )

        self.assertTrue(config.use_lora)
        self.assertEqual(config.adapter_type, "qformer")
        self.assertEqual(config.adapter_hidden_dim, 768)
        self.assertEqual(config.learning_rate, 0.000005)
        self.assertEqual(config.max_grad_norm, 0.5)
        self.assertIn("visual_adapter_epoch_2.pt", config.init_visual_adapter)
        self.assertIn("stage2-lora", config.output_dir)

    def test_qformer_configs_use_distilbert_768_path_only(self) -> None:
        qformer_configs = sorted(Path("configs").glob("*qformer*.json"))
        self.assertGreater(len(qformer_configs), 0)

        for config_path in qformer_configs:
            config = load_config(config_path)
            self.assertEqual(config.adapter_type, "qformer", config_path.name)
            self.assertEqual(config.adapter_hidden_dim, 768, config_path.name)
            if "itc" in config_path.name and "to-llm" not in config_path.name:
                self.assertTrue(config.qformer_init_from_text, config_path.name)
                self.assertEqual(config.qformer_text_model_id, "distilbert-base-uncased", config_path.name)

    def test_rejects_distilbert_qformer_when_hidden_dim_is_not_768(self) -> None:
        with self.assertRaises(ValueError):
            MiniVlmConfig.from_dict(
                {
                    "adapter_type": "qformer",
                    "adapter_hidden_dim": 512,
                    "qformer_init_from_text": True,
                }
            )

    def test_parses_generation_and_lora_fields(self) -> None:
        config = MiniVlmConfig.from_dict(
            {
                "adapter_type": "perceiver",
                "use_lora": True,
                "device": "cpu",
                "seed": 7,
                "max_grad_norm": 0.5,
                "init_visual_adapter": "artifacts/init/visual_adapter.pt",
                "lora_r": 4,
                "lora_target_modules": ["q_proj", "v_proj"],
                "qformer_text_model_id": "distilbert-base-cased",
                "qformer_init_from_text": True,
                "contrastive_dim": 128,
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
        self.assertEqual(config.init_visual_adapter, "artifacts/init/visual_adapter.pt")
        self.assertEqual(config.lora_r, 4)
        self.assertEqual(config.lora_target_modules, ("q_proj", "v_proj"))
        self.assertEqual(config.qformer_text_model_id, "distilbert-base-cased")
        self.assertTrue(config.qformer_init_from_text)
        self.assertEqual(config.contrastive_dim, 128)
        self.assertEqual(config.max_new_tokens, 12)
        self.assertEqual(config.repetition_penalty, 1.2)
        self.assertEqual(config.stop_strings, ("STOP",))


if __name__ == "__main__":
    unittest.main()
