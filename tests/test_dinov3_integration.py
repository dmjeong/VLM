from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path


RUN_INTEGRATION = os.environ.get("RUN_MODEL_INTEGRATION_TESTS") == "1"


@unittest.skipUnless(RUN_INTEGRATION, "RUN_MODEL_INTEGRATION_TESTS=1일 때만 DINOv3 integration test를 실행합니다.")
class DinoV3IntegrationTest(unittest.TestCase):
    def test_dinov3_encoder_extracts_patch_features(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch가 설치되어 있지 않습니다.")
        if importlib.util.find_spec("transformers") is None:
            self.skipTest("transformers가 설치되어 있지 않습니다.")
        if importlib.util.find_spec("PIL") is None:
            self.skipTest("pillow가 설치되어 있지 않습니다.")

        from PIL import Image
        from transformers import AutoImageProcessor
        from huggingface_hub import get_token

        from mini_vlm.config import load_config
        from mini_vlm.models.vision_encoder import DinoVisionEncoder

        config = load_config("configs/dinov3-mini-vlm-smoke.json")
        if config.vision_model_id.startswith("facebook/dinov3") and not get_token():
            self.skipTest("DINOv3는 Hugging Face gated repo라서 HF_TOKEN과 접근 승인이 필요합니다.")
        image_path = Path("data/samples/images/sample-grid.ppm")
        image = Image.open(image_path).convert("RGB")
        processor = AutoImageProcessor.from_pretrained(config.vision_model_id)
        processed = processor(images=[image], return_tensors="pt")
        pixel_values = processed["pixel_values"] if isinstance(processed, dict) else processed.pixel_values
        encoder = DinoVisionEncoder(config.vision_model_id, freeze=True)

        features = encoder(pixel_values)

        self.assertEqual(features.cls_token.shape[0], 1)
        self.assertEqual(features.patch_tokens.shape[0], 1)
        self.assertGreater(features.patch_tokens.shape[1], 0)
        self.assertEqual(features.patch_tokens.shape[-1], features.cls_token.shape[-1])


if __name__ == "__main__":
    unittest.main()
