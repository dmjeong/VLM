from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path


RUN_LOCAL_DINOV3 = os.environ.get("RUN_DINOV3_LOCAL_TESTS") == "1"


@unittest.skipUnless(RUN_LOCAL_DINOV3, "RUN_DINOV3_LOCAL_TESTS=1일 때만 로컬 DINOv3 weight 테스트를 실행합니다.")
class LocalDinoV3BackendTest(unittest.TestCase):
    def test_local_dinov3_vits16_extracts_patch_features(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch가 설치되어 있지 않습니다.")
        if importlib.util.find_spec("PIL") is None:
            self.skipTest("pillow가 설치되어 있지 않습니다.")
        if not Path("external/dinov3").exists():
            self.skipTest("external/dinov3 repo가 없습니다.")
        if not Path("models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth").exists():
            self.skipTest("로컬 DINOv3 ViT-S/16 weight가 없습니다.")

        from PIL import Image

        from mini_vlm.config import load_config
        from mini_vlm.models.vision_encoder import DinoVisionEncoder, LocalDinov3ImageProcessor

        config = load_config("configs/dinov3-local-vits16-qwen-smoke.json")
        processor = LocalDinov3ImageProcessor(config.vision_image_size)
        image = Image.open("data/samples/images/sample-grid.ppm").convert("RGB")
        processed = processor(images=[image], return_tensors="pt")
        encoder = DinoVisionEncoder(
            config.vision_model_id,
            freeze=True,
            backend=config.vision_backend,
            repo_dir=config.vision_repo_dir,
            model_name=config.vision_model_name,
            weights=config.vision_weights,
        )

        features = encoder(processed["pixel_values"])

        self.assertEqual(tuple(features.cls_token.shape), (1, 384))
        self.assertEqual(tuple(features.patch_tokens.shape), (1, 196, 384))


if __name__ == "__main__":
    unittest.main()
