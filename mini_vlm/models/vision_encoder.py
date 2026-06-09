from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import nn


@dataclass(frozen=True)
class VisionFeatures:
    cls_token: torch.Tensor
    patch_tokens: torch.Tensor
    pooled_output: torch.Tensor | None = None


class DinoVisionEncoder(nn.Module):
    """DINO/CLIP/SigLIP 계열 vision model wrapper.

    의도: vision encoder별 출력 형태를 mini VLM 내부 표준인 cls_token/patch_tokens로 고정한다.
    참고: DINOv3는 Meta torch hub 구현과 Hugging Face 구현이 모두 있으며, CLIP/SigLIP은
    Hugging Face에서 vision submodule output 구조가 다르다.
    선택 이유: adapter와 MiniVLM 본체가 특정 backend output class에 직접 의존하지 않게 한다.
    """

    def __init__(
        self,
        model_id: str,
        freeze: bool = True,
        backend: str = "hf",
        repo_dir: str = "",
        model_name: str = "dinov3_vits16",
        weights: str = "",
    ) -> None:
        super().__init__()
        self.backend = backend
        self.freeze = freeze
        if self.backend == "hf":
            self.model = _load_hf_model(model_id)
            self.hidden_size = resolve_hf_hidden_size(self.model.config)
        elif self.backend == "torchhub":
            self.model = _load_local_dinov3_model(repo_dir=repo_dir, model_name=model_name, weights=weights)
            self.hidden_size = int(getattr(self.model, "embed_dim"))
        else:
            raise ValueError("backend는 hf 또는 torchhub 중 하나여야 합니다.")
        if self.freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def forward(self, pixel_values: torch.Tensor) -> VisionFeatures:
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            if self.backend == "hf":
                outputs = self.model(pixel_values=pixel_values)
                last_hidden_state = extract_hf_last_hidden_state(outputs)
                pooled_output = getattr(outputs, "pooler_output", None)
                cls_token, patch_tokens = split_hf_vision_sequence(
                    last_hidden_state=last_hidden_state,
                    pooled_output=pooled_output,
                    config=self.model.config,
                )
            else:
                outputs = self.model.forward_features(pixel_values)
                cls_token = outputs["x_norm_clstoken"]
                patch_tokens = outputs["x_norm_patchtokens"]
                pooled_output = None
        return VisionFeatures(cls_token=cls_token, patch_tokens=patch_tokens, pooled_output=pooled_output)


class LocalDinov3ImageProcessor:
    """Meta DINOv3 local backend용 최소 이미지 전처리기.

    의도: Hugging Face `AutoImageProcessor` 없이도 Meta에서 받은 `.pth` weight를 바로 실험한다.
    참고: DINO 계열 ViT 입력은 일반적으로 ImageNet mean/std 정규화된 RGB tensor를 사용한다.
    선택 이유: 현재 목표는 SaaS가 아니라 모델 배선 검증이므로, 필요한 전처리 계약만 작고 명확하게 둔다.
    """

    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)

    def __call__(self, images, return_tensors: str = "pt"):
        if return_tensors != "pt":
            raise ValueError("LocalDinov3ImageProcessor는 return_tensors='pt'만 지원합니다.")
        if not isinstance(images, list):
            images = [images]
        tensors = [self._preprocess_image(image) for image in images]
        return {"pixel_values": torch.stack(tensors, dim=0)}

    def _preprocess_image(self, image) -> torch.Tensor:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("로컬 DINOv3 이미지 전처리에는 numpy가 필요합니다.") from exc
        resized = image.convert("RGB").resize((self.image_size, self.image_size))
        array = np.asarray(resized, dtype="float32") / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.mean) / self.std


def _load_hf_model(model_id: str) -> nn.Module:
    try:
        from transformers import AutoConfig, AutoModel, CLIPVisionModel, SiglipVisionModel
    except ImportError as exc:
        raise RuntimeError("Hugging Face vision backend를 사용하려면 transformers를 설치해야 합니다.") from exc

    config = AutoConfig.from_pretrained(model_id)
    model_type = str(getattr(config, "model_type", ""))
    if model_type == "clip":
        return CLIPVisionModel.from_pretrained(model_id)
    if model_type in {"siglip", "siglip2"}:
        return SiglipVisionModel.from_pretrained(model_id)
    return AutoModel.from_pretrained(model_id)


def resolve_hf_hidden_size(config) -> int:
    """Hugging Face vision config에서 hidden size를 얻는다."""

    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)
    vision_config = getattr(config, "vision_config", None)
    vision_hidden_size = getattr(vision_config, "hidden_size", None)
    if vision_hidden_size is not None:
        return int(vision_hidden_size)
    raise ValueError("Hugging Face vision model config에서 hidden_size를 찾을 수 없습니다.")


def extract_hf_last_hidden_state(outputs) -> torch.Tensor:
    """Hugging Face vision model output에서 patch sequence를 추출한다."""

    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is not None:
        return last_hidden_state
    vision_output = getattr(outputs, "vision_model_output", None)
    if vision_output is not None and getattr(vision_output, "last_hidden_state", None) is not None:
        return vision_output.last_hidden_state
    if isinstance(outputs, (tuple, list)) and outputs:
        first_output = outputs[0]
        if isinstance(first_output, torch.Tensor) and first_output.ndim == 3:
            return first_output
    raise ValueError("Hugging Face vision model output에서 last_hidden_state를 찾을 수 없습니다.")


def split_hf_vision_sequence(
    *,
    last_hidden_state: torch.Tensor,
    pooled_output: torch.Tensor | None,
    config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF vision sequence를 cls token과 patch token으로 나눈다.

    의도: CLIP은 CLS token을 포함하지만 SigLIP은 patch token만 반환한다. 무조건 첫 token을 떼면 SigLIP의
    patch 하나를 잃으므로, config의 image/patch size로 기대 patch 수를 계산해 분기한다.
    """

    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state shape는 [batch, sequence, hidden]이어야 합니다.")
    expected_patch_count = resolve_hf_patch_count(config)
    sequence_length = last_hidden_state.shape[1]
    if expected_patch_count is not None and sequence_length == expected_patch_count:
        cls_token = pooled_output if pooled_output is not None else last_hidden_state.mean(dim=1)
        return cls_token, last_hidden_state
    return last_hidden_state[:, 0, :], last_hidden_state[:, 1:, :]


def resolve_hf_patch_count(config) -> int | None:
    image_size = getattr(config, "image_size", None)
    patch_size = getattr(config, "patch_size", None)
    if image_size is None or patch_size is None:
        vision_config = getattr(config, "vision_config", None)
        image_size = getattr(vision_config, "image_size", None)
        patch_size = getattr(vision_config, "patch_size", None)
    if isinstance(image_size, (list, tuple)):
        height, width = int(image_size[0]), int(image_size[1])
    elif image_size is not None:
        height = width = int(image_size)
    else:
        return None
    if isinstance(patch_size, (list, tuple)):
        patch_height, patch_width = int(patch_size[0]), int(patch_size[1])
    elif patch_size is not None:
        patch_height = patch_width = int(patch_size)
    else:
        return None
    if patch_height <= 0 or patch_width <= 0:
        return None
    return (height // patch_height) * (width // patch_width)


def _load_local_dinov3_model(*, repo_dir: str, model_name: str, weights: str) -> nn.Module:
    repo_path = Path(repo_dir)
    weights_path = Path(weights)
    if not repo_path.exists():
        raise FileNotFoundError(f"DINOv3 repo를 찾을 수 없습니다: {repo_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"DINOv3 weight 파일을 찾을 수 없습니다: {weights_path}")
    repo_resolved = str(repo_path.resolve())
    if repo_resolved not in sys.path:
        sys.path.insert(0, repo_resolved)
    from dinov3.hub import backbones

    if not hasattr(backbones, model_name):
        raise ValueError(f"지원하지 않는 DINOv3 backbone 이름입니다: {model_name}")
    factory = getattr(backbones, model_name)
    return factory(weights=str(weights_path.resolve()))
