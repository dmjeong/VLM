from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from mini_vlm.data.dataset import MiniVlmSample


class TokenizerLike(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ...


@dataclass(frozen=True)
class MiniVlmBatch:
    """모델 forward에 넘길 batch.

    `pixel_values`, `input_ids` 등은 실제 실행 시 torch.Tensor가 되지만, 기본 단위 테스트가 torch 설치에
    의존하지 않도록 타입은 object로 둔다.
    """

    pixel_values: object
    input_ids: object
    attention_mask: object
    labels: object
    sample_ids: list[str]
    image_paths: list[str]


class MiniVlmCollator:
    """샘플을 batch tensor로 변환한다.

    의도: 질문 token은 label에서 제외하고 답변 token만 causal LM loss에 들어가게 한다.
    참고: 설계서 5.3 Batch 구조와 6.6 label 확장 정책.
    선택 이유: VLM 학습에서 visual token과 prompt token까지 맞히도록 학습하면 모델이 실제 답변보다
    입력 복사에 loss를 낭비할 수 있다.
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        image_processor: Any | None = None,
        image_root: str | Path = ".",
        max_text_length: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_root = Path(image_root)
        self.max_text_length = max_text_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else self.pad_token_id

    def build_text_features(self, sample: MiniVlmSample) -> tuple[list[int], list[int]]:
        prompt = f"Question: {sample.question}\nAnswer:"
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(f" {sample.answer}", add_special_tokens=False) + [self.eos_token_id]
        input_ids = (prompt_ids + answer_ids)[: self.max_text_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_text_length]
        return input_ids, labels

    def __call__(self, samples: Sequence[MiniVlmSample]) -> MiniVlmBatch:
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        image_paths: list[str] = []
        for sample in samples:
            input_ids, labels = self.build_text_features(sample)
            input_rows.append(input_ids)
            label_rows.append(labels)
            image_paths.append(str(self._resolve_image_path(sample.image)))

        max_length = max(len(row) for row in input_rows)
        padded_inputs = [row + [self.pad_token_id] * (max_length - len(row)) for row in input_rows]
        attention_mask = [[1] * len(row) + [0] * (max_length - len(row)) for row in input_rows]
        padded_labels = [row + [-100] * (max_length - len(row)) for row in label_rows]

        pixel_values = self._process_images(image_paths)
        return MiniVlmBatch(
            pixel_values=pixel_values,
            input_ids=_to_tensor_if_available(padded_inputs, dtype_name="long"),
            attention_mask=_to_tensor_if_available(attention_mask, dtype_name="long"),
            labels=_to_tensor_if_available(padded_labels, dtype_name="long"),
            sample_ids=[sample.sample_id for sample in samples],
            image_paths=image_paths,
        )

    def _resolve_image_path(self, image: str) -> Path:
        path = Path(image)
        if path.is_absolute():
            return path
        return self.image_root / path

    def _process_images(self, image_paths: list[str]) -> object:
        if self.image_processor is None:
            return image_paths
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("이미지 처리에는 pillow가 필요합니다.") from exc
        images = [Image.open(path).convert("RGB") for path in image_paths]
        processed = self.image_processor(images=images, return_tensors="pt")
        return processed["pixel_values"] if isinstance(processed, dict) else processed.pixel_values


def _to_tensor_if_available(values: list[list[int]], dtype_name: str = "long") -> object:
    try:
        import torch
    except ImportError:
        return values
    dtype = getattr(torch, dtype_name)
    return torch.tensor(values, dtype=dtype)
