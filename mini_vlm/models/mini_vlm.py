from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from mini_vlm.models.vision_encoder import VisionFeatures


class VisionEncoderLike(Protocol):
    def __call__(self, pixel_values: torch.Tensor) -> VisionFeatures:
        ...


class VisualAdapterLike(Protocol):
    def __call__(self, patch_tokens: torch.Tensor, cls_token: torch.Tensor | None = None) -> torch.Tensor:
        ...


class LlmLike(Protocol):
    def get_input_embeddings(self):
        ...

    def __call__(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        ...


@dataclass(frozen=True)
class MiniVlmForwardOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    visual_tokens: torch.Tensor
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None


class MiniVlmForConditionalGeneration(nn.Module):
    """DINOv3 visual token과 LLM text embedding을 결합하는 mini VLM wrapper.

    의도: 이미지 feature를 LLM 앞쪽 soft prompt로 붙여 causal LM이 답변 token을 생성하도록 만든다.
    참고: 설계서 6.6 MiniVlmForConditionalGeneration.
    선택 이유: LLM tokenizer vocabulary를 건드리지 않고 `inputs_embeds` 경로로 visual token을 삽입하면
    작은 실험에서 구조를 이해하기 쉽다.
    """

    def __init__(
        self,
        vision_encoder: VisionEncoderLike,
        visual_adapter: VisualAdapterLike,
        llm: LlmLike,
        visual_token_count: int,
        freeze_vision: bool = True,
        freeze_llm: bool = True,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.visual_adapter = visual_adapter
        self.llm = llm
        self.visual_token_count = visual_token_count
        self.freeze_vision = freeze_vision
        self.freeze_llm = freeze_llm
        if self.freeze_vision and hasattr(self.vision_encoder, "requires_grad_"):
            self.vision_encoder.requires_grad_(False)
        if self.freeze_llm and hasattr(self.llm, "requires_grad_"):
            self.llm.requires_grad_(False)
        self.enforce_freeze_modes()

    def train(self, mode: bool = True) -> "MiniVlmForConditionalGeneration":
        """학습 모드 전환 뒤 freeze module은 다시 eval로 고정한다.

        의도: PyTorch의 `model.train()`은 모든 child module을 train mode로 바꾼다. Stage 1에서는
        DINOv3와 LLM을 feature extractor/backbone으로 고정하고 adapter만 학습하므로, dropout 같은
        train-mode 동작이 freeze된 backbone 안에서 켜지지 않게 한다.
        참고: Check 분석 5.3 frozen LLM eval mode 흔들림.
        """

        super().train(mode)
        self.enforce_freeze_modes()
        return self

    def enforce_freeze_modes(self) -> None:
        """freeze 정책을 parameter와 module mode 양쪽에 반영한다."""

        if self.freeze_vision and hasattr(self.vision_encoder, "eval"):
            self.vision_encoder.eval()
        if self.freeze_llm and hasattr(self.llm, "eval"):
            self.llm.eval()

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> MiniVlmForwardOutput:
        features = self.vision_encoder(pixel_values)
        visual_tokens = self.visual_adapter(features.patch_tokens, features.cls_token)
        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        if visual_tokens.dtype != text_embeddings.dtype:
            visual_tokens = visual_tokens.to(dtype=text_embeddings.dtype)
        inputs_embeds = torch.cat([visual_tokens, text_embeddings], dim=1)
        combined_attention = prepend_visual_attention(attention_mask, visual_tokens.shape[1])
        combined_labels = prepend_visual_labels(labels, visual_tokens.shape[1]) if labels is not None else None
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=combined_attention, labels=combined_labels)
        return MiniVlmForwardOutput(
            loss=getattr(outputs, "loss", None),
            logits=outputs.logits,
            visual_tokens=visual_tokens,
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention,
            labels=combined_labels,
        )


def prepend_visual_attention(attention_mask: torch.Tensor, visual_token_count: int) -> torch.Tensor:
    visual_attention = torch.ones(
        attention_mask.shape[0],
        visual_token_count,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([visual_attention, attention_mask], dim=1)


def prepend_visual_labels(labels: torch.Tensor, visual_token_count: int) -> torch.Tensor:
    visual_labels = torch.full(
        (labels.shape[0], visual_token_count),
        fill_value=-100,
        dtype=labels.dtype,
        device=labels.device,
    )
    return torch.cat([visual_labels, labels], dim=1)
