from __future__ import annotations

"""Q-Former 계열 모델 구성 요소.

이 파일은 예전 placeholder를 실제 구현 진입점으로 교체한다. adapter 자체는 기존 import 경로를
깨지 않기 위해 `visual_adapter.py`의 구현을 재노출하고, ITC 사전정렬 모델은 여기서 정의한다.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from mini_vlm.models.vision_encoder import VisionFeatures
from mini_vlm.models.visual_adapter import QFormerVisualAdapter


@dataclass(frozen=True)
class QFormerItcOutput:
    """Q-Former image-text contrastive forward 결과."""

    loss: torch.Tensor
    logits_per_image: torch.Tensor
    logits_per_text: torch.Tensor
    image_features: torch.Tensor
    text_features: torch.Tensor


class QFormerItcModel(nn.Module):
    """Q-Former를 이미지-텍스트 대조학습으로 먼저 정렬하는 모델.

    의도: 바로 LLM answer loss로 Q-Former를 학습하면 모델이 이미지 grounding보다 문장 패턴을 먼저 외울 수
    있다. ITC는 BLIP-2의 첫 단계처럼 맞는 이미지-텍스트 쌍은 가깝게, 틀린 쌍은 멀게 만들어 Q-Former가
    LLM 연결 전에 시각-언어 정렬 감각을 갖도록 한다.
    참고: CLIP/BLIP-2의 image-text contrastive objective.
    선택 이유: ITM/ITG까지 한 번에 넣으면 학습 루프와 디버깅 표면이 급격히 커진다. ITC는 가장 작은
    사전정렬 단위이면서도 "이미지를 실제로 쓰는가"를 확인하는 데 직접적인 신호를 준다.
    """

    def __init__(
        self,
        *,
        vision_encoder: nn.Module,
        visual_adapter: QFormerVisualAdapter,
        text_encoder: nn.Module,
        text_hidden_dim: int,
        contrastive_dim: int = 256,
        freeze_vision: bool = True,
        freeze_text_encoder: bool = True,
    ) -> None:
        super().__init__()
        if contrastive_dim <= 0:
            raise ValueError("contrastive_dim은 1 이상이어야 합니다.")
        self.vision_encoder = vision_encoder
        self.visual_adapter = visual_adapter
        self.text_encoder = text_encoder
        self.freeze_vision = freeze_vision
        self.freeze_text_encoder = freeze_text_encoder
        self.image_projection = nn.Linear(visual_adapter.llm_dim, contrastive_dim)
        self.text_projection = nn.Linear(text_hidden_dim, contrastive_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        if self.freeze_vision and hasattr(self.vision_encoder, "requires_grad_"):
            self.vision_encoder.requires_grad_(False)
        if self.freeze_text_encoder and hasattr(self.text_encoder, "requires_grad_"):
            self.text_encoder.requires_grad_(False)
        self.enforce_freeze_modes()

    def train(self, mode: bool = True) -> "QFormerItcModel":
        super().train(mode)
        self.enforce_freeze_modes()
        return self

    def enforce_freeze_modes(self) -> None:
        if self.freeze_vision and hasattr(self.vision_encoder, "eval"):
            self.vision_encoder.eval()
        if self.freeze_text_encoder and hasattr(self.text_encoder, "eval"):
            self.text_encoder.eval()

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> QFormerItcOutput:
        image_features = self.encode_image(pixel_values)
        text_features = self.encode_text(input_ids=input_ids, attention_mask=attention_mask)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        loss = symmetric_itc_loss(logits_per_image=logits_per_image, logits_per_text=logits_per_text)
        return QFormerItcOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            image_features=image_features,
            text_features=text_features,
        )

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features: VisionFeatures = self.vision_encoder(pixel_values)
        visual_tokens = self.visual_adapter(features.patch_tokens, features.cls_token)
        pooled = visual_tokens.mean(dim=1)
        return F.normalize(self.image_projection(pooled.float()), dim=-1)

    def encode_text(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_text_encoder else torch.enable_grad()
        with context:
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            pooled = masked_mean_pool(hidden_states, attention_mask)
        return F.normalize(self.text_projection(pooled.float()), dim=-1)


def symmetric_itc_loss(*, logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    if logits_per_image.ndim != 2 or logits_per_text.ndim != 2:
        raise ValueError("ITC logits는 2차원 tensor여야 합니다.")
    if logits_per_image.shape[0] != logits_per_image.shape[1]:
        raise ValueError("ITC batch 안에서는 image와 text 수가 같아야 합니다.")
    labels = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=logits_per_image.device)
    image_loss = F.cross_entropy(logits_per_image, labels)
    text_loss = F.cross_entropy(logits_per_text, labels)
    return (image_loss + text_loss) / 2


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states shape는 [batch, sequence, dim]이어야 합니다.")
    mask = attention_mask.to(dtype=hidden_states.dtype, device=hidden_states.device).unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def initialize_qformer_from_distilbert(adapter: QFormerVisualAdapter, text_encoder: nn.Module) -> int:
    """DistilBERT self-attention/FFN weight를 Q-Former query block에 복사한다.

    의도: BLIP-2 Q-Former의 중요한 성질은 query transformer가 BERT 계열 언어 모델 구조에서 시작한다는
    점이다. 이 함수는 Q-Former의 self-attention과 FFN을 DistilBERT layer에서 초기화하고, vision
    cross-attention은 새로 학습하도록 둔다.
    제한: adapter hidden dim과 DistilBERT hidden dim이 같아야 한다. 일반 `distilbert-base-uncased`는
    hidden size 768이므로, 이 초기화를 쓰려면 `adapter_hidden_dim=768`이 필요하다.
    """

    if not hasattr(text_encoder, "transformer") or not hasattr(text_encoder.transformer, "layer"):
        raise ValueError("DistilBERT 호환 text_encoder만 Q-Former 초기화에 사용할 수 있습니다.")
    source_layers = list(text_encoder.transformer.layer)
    if not source_layers:
        raise ValueError("text_encoder에 복사할 transformer layer가 없습니다.")
    initialized = 0
    with torch.no_grad():
        for index, target_layer in enumerate(adapter.layers):
            source_layer = source_layers[index % len(source_layers)]
            ensure_compatible_qformer_layer(target_layer, source_layer)
            copy_distilbert_attention(target_layer.self_attention, source_layer.attention)
            target_layer.self_attention_norm.load_state_dict(source_layer.sa_layer_norm.state_dict())
            target_layer.ffn[0].weight.copy_(source_layer.ffn.lin1.weight)
            target_layer.ffn[0].bias.copy_(source_layer.ffn.lin1.bias)
            target_layer.ffn[2].weight.copy_(source_layer.ffn.lin2.weight)
            target_layer.ffn[2].bias.copy_(source_layer.ffn.lin2.bias)
            target_layer.ffn_norm.load_state_dict(source_layer.output_layer_norm.state_dict())
            initialized += 1
    return initialized


def ensure_compatible_qformer_layer(target_layer: nn.Module, source_layer: nn.Module) -> None:
    target_dim = target_layer.self_attention.embed_dim
    source_dim = source_layer.attention.q_lin.weight.shape[0]
    if target_dim != source_dim:
        raise ValueError(
            "DistilBERT 초기화 hidden dim이 맞지 않습니다: "
            f"adapter_hidden_dim={target_dim}, text_hidden_dim={source_dim}. "
            "distilbert-base-uncased를 쓰려면 adapter_hidden_dim=768로 설정하세요."
        )
    target_ffn_dim = target_layer.ffn[0].weight.shape[0]
    source_ffn_dim = source_layer.ffn.lin1.weight.shape[0]
    if target_ffn_dim != source_ffn_dim:
        raise ValueError(
            "DistilBERT 초기화 FFN dim이 맞지 않습니다: "
            f"adapter_ffn_dim={target_ffn_dim}, text_ffn_dim={source_ffn_dim}."
        )


def copy_distilbert_attention(target_attention: nn.MultiheadAttention, source_attention: nn.Module) -> None:
    target_attention.in_proj_weight.copy_(
        torch.cat(
            [
                source_attention.q_lin.weight,
                source_attention.k_lin.weight,
                source_attention.v_lin.weight,
            ],
            dim=0,
        )
    )
    target_attention.in_proj_bias.copy_(
        torch.cat(
            [
                source_attention.q_lin.bias,
                source_attention.k_lin.bias,
                source_attention.v_lin.bias,
            ],
            dim=0,
        )
    )
    target_attention.out_proj.weight.copy_(source_attention.out_lin.weight)
    target_attention.out_proj.bias.copy_(source_attention.out_lin.bias)
