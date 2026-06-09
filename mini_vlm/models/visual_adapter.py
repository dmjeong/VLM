from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class VisualAdapterOutput:
    visual_tokens: torch.Tensor


class MlpVisualAdapter(nn.Module):
    """DINOv3 patch token을 LLM embedding token으로 변환하는 baseline adapter.

    의도: Q-Former로 바로 가기 전에 가장 단순한 projector로 전체 VLM forward 계약을 검증한다.
    참고: docs/02-design/features/dinov3-mini-vlm.design.md#63-mlpvisualadapter
    선택 이유: 평균 pooling + MLP는 성능 상한은 낮지만, shape 오류와 학습 루프 문제를 가장 빠르게 드러낸다.
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        visual_token_count: int = 32,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        if visual_token_count <= 0:
            raise ValueError("visual_token_count는 1 이상이어야 합니다.")
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.visual_token_count = visual_token_count
        self.norm = nn.LayerNorm(vision_dim)
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
        )

    def forward(self, patch_tokens: torch.Tensor, cls_token: torch.Tensor | None = None) -> torch.Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError("patch_tokens shape는 [batch, patches, vision_dim]이어야 합니다.")
        pooled_tokens = adaptive_sequence_mean_pool(patch_tokens, self.visual_token_count)
        return self.projector(self.norm(pooled_tokens))


class PerceiverResamplerAdapter(nn.Module):
    """learnable query가 이미지 patch token을 읽어 visual token으로 압축하는 adapter.

    의도: 단순 mean pooling MLP보다 각 visual token이 이미지의 다른 부분에 주의를 둘 수 있게 한다.
    참고: Flamingo 계열 Perceiver Resampler 아이디어를 mini VLM 실험용으로 축소했다.
    선택 이유: Q-Former보다 구현과 디버깅이 단순하면서도, patch token을 learnable query로 재샘플링하는 구조라
    MLP baseline의 정보 병목을 줄일 수 있다.
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        visual_token_count: int = 32,
        hidden_dim: int = 1024,
        layer_count: int = 2,
    ) -> None:
        super().__init__()
        if visual_token_count <= 0:
            raise ValueError("visual_token_count는 1 이상이어야 합니다.")
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.visual_token_count = visual_token_count
        self.query_tokens = nn.Parameter(torch.randn(visual_token_count, vision_dim) * 0.02)
        self.input_norm = nn.LayerNorm(vision_dim)
        self.layers = nn.ModuleList(
            [PerceiverResamplerLayer(vision_dim=vision_dim, hidden_dim=hidden_dim) for _ in range(max(1, layer_count))]
        )
        self.output_norm = nn.LayerNorm(vision_dim)
        self.projector = nn.Linear(vision_dim, llm_dim)

    def forward(self, patch_tokens: torch.Tensor, cls_token: torch.Tensor | None = None) -> torch.Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError("patch_tokens shape는 [batch, patches, vision_dim]이어야 합니다.")
        memory = self.input_norm(patch_tokens)
        if cls_token is not None:
            memory = torch.cat([self.input_norm(cls_token).unsqueeze(1), memory], dim=1)
        query_tokens = self.query_tokens.unsqueeze(0).expand(patch_tokens.shape[0], -1, -1)
        for layer in self.layers:
            query_tokens = layer(query_tokens=query_tokens, memory=memory)
        return self.projector(self.output_norm(query_tokens))


class QFormerVisualAdapter(nn.Module):
    """BLIP-2 Q-Former 아이디어를 축소한 visual adapter.

    의도: DINOv3 patch token을 그대로 LLM에 밀어 넣지 않고, 학습 가능한 query token이 이미지 token을
    반복적으로 읽어 언어 모델에 넘길 compact visual token을 만든다.
    참고: BLIP-2의 Q-Former는 BERT 계열 query transformer에 vision cross-attention을 추가한다.
    선택 이유: 지금 Perceiver adapter는 query가 image token을 읽는 점은 비슷하지만 query끼리의
    self-attention 단계가 없다. Q-Former 구조는 query token들이 서로 정보를 교환한 뒤 이미지 token에
    cross-attention하므로, 객체/속성/관계 정보를 역할별 query로 나눠 담는 실험에 더 적합하다.

    주의: 이 구현은 DistilBERT weight로 초기화한 완전한 BLIP-2 Q-Former가 아니라, mini VLM 실험을 위한
    작은 Q-Former block이다. DistilBERT 초기화와 ITC/ITM/ITG pretraining은 다음 단계에서 붙인다.
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        visual_token_count: int = 32,
        hidden_dim: int = 768,
        layer_count: int = 2,
    ) -> None:
        super().__init__()
        if visual_token_count <= 0:
            raise ValueError("visual_token_count는 1 이상이어야 합니다.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim은 1 이상이어야 합니다.")
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.visual_token_count = visual_token_count
        self.hidden_dim = hidden_dim

        # DINOv3 feature 차원과 Q-Former 내부 차원을 분리한다. 이렇게 해두면 vision encoder나 LLM을
        # 바꿔도 Q-Former block의 크기는 config.adapter_hidden_dim으로 독립적으로 조절할 수 있다.
        self.vision_norm = nn.LayerNorm(vision_dim)
        self.vision_projector = nn.Linear(vision_dim, hidden_dim)
        self.query_tokens = nn.Parameter(torch.randn(visual_token_count, hidden_dim) * 0.02)
        self.layers = nn.ModuleList(
            [QFormerLayer(hidden_dim=hidden_dim) for _ in range(max(1, layer_count))]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.llm_projector = nn.Linear(hidden_dim, llm_dim)

    def forward(self, patch_tokens: torch.Tensor, cls_token: torch.Tensor | None = None) -> torch.Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError("patch_tokens shape는 [batch, patches, vision_dim]이어야 합니다.")
        memory = self.vision_projector(self.vision_norm(patch_tokens))
        if cls_token is not None:
            cls_memory = self.vision_projector(self.vision_norm(cls_token)).unsqueeze(1)
            memory = torch.cat([cls_memory, memory], dim=1)

        query_tokens = self.query_tokens.unsqueeze(0).expand(patch_tokens.shape[0], -1, -1)
        for layer in self.layers:
            query_tokens = layer(query_tokens=query_tokens, image_memory=memory)
        return self.llm_projector(self.output_norm(query_tokens))


class QFormerLayer(nn.Module):
    """Q-Former 한 층.

    구성: query self-attention -> image cross-attention -> feed-forward.
    BLIP-2에서는 BERT block 일부에 cross-attention을 삽입하지만, 여기서는 실험과 디버깅을 쉽게 하려고
    세 단계를 명시적으로 분리했다.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        num_heads = choose_attention_head_count(hidden_dim)
        self.self_attention_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.cross_attention_query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention_memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, *, query_tokens: torch.Tensor, image_memory: torch.Tensor) -> torch.Tensor:
        self_attended, _ = self.self_attention(
            query=self.self_attention_norm(query_tokens),
            key=self.self_attention_norm(query_tokens),
            value=query_tokens,
            need_weights=False,
        )
        query_tokens = query_tokens + self_attended
        cross_attended, _ = self.cross_attention(
            query=self.cross_attention_query_norm(query_tokens),
            key=self.cross_attention_memory_norm(image_memory),
            value=image_memory,
            need_weights=False,
        )
        query_tokens = query_tokens + cross_attended
        return query_tokens + self.ffn(self.ffn_norm(query_tokens))


class PerceiverResamplerLayer(nn.Module):
    def __init__(self, vision_dim: int, hidden_dim: int) -> None:
        super().__init__()
        num_heads = choose_attention_head_count(vision_dim)
        self.query_norm = nn.LayerNorm(vision_dim)
        self.memory_norm = nn.LayerNorm(vision_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=vision_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(vision_dim)
        self.ffn = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vision_dim),
        )

    def forward(self, *, query_tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended, _ = self.cross_attention(
            query=self.query_norm(query_tokens),
            key=self.memory_norm(memory),
            value=memory,
            need_weights=False,
        )
        query_tokens = query_tokens + attended
        return query_tokens + self.ffn(self.ffn_norm(query_tokens))


def choose_attention_head_count(hidden_dim: int, preferred_max: int = 8) -> int:
    for candidate in range(min(preferred_max, hidden_dim), 0, -1):
        if hidden_dim % candidate == 0:
            return candidate
    return 1


def adaptive_sequence_mean_pool(tokens: torch.Tensor, output_length: int) -> torch.Tensor:
    """가변 patch token sequence를 고정 길이 visual token으로 압축한다.

    의도: DINOv3 입력 해상도에 따라 patch 수가 달라져도 LLM 앞에 붙는 visual token 수는 고정한다.
    선택 이유: LLM context 길이를 예측 가능하게 유지하고, Q-Former 이전 baseline을 단순하게 만든다.
    """

    if tokens.ndim != 3:
        raise ValueError("tokens shape는 [batch, sequence, dim]이어야 합니다.")
    batch_size, sequence_length, hidden_dim = tokens.shape
    if sequence_length == 0:
        raise ValueError("sequence_length는 1 이상이어야 합니다.")
    if output_length <= 0:
        raise ValueError("output_length는 1 이상이어야 합니다.")

    # torch.linspace로 구간 경계를 만들면 patch 수가 output_length보다 작아도 모든 output slot을 채울 수 있다.
    boundaries = torch.linspace(0, sequence_length, output_length + 1, device=tokens.device)
    pooled: list[torch.Tensor] = []
    for index in range(output_length):
        start = int(torch.floor(boundaries[index]).item())
        end = int(torch.ceil(boundaries[index + 1]).item())
        start = min(start, sequence_length - 1)
        end = max(start + 1, min(end, sequence_length))
        pooled.append(tokens[:, start:end, :].mean(dim=1))
    return torch.stack(pooled, dim=1).reshape(batch_size, output_length, hidden_dim)
