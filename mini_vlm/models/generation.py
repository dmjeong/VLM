from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GenerationResult:
    token_ids: list[int]
    text: str


def greedy_generate_from_visual_prefix(
    *,
    model,
    tokenizer,
    pixel_values: torch.Tensor,
    prompt_input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    stop_strings: tuple[str, ...] = (),
) -> GenerationResult:
    """visual token prefix와 prompt embedding을 사용해 수동 greedy decoding을 수행한다.

    의도: Hugging Face 모델별 `generate(inputs_embeds=...)` 지원 차이를 피한다.
    참고: 설계서 9.2 Generation 방식의 fallback 구현.
    선택 이유: mini VLM은 text token id 앞에 실제 vocabulary token이 아닌 visual embedding을 붙인다.
    수동 decoding은 느리지만 구조를 이해하기 쉽고 모델 호환성 문제가 적다.
    """

    if hasattr(model, "eval"):
        model.eval()
    generated: list[int] = []
    with torch.no_grad():
        features = model.vision_encoder(pixel_values)
        visual_tokens = model.visual_adapter(features.patch_tokens, features.cls_token)
        text_embeddings = model.llm.get_input_embeddings()(prompt_input_ids)
        if visual_tokens.dtype != text_embeddings.dtype:
            visual_tokens = visual_tokens.to(dtype=text_embeddings.dtype)
        inputs_embeds = torch.cat([visual_tokens, text_embeddings], dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=inputs_embeds.device,
        )

        eos_token_id = tokenizer.eos_token_id
        for _ in range(max_new_tokens):
            outputs = model.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :].clone()
            apply_repetition_penalty(logits, generated, repetition_penalty)
            apply_no_repeat_ngram_mask(logits, generated, no_repeat_ngram_size)
            next_token_id = int(logits.argmax(dim=-1).item())
            if eos_token_id is not None and next_token_id == eos_token_id:
                break
            generated.append(next_token_id)
            decoded_so_far = tokenizer.decode(generated, skip_special_tokens=True)
            if contains_stop_string(decoded_so_far, stop_strings):
                break
            next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=inputs_embeds.device)
            next_embedding = model.llm.get_input_embeddings()(next_token)
            inputs_embeds = torch.cat([inputs_embeds, next_embedding], dim=1)
            next_attention = torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, next_attention], dim=1)

    text = trim_stop_strings(tokenizer.decode(generated, skip_special_tokens=True), stop_strings)
    return GenerationResult(token_ids=generated, text=text)


def apply_repetition_penalty(logits: torch.Tensor, generated_token_ids: list[int], penalty: float) -> None:
    if penalty <= 1.0 or not generated_token_ids:
        return
    for token_id in set(generated_token_ids):
        token_logit = logits[:, token_id]
        logits[:, token_id] = torch.where(token_logit < 0, token_logit * penalty, token_logit / penalty)


def apply_no_repeat_ngram_mask(logits: torch.Tensor, generated_token_ids: list[int], ngram_size: int) -> None:
    banned_tokens = get_banned_ngram_next_tokens(generated_token_ids, ngram_size)
    if banned_tokens:
        logits[:, sorted(banned_tokens)] = float("-inf")


def get_banned_ngram_next_tokens(generated_token_ids: list[int], ngram_size: int) -> set[int]:
    if ngram_size <= 0 or len(generated_token_ids) + 1 < ngram_size:
        return set()
    prefix_length = ngram_size - 1
    current_prefix = tuple(generated_token_ids[-prefix_length:])
    banned_tokens: set[int] = set()
    for index in range(len(generated_token_ids) - ngram_size + 1):
        previous_prefix = tuple(generated_token_ids[index : index + prefix_length])
        if previous_prefix == current_prefix:
            banned_tokens.add(generated_token_ids[index + prefix_length])
    return banned_tokens


def contains_stop_string(text: str, stop_strings: tuple[str, ...]) -> bool:
    return any(stop_string and stop_string in text for stop_string in stop_strings)


def trim_stop_strings(text: str, stop_strings: tuple[str, ...]) -> str:
    cut_index = len(text)
    for stop_string in stop_strings:
        if not stop_string:
            continue
        found_index = text.find(stop_string)
        if found_index >= 0:
            cut_index = min(cut_index, found_index)
    return text[:cut_index].strip()
