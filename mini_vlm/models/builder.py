from __future__ import annotations

from dataclasses import dataclass

from mini_vlm.config import MiniVlmConfig
from mini_vlm.models.mini_vlm import MiniVlmForConditionalGeneration
from mini_vlm.models.vision_encoder import DinoVisionEncoder, LocalDinov3ImageProcessor
from mini_vlm.models.visual_adapter import MlpVisualAdapter, PerceiverResamplerAdapter, QFormerVisualAdapter


@dataclass(frozen=True)
class BuiltMiniVlm:
    model: MiniVlmForConditionalGeneration
    tokenizer: object
    image_processor: object


def build_mini_vlm(config: MiniVlmConfig) -> BuiltMiniVlm:
    """설정에서 실제 DINOv3 + LLM mini VLM을 만든다.

    의도: 학습/추론 스크립트가 모델 생성 세부사항을 중복 구현하지 않게 한다.
    참고: 설계서 6장 모델 모듈 설계.
    선택 이유: DINOv3 hidden dim과 LLM embedding dim은 실제 로딩 후 알 수 있으므로 builder가 두 값을
    읽어 adapter를 구성하는 책임을 가진다.
    """

    try:
        from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("실제 모델을 빌드하려면 transformers와 torch가 필요합니다.") from exc

    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if config.vision_backend == "hf":
        image_processor = AutoImageProcessor.from_pretrained(config.vision_model_id)
    else:
        image_processor = LocalDinov3ImageProcessor(config.vision_image_size)
    vision_encoder = DinoVisionEncoder(
        config.vision_model_id,
        freeze=config.freeze_vision,
        backend=config.vision_backend,
        repo_dir=config.vision_repo_dir,
        model_name=config.vision_model_name,
        weights=config.vision_weights,
    )
    llm = AutoModelForCausalLM.from_pretrained(config.llm_model_id, trust_remote_code=True)
    if config.use_lora:
        llm = apply_lora_to_llm(llm, config)
    elif config.freeze_llm:
        llm.requires_grad_(False)
        llm.eval()

    vision_dim = int(vision_encoder.hidden_size)
    llm_dim = int(llm.get_input_embeddings().embedding_dim)
    visual_adapter = build_visual_adapter(config, vision_dim=vision_dim, llm_dim=llm_dim)
    model = MiniVlmForConditionalGeneration(
        vision_encoder=vision_encoder,
        visual_adapter=visual_adapter,
        llm=llm,
        visual_token_count=config.visual_token_count,
        freeze_vision=config.freeze_vision,
        freeze_llm=config.freeze_llm and not config.use_lora,
    )
    return BuiltMiniVlm(model=model, tokenizer=tokenizer, image_processor=image_processor)


def build_visual_adapter(config: MiniVlmConfig, *, vision_dim: int, llm_dim: int):
    if config.adapter_type == "mlp":
        return MlpVisualAdapter(
            vision_dim=vision_dim,
            llm_dim=llm_dim,
            visual_token_count=config.visual_token_count,
            hidden_dim=config.adapter_hidden_dim,
        )
    if config.adapter_type == "perceiver":
        return PerceiverResamplerAdapter(
            vision_dim=vision_dim,
            llm_dim=llm_dim,
            visual_token_count=config.visual_token_count,
            hidden_dim=config.adapter_hidden_dim,
            layer_count=config.adapter_layer_count,
        )
    if config.adapter_type == "qformer":
        return QFormerVisualAdapter(
            vision_dim=vision_dim,
            llm_dim=llm_dim,
            visual_token_count=config.visual_token_count,
            hidden_dim=config.adapter_hidden_dim,
            layer_count=config.adapter_layer_count,
        )
    raise NotImplementedError(f"지원하지 않는 adapter_type입니다: {config.adapter_type}")


def apply_lora_to_llm(llm, config: MiniVlmConfig):
    """Qwen 본체는 유지하고 LoRA adapter만 학습 가능하게 만든다."""

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("LoRA를 사용하려면 peft가 필요합니다. `pip install '.[model]'` 후 다시 실행하세요.") from exc
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
    )
    return get_peft_model(llm, peft_config)
