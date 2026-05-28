from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


AdapterType = Literal["mlp", "qformer", "perceiver"]
VisionBackend = Literal["hf", "torchhub"]


@dataclass(frozen=True)
class MiniVlmConfig:
    """mini VLM 실험 전체 설정.

    의도: 모델명, visual token 수, freeze 정책, 데이터 경로를 코드에 흩뿌리지 않고 하나의 재현 가능한
    설정으로 묶는다.
    참고: docs/02-design/features/dinov3-mini-vlm.design.md#4-설정-설계
    선택 이유: VLM 실험은 작은 차이로도 결과가 바뀐다. 설정을 dataclass로 고정해야 실험 기록과
    checkpoint를 나중에 정확히 연결할 수 있다.
    """

    experiment_name: str = "dinov3-mini-vlm-mlp-baseline"
    vision_backend: VisionBackend = "hf"
    vision_model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    vision_repo_dir: str = ""
    vision_model_name: str = "dinov3_vits16"
    vision_weights: str = ""
    vision_image_size: int = 224
    llm_model_id: str = "Qwen/Qwen3-0.6B"
    adapter_type: AdapterType = "mlp"
    visual_token_count: int = 32
    adapter_hidden_dim: int = 1024
    max_text_length: int = 256
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    seed: int = 42
    freeze_vision: bool = True
    freeze_llm: bool = True
    use_lora: bool = False
    device: str = "auto"
    max_grad_norm: float = 1.0
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    max_new_tokens: int = 32
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 3
    stop_strings: tuple[str, ...] = ("\nQuestion:", "\nAnswer:")
    train_jsonl: str = "data/samples/train.jsonl"
    validation_jsonl: str = "data/samples/validation.jsonl"
    test_jsonl: str = ""
    image_root: str = "data/samples"
    output_dir: str = "artifacts/dinov3-mini-vlm/mlp-baseline"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MiniVlmConfig":
        adapter_type = str(payload.get("adapter_type", cls.adapter_type))
        if adapter_type not in {"mlp", "qformer", "perceiver"}:
            raise ValueError("adapter_type은 mlp, qformer, perceiver 중 하나여야 합니다.")
        vision_backend = str(payload.get("vision_backend", cls.vision_backend))
        if vision_backend not in {"hf", "torchhub"}:
            raise ValueError("vision_backend는 hf 또는 torchhub 중 하나여야 합니다.")
        return cls(
            experiment_name=str(payload.get("experiment_name", cls.experiment_name)),
            vision_backend=vision_backend,  # type: ignore[arg-type]
            vision_model_id=str(payload.get("vision_model_id", cls.vision_model_id)),
            vision_repo_dir=str(payload.get("vision_repo_dir", cls.vision_repo_dir)),
            vision_model_name=str(payload.get("vision_model_name", cls.vision_model_name)),
            vision_weights=str(payload.get("vision_weights", cls.vision_weights)),
            vision_image_size=int(payload.get("vision_image_size", cls.vision_image_size)),
            llm_model_id=str(payload.get("llm_model_id", cls.llm_model_id)),
            adapter_type=adapter_type,  # type: ignore[arg-type]
            visual_token_count=int(payload.get("visual_token_count", cls.visual_token_count)),
            adapter_hidden_dim=int(payload.get("adapter_hidden_dim", cls.adapter_hidden_dim)),
            max_text_length=int(payload.get("max_text_length", cls.max_text_length)),
            train_batch_size=int(payload.get("train_batch_size", cls.train_batch_size)),
            gradient_accumulation_steps=int(
                payload.get("gradient_accumulation_steps", cls.gradient_accumulation_steps)
            ),
            learning_rate=float(payload.get("learning_rate", cls.learning_rate)),
            num_train_epochs=int(payload.get("num_train_epochs", cls.num_train_epochs)),
            seed=int(payload.get("seed", cls.seed)),
            freeze_vision=bool(payload.get("freeze_vision", cls.freeze_vision)),
            freeze_llm=bool(payload.get("freeze_llm", cls.freeze_llm)),
            use_lora=bool(payload.get("use_lora", cls.use_lora)),
            device=str(payload.get("device", cls.device)),
            max_grad_norm=float(payload.get("max_grad_norm", cls.max_grad_norm)),
            lora_r=int(payload.get("lora_r", cls.lora_r)),
            lora_alpha=int(payload.get("lora_alpha", cls.lora_alpha)),
            lora_dropout=float(payload.get("lora_dropout", cls.lora_dropout)),
            lora_target_modules=_coerce_string_tuple(
                payload.get("lora_target_modules", cls.lora_target_modules),
                "lora_target_modules",
            ),
            max_new_tokens=int(payload.get("max_new_tokens", cls.max_new_tokens)),
            repetition_penalty=float(payload.get("repetition_penalty", cls.repetition_penalty)),
            no_repeat_ngram_size=int(payload.get("no_repeat_ngram_size", cls.no_repeat_ngram_size)),
            stop_strings=_coerce_string_tuple(payload.get("stop_strings", cls.stop_strings), "stop_strings"),
            train_jsonl=str(payload.get("train_jsonl", cls.train_jsonl)),
            validation_jsonl=str(payload.get("validation_jsonl", cls.validation_jsonl)),
            test_jsonl=str(payload.get("test_jsonl", cls.test_jsonl)),
            image_root=str(payload.get("image_root", cls.image_root)),
            output_dir=str(payload.get("output_dir", cls.output_dir)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_path(self, value: str, root: Path | None = None) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (root or Path.cwd()) / path


def load_config(path: str | Path) -> MiniVlmConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError("설정 파일의 최상위 값은 객체여야 합니다.")
    return MiniVlmConfig.from_dict(payload)


def _coerce_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raise TypeError(f"{field_name} 값은 문자열 또는 문자열 목록이어야 합니다.")
