from __future__ import annotations

from pathlib import Path

from mini_vlm.config import MiniVlmConfig


def load_checkpoint_into_model(model, config: MiniVlmConfig, checkpoint: str | Path) -> dict[str, str]:
    """학습 산출물 폴더의 adapter weight를 모델에 로드한다.

    의도: inference/evaluation 코드가 `visual_adapter.pt`와 `llm_lora/` 로딩 규칙을 중복 구현하지 않게 한다.
    참고: Stage 1은 visual adapter만 저장하고, Stage 2는 PEFT LoRA adapter를 `llm_lora/`에 저장한다.
    선택 이유: checkpoint 폴더 구조가 바뀌면 이 함수 하나만 고치면 된다.
    """

    import torch

    checkpoint_dir = Path(checkpoint)
    loaded: dict[str, str] = {}
    adapter_path = checkpoint_dir / "visual_adapter.pt"
    if not adapter_path.exists():
        adapter_path = latest_epoch_adapter_path(checkpoint_dir)
    if adapter_path.exists():
        state = torch.load(adapter_path, map_location="cpu")
        model.visual_adapter.load_state_dict(state)
        loaded["visual_adapter"] = str(adapter_path)

    lora_path = checkpoint_dir / "llm_lora"
    if config.use_lora and lora_path.exists():
        load_lora_adapter(model.llm, lora_path)
        loaded["llm_lora"] = str(lora_path)
    return loaded


def latest_epoch_adapter_path(checkpoint_dir: Path) -> Path:
    """최종 adapter가 없을 때 가장 마지막 epoch checkpoint를 찾는다."""

    candidates = sorted(
        checkpoint_dir.glob("visual_adapter_epoch_*.pt"),
        key=lambda path: epoch_number_from_adapter_path(path),
    )
    return candidates[-1] if candidates else checkpoint_dir / "visual_adapter.pt"


def epoch_number_from_adapter_path(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def load_lora_adapter(llm, lora_path: Path) -> None:
    """PEFT LoRA adapter를 inference/evaluation 모드로 붙인다."""

    if not hasattr(llm, "load_adapter"):
        raise RuntimeError("현재 LLM 객체가 PEFT adapter 로드를 지원하지 않습니다. config.use_lora를 확인하세요.")
    llm.load_adapter(str(lora_path), adapter_name="trained", is_trainable=False)
    if hasattr(llm, "set_adapter"):
        llm.set_adapter("trained")
