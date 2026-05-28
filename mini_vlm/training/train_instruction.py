from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

from mini_vlm.config import MiniVlmConfig, load_config


@dataclass(frozen=True)
class InstructionTrainingPlan:
    """Stage 2 instruction tuning 실행 전 확인할 계약.

    의도: Stage 1 adapter alignment가 검증되기 전에는 instruction tuning을 실제로 돌리지 않는다.
    참고: 설계서 8.3 Stage 2 Visual Instruction Tuning.
    선택 이유: VLM은 Stage 1의 visual-token 정렬이 실패하면 Stage 2에서 loss가 줄어도 이미지 grounding이
    좋아졌는지 판단하기 어렵다. 따라서 Stage 2 진입점은 만들되, 지금은 명시적인 gate로 둔다.
    """

    experiment_name: str
    adapter_checkpoint_required: bool
    trainable_parts: tuple[str, ...]
    blocked_reason: str


def build_instruction_training_plan(config: MiniVlmConfig) -> InstructionTrainingPlan:
    trainable_parts = ("visual_adapter",)
    if config.use_lora:
        trainable_parts = trainable_parts + ("llm_lora",)
    return InstructionTrainingPlan(
        experiment_name=f"{config.experiment_name}-instruction",
        adapter_checkpoint_required=True,
        trainable_parts=trainable_parts,
        blocked_reason="Stage 1 tiny overfit과 실제 inference smoke가 끝난 뒤 instruction tuning을 실행한다.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv3 Mini VLM Stage 2 instruction tuning 준비")
    parser.add_argument("--config", default="configs/dinov3-mini-vlm.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    plan = build_instruction_training_plan(config)
    if args.dry_run:
        print("Stage 2 instruction tuning 계획")
        for key, value in asdict(plan).items():
            print(f"{key}: {value}")
        return

    raise SystemExit(
        "Stage 2는 아직 실행하지 않습니다. 먼저 Stage 1 adapter checkpoint, tiny overfit, 실제 추론 smoke를 "
        "완료한 뒤 이 스크립트의 실행 gate를 열어주세요."
    )


if __name__ == "__main__":
    main()
