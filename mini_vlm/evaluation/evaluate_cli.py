from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mini_vlm.config import MiniVlmConfig, load_config
from mini_vlm.utils.checkpoints import write_json


@dataclass(frozen=True)
class EvaluationPaths:
    """평가 실행 후 생성되는 파일 경로 묶음."""

    summary_path: Path
    predictions_path: Path
    report_path: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="학습된 DINOv3 Mini VLM checkpoint 평가")
    parser.add_argument("--config", default="configs/dinov3-mini-vlm.json")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--jsonl", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-generation-samples", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = args.checkpoint or config.output_dir
    dataset_path = resolve_split_path(config, split=args.split, override_jsonl=args.jsonl)
    output_dir = Path(args.output_dir) if args.output_dir else Path(checkpoint) / "evaluation"

    if args.dry_run:
        print(f"실험명: {config.experiment_name}")
        print(f"checkpoint: {checkpoint}")
        print(f"split: {args.split}")
        print(f"dataset: {dataset_path}")
        print(f"output_dir: {output_dir}")
        print("dry-run: 실제 모델 평가는 실행하지 않았습니다.")
        return

    run_evaluation(
        config=config,
        checkpoint=checkpoint,
        split=args.split,
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_generation_samples=args.max_generation_samples,
    )


def run_evaluation(
    *,
    config: MiniVlmConfig,
    checkpoint: str | Path,
    split: str,
    dataset_path: str | Path,
    output_dir: str | Path,
    max_generation_samples: int,
) -> EvaluationPaths:
    try:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("실제 평가는 torch와 pillow가 필요합니다. `pip install '.[model]'` 후 다시 실행하세요.") from exc

    from mini_vlm.data import MiniVlmCollator, MiniVlmDataset
    from mini_vlm.models.builder import build_mini_vlm
    from mini_vlm.models.generation import greedy_generate_from_visual_prefix
    from mini_vlm.utils.device import select_torch_device
    from mini_vlm.utils.model_loading import load_checkpoint_into_model

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    built = build_mini_vlm(config)
    loaded_files = load_checkpoint_into_model(built.model, config, checkpoint)
    device = select_torch_device(config.device)
    model = built.model.to(device)
    model.eval()

    dataset = MiniVlmDataset(dataset_path, image_root=config.image_root)
    collator = MiniVlmCollator(
        tokenizer=built.tokenizer,
        image_processor=built.image_processor,
        image_root=config.image_root,
        max_text_length=config.max_text_length,
    )
    data_loader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=False, collate_fn=collator)
    loss_summary = evaluate_loss(model=model, data_loader=data_loader, device=device)

    predictions_path = output_path / f"{split}_predictions.jsonl"
    generation_summary = generate_prediction_samples(
        model=model,
        tokenizer=built.tokenizer,
        image_processor=built.image_processor,
        dataset=dataset,
        device=device,
        output_path=predictions_path,
        max_samples=max_generation_samples,
        max_new_tokens=config.max_new_tokens,
        repetition_penalty=config.repetition_penalty,
        no_repeat_ngram_size=config.no_repeat_ngram_size,
        stop_strings=config.stop_strings,
        image_class=Image,
        generate_fn=greedy_generate_from_visual_prefix,
        torch_module=torch,
    )
    summary = {
        "split": split,
        "dataset_path": str(dataset_path),
        "checkpoint": str(checkpoint),
        "loaded_files": loaded_files,
        "sample_count": len(dataset),
        **loss_summary,
        **generation_summary,
    }
    summary_path = output_path / f"{split}_summary.json"
    report_path = output_path / f"{split}_report.md"
    dashboard_path = output_path / "dashboard.html"
    write_json(summary, summary_path)
    write_evaluation_report(summary, report_path)
    write_dashboard_if_possible(
        summary_path=summary_path,
        predictions_path=predictions_path,
        dataset_path=dataset_path,
        image_root=config.image_root,
        dashboard_path=dashboard_path,
    )

    print(f"평가 완료: {output_path}", flush=True)
    print(f"summary: {summary_path}", flush=True)
    print(f"predictions: {predictions_path}", flush=True)
    print(f"report: {report_path}", flush=True)
    print(f"dashboard: {dashboard_path}", flush=True)
    return EvaluationPaths(summary_path=summary_path, predictions_path=predictions_path, report_path=report_path)


def resolve_split_path(config: MiniVlmConfig, *, split: str, override_jsonl: str = "") -> str:
    if override_jsonl:
        return override_jsonl
    if split == "train":
        return config.train_jsonl
    if split == "validation":
        return config.validation_jsonl
    if split == "test":
        if not config.test_jsonl:
            raise ValueError("config.test_jsonl이 비어 있습니다. --jsonl로 평가 파일을 직접 지정하세요.")
        return config.test_jsonl
    raise ValueError(f"알 수 없는 split입니다: {split}")


def evaluate_loss(*, model, data_loader, device) -> dict[str, float | int]:
    import torch

    losses: list[float] = []
    with torch.no_grad():
        for batch in data_loader:
            output = model(
                pixel_values=batch.pixel_values.to(device),
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
                labels=batch.labels.to(device),
            )
            if output.loss is None:
                raise RuntimeError("평가 중 LLM output에 loss가 없습니다.")
            loss_value = float(output.loss.item())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"평가 loss가 비정상 값입니다: {loss_value}")
            losses.append(loss_value)
    if not losses:
        raise RuntimeError("평가 데이터가 비어 있습니다.")
    return {
        "batch_count": len(losses),
        "avg_loss": sum(losses) / len(losses),
        "min_loss": min(losses),
        "max_loss": max(losses),
    }


def generate_prediction_samples(
    *,
    model,
    tokenizer,
    image_processor,
    dataset,
    device,
    output_path: Path,
    max_samples: int,
    max_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    stop_strings: tuple[str, ...],
    image_class,
    generate_fn,
    torch_module,
) -> dict[str, float | int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = max(0, min(max_samples, len(dataset)))
    scores: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as file:
        for index in range(count):
            sample = dataset[index]
            image = image_class.open(dataset.image_path_for(sample)).convert("RGB")
            processed = image_processor(images=[image], return_tensors="pt")
            pixel_values = processed["pixel_values"] if isinstance(processed, dict) else processed.pixel_values
            prompt = f"Question: {sample.question}\nAnswer:"
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch_module.tensor([prompt_ids], dtype=torch_module.long, device=device)
            result = generate_fn(
                model=model,
                tokenizer=tokenizer,
                pixel_values=pixel_values.to(device),
                prompt_input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                stop_strings=stop_strings,
            )
            score = score_generation(result.text, sample.answer)
            scores.append(score)
            file.write(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "image": sample.image,
                        "question": sample.question,
                        "expected_answer": sample.answer,
                        "generated_answer": result.text,
                        **score,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    if not scores:
        return {
            "generation_sample_count": 0,
            "exact_match_rate": 0.0,
            "contains_answer_rate": 0.0,
            "avg_token_overlap": 0.0,
        }
    return {
        "generation_sample_count": len(scores),
        "exact_match_rate": sum(1 for score in scores if score["exact_match"]) / len(scores),
        "contains_answer_rate": sum(1 for score in scores if score["contains_answer"]) / len(scores),
        "avg_token_overlap": sum(float(score["token_overlap"]) for score in scores) / len(scores),
    }


def score_generation(generated: str, expected: str) -> dict[str, bool | float]:
    normalized_generated = normalize_answer(generated)
    normalized_expected = normalize_answer(expected)
    expected_tokens = normalized_expected.split()
    generated_tokens = normalized_generated.split()
    if expected_tokens:
        overlap = len(set(expected_tokens) & set(generated_tokens)) / len(set(expected_tokens))
    else:
        overlap = 0.0
    return {
        "exact_match": bool(normalized_expected and normalized_generated == normalized_expected),
        "contains_answer": bool(normalized_expected and normalized_expected in normalized_generated),
        "token_overlap": overlap,
    }


def normalize_answer(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^0-9a-z가-힣]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def write_evaluation_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# 모델 평가 리포트",
        "",
        f"- split: `{summary.get('split')}`",
        f"- dataset: `{summary.get('dataset_path')}`",
        f"- checkpoint: `{summary.get('checkpoint')}`",
        f"- sample 수: `{summary.get('sample_count')}`",
        f"- 평균 loss: `{_format_metric(summary.get('avg_loss'))}`",
        f"- 최소/최대 loss: `{_format_metric(summary.get('min_loss'))}` / `{_format_metric(summary.get('max_loss'))}`",
        f"- 생성 샘플 수: `{summary.get('generation_sample_count')}`",
        f"- exact match rate: `{_format_metric(summary.get('exact_match_rate'))}`",
        f"- contains answer rate: `{_format_metric(summary.get('contains_answer_rate'))}`",
        f"- 평균 token overlap: `{_format_metric(summary.get('avg_token_overlap'))}`",
        "",
        "## 해석 메모",
        "",
        "- `avg_loss`는 정답 토큰을 얼마나 잘 예측하는지 보는 기본 지표다.",
        "- `exact_match_rate`는 생성 답변이 정답과 완전히 같은 비율이라 매우 엄격하다.",
        "- `contains_answer_rate`와 `token_overlap`은 짧은 template QA에서 빠르게 방향성을 보는 보조 지표다.",
        "- 실제 품질 판단은 `*_predictions.jsonl`의 질문, 정답, 생성 답변을 함께 확인해야 한다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dashboard_if_possible(
    *,
    summary_path: Path,
    predictions_path: Path,
    dataset_path: str | Path,
    image_root: str | Path,
    dashboard_path: Path,
) -> None:
    """평가 리포트와 함께 HTML 대시보드를 생성한다.

    의도: 평가 실패가 HTML 생성 실패로 이어져 핵심 평가 결과까지 잃지 않도록 dashboard 단계만 분리한다.
    """

    from mini_vlm.evaluation.dashboard import write_evaluation_dashboard

    write_evaluation_dashboard(
        summary_path=summary_path,
        predictions_path=predictions_path,
        dataset_path=dataset_path,
        image_root=image_root,
        output_path=dashboard_path,
    )


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "N/A"


if __name__ == "__main__":
    main()
