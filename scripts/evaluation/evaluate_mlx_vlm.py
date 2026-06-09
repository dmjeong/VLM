from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from mini_vlm.evaluation.benchmark import (
    index_samples_by_id,
    read_jsonl,
    summarize_predictions,
    summary_to_dict,
    write_json,
    write_jsonl,
)
from mini_vlm.evaluation.dashboard import is_yes_no_question
from mini_vlm.evaluation.evaluate_cli import score_generation, write_dashboard_if_possible, write_evaluation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX VLM을 Eval80 기준으로 평가")
    parser.add_argument("--model-id", default="mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
    parser.add_argument("--dataset", default="data/eval80/test.jsonl")
    parser.add_argument("--image-root", default="data/external_vlm_10k")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--prompt-style", choices=("standard", "strict"), default="strict")
    args = parser.parse_args()

    evaluate_mlx_vlm(
        model_id=args.model_id,
        dataset_path=Path(args.dataset),
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        prompt_style=args.prompt_style,
    )


def evaluate_mlx_vlm(
    *,
    model_id: str,
    dataset_path: Path,
    image_root: Path,
    output_dir: Path,
    max_samples: int,
    max_new_tokens: int,
    prompt_style: str,
) -> None:
    try:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
    except ImportError as exc:
        raise SystemExit("MLX VLM 평가는 `pip install mlx-vlm`이 필요합니다.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(dataset_path)[:max_samples]
    print(f"MLX 모델 로드: {model_id}", flush=True)
    model, processor = load(model_id)

    predictions_path = output_dir / "test_predictions.jsonl"
    predictions: list[dict[str, Any]] = []
    with predictions_path.open("w", encoding="utf-8") as file:
        for index, sample in enumerate(samples, start=1):
            image_path = resolve_image_path(image_root=image_root, image=str(sample["image"]))
            prompt = build_question(
                str(sample["question"]),
                str(sample.get("task") or ""),
                prompt_style=prompt_style,
            )
            formatted_prompt = apply_chat_template(processor, model.config, prompt, num_images=1)
            result = generate(
                model,
                processor,
                formatted_prompt,
                image=str(image_path),
                max_tokens=max_new_tokens,
                temperature=0.0,
                skip_special_tokens=True,
                verbose=False,
            )
            generated = clean_generated_text(result.text)
            score = score_generation(generated, str(sample["answer"]))
            row = {
                "sample_id": sample["sample_id"],
                "image": sample["image"],
                "question": sample["question"],
                "expected_answer": sample["answer"],
                "generated_answer": generated,
                **score,
            }
            predictions.append(row)
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            print(f"[{index}/{len(samples)}] {sample['sample_id']} -> {generated[:80]}", flush=True)

    benchmark_summary, scored = summarize_predictions(
        predictions=predictions,
        samples_by_id=index_samples_by_id(samples),
        target_accuracy=0.9,
    )
    summary = {
        "split": "test",
        "dataset_path": str(dataset_path),
        "checkpoint": model_id,
        "prompt_style": prompt_style,
        "sample_count": len(samples),
        "generation_sample_count": len(predictions),
        "generation_unique_image_count": len({row["image"] for row in predictions}),
        "avg_loss": None,
        "exact_match_rate": mean_bool(predictions, "exact_match"),
        "contains_answer_rate": mean_bool(predictions, "contains_answer"),
        "avg_token_overlap": mean_float(predictions, "token_overlap"),
        "benchmark_accuracy": benchmark_summary.accuracy,
        "benchmark_correct_count": benchmark_summary.correct_count,
        "benchmark_target_reached": benchmark_summary.target_reached,
    }
    summary_path = output_dir / "test_summary.json"
    write_json(summary_path, summary)
    write_json(output_dir / "benchmark_summary.json", summary_to_dict(benchmark_summary))
    write_jsonl(output_dir / "benchmark_scored_predictions.jsonl", scored)
    write_evaluation_report(summary, output_dir / "test_report.md")
    write_dashboard_if_possible(
        summary_path=summary_path,
        predictions_path=predictions_path,
        dataset_path=dataset_path,
        image_root=image_root,
        dashboard_path=output_dir / "dashboard.html",
    )
    print(
        "Eval80 benchmark: "
        f"{benchmark_summary.correct_count}/{benchmark_summary.sample_count} "
        f"({benchmark_summary.accuracy * 100:.1f}%), "
        f"target={benchmark_summary.target_correct_count}, reached={benchmark_summary.target_reached}",
        flush=True,
    )
    if not benchmark_summary.target_reached:
        raise SystemExit(2)


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if bool(row.get(key))) / len(rows)


def mean_float(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key) or 0.0) for row in rows if math.isfinite(float(row.get(key) or 0.0))]
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_question(question: str, task: str, *, prompt_style: str = "standard") -> str:
    if prompt_style == "strict":
        if task == "multiple-choice-vqa":
            return question + "\nChoose exactly one option. Respond with only the option letter and option text."
        if "count" in task:
            return question + "\nCount carefully. Respond with only the final number."
        if is_yes_no_question(question):
            return question + "\nRespond with exactly yes or no."
        return question + "\nRespond with a concise answer phrase using only visible evidence."
    if task == "multiple-choice-vqa":
        return question + "\nRespond with only the option letter and answer text."
    return question + "\nAnswer concisely."


def resolve_image_path(*, image_root: Path, image: str) -> Path:
    image_path = Path(image)
    return image_path if image_path.is_absolute() else image_root / image_path


def clean_generated_text(text: str) -> str:
    for token in ("<|endoftext|>", "<|im_end|>", "</s>"):
        text = text.replace(token, " ")
    return " ".join(text.split()).strip()


if __name__ == "__main__":
    main()
