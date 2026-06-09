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
)
from mini_vlm.evaluation.dashboard import is_yes_no_question
from mini_vlm.evaluation.evaluate_cli import score_generation, write_dashboard_if_possible, write_evaluation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Hugging Face pretrained VLM을 Eval80 기준으로 평가")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
    parser.add_argument("--dataset", default="data/eval80/test.jsonl")
    parser.add_argument("--image-root", default="data/external_vlm_10k")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--prompt-style", choices=("standard", "strict"), default="standard")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"model_id: {args.model_id}")
        print(f"dataset: {args.dataset}")
        print(f"image_root: {args.image_root}")
        print(f"output_dir: {args.output_dir}")
        return

    evaluate_pretrained_vlm(
        model_id=args.model_id,
        dataset_path=Path(args.dataset),
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        device_name=args.device,
        prompt_style=args.prompt_style,
    )


def evaluate_pretrained_vlm(
    *,
    model_id: str,
    dataset_path: Path,
    image_root: Path,
    output_dir: Path,
    max_samples: int,
    max_new_tokens: int,
    device_name: str,
    prompt_style: str,
) -> None:
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise SystemExit("pretrained VLM 평가는 torch, pillow, transformers가 필요합니다.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(dataset_path)[:max_samples]
    device = resolve_device(torch, device_name)
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    print(f"모델 로드: {model_id} device={device} dtype={dtype}", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    predictions_path = output_dir / "test_predictions.jsonl"
    predictions: list[dict[str, Any]] = []
    with predictions_path.open("w", encoding="utf-8") as file:
        for index, sample in enumerate(samples, start=1):
            image_path = resolve_image_path(image_root=image_root, image=str(sample["image"]))
            image = Image.open(image_path).convert("RGB")
            question = build_question(
                str(sample["question"]),
                str(sample.get("task") or ""),
                prompt_style=prompt_style,
            )
            generated = generate_answer(
                model=model,
                processor=processor,
                image=image,
                question=question,
                device=device,
                max_new_tokens=max_new_tokens,
            )
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

    exact_rate = mean_bool(predictions, "exact_match")
    contains_rate = mean_bool(predictions, "contains_answer")
    avg_overlap = mean_float(predictions, "token_overlap")
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
        "exact_match_rate": exact_rate,
        "contains_answer_rate": contains_rate,
        "avg_token_overlap": avg_overlap,
        "benchmark_accuracy": benchmark_summary.accuracy,
        "benchmark_correct_count": benchmark_summary.correct_count,
        "benchmark_target_reached": benchmark_summary.target_reached,
    }
    summary_path = output_dir / "test_summary.json"
    report_path = output_dir / "test_report.md"
    write_json(summary_path, summary)
    write_json(output_dir / "benchmark_summary.json", summary_to_dict(benchmark_summary))
    write_scored_predictions(output_dir / "benchmark_scored_predictions.jsonl", scored)
    write_evaluation_report(summary, report_path)
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


def generate_answer(*, model, processor, image, question: str, device, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    if hasattr(processor, "apply_chat_template"):
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    else:
        prompt = f"Question: {question}\nAnswer:"
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_length = int(inputs["input_ids"].shape[1])
    new_ids = generated_ids[:, prompt_length:]
    text = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip()


def resolve_device(torch_module, device_name: str):
    if device_name == "auto":
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")
    return torch_module.device(device_name)


def resolve_image_path(*, image_root: Path, image: str) -> Path:
    image_path = Path(image)
    return image_path if image_path.is_absolute() else image_root / image_path


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if bool(row.get(key))) / len(rows)


def mean_float(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key) or 0.0) for row in rows if math.isfinite(float(row.get(key) or 0.0))]
    if not values:
        return 0.0
    return sum(values) / len(values)


def write_scored_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
