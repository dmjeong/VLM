from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mini_vlm.evaluation.dashboard import merge_prediction_with_sample, prediction_success


@dataclass(frozen=True)
class BenchmarkSummary:
    """고정 평가셋 성능 요약."""

    sample_count: int
    correct_count: int
    accuracy: float
    target_accuracy: float
    target_correct_count: int
    target_reached: bool
    by_source: dict[str, dict[str, float | int]]
    by_task: dict[str, dict[str, float | int]]


def summarize_predictions(
    *,
    predictions: list[dict[str, Any]],
    samples_by_id: dict[str, dict[str, Any]],
    target_accuracy: float = 0.9,
) -> tuple[BenchmarkSummary, list[dict[str, Any]]]:
    scored_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        sample = samples_by_id.get(str(prediction.get("sample_id")), {})
        merged = merge_prediction_with_sample(prediction, sample)
        merged["benchmark_success"] = prediction_success(merged)
        scored_rows.append(merged)

    sample_count = len(scored_rows)
    correct_count = sum(1 for row in scored_rows if bool(row.get("benchmark_success")))
    target_correct_count = int(target_accuracy * sample_count + 0.999999)
    summary = BenchmarkSummary(
        sample_count=sample_count,
        correct_count=correct_count,
        accuracy=(correct_count / sample_count) if sample_count else 0.0,
        target_accuracy=target_accuracy,
        target_correct_count=target_correct_count,
        target_reached=correct_count >= target_correct_count if sample_count else False,
        by_source=summarize_group(scored_rows, key=lambda row: str(row.get("metadata", {}).get("source") or "unknown")),
        by_task=summarize_group(scored_rows, key=lambda row: str(row.get("task") or "unknown")),
    )
    return summary, scored_rows


def summarize_group(rows: list[dict[str, Any]], *, key) -> dict[str, dict[str, float | int]]:
    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for row in rows:
        name = key(row)
        counts[name] += 1
        if bool(row.get("benchmark_success")):
            correct[name] += 1
    return {
        name: {
            "sample_count": counts[name],
            "correct_count": correct[name],
            "accuracy": correct[name] / counts[name],
        }
        for name in sorted(counts)
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def index_samples_by_id(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(sample.get("sample_id")): sample for sample in samples}


def summary_to_dict(summary: BenchmarkSummary) -> dict[str, Any]:
    return asdict(summary)

