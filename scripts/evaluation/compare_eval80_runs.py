from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mini_vlm.evaluation.benchmark import read_jsonl, write_json


DEFAULT_RUNS = [
    "dino-qwen-lora-stage2",
    "smolvlm2-256m",
    "qwen2_5-vl-3b",
    "qwen2_5-vl-3b-strict",
    "mlx-qwen2_5-vl-7b-4bit-strict",
    "mlx-internvl3-8b-4bit-strict",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval80 run 성능 비교 리포트 생성")
    parser.add_argument("--root", default="artifacts/dinov3-mini-vlm/eval80")
    parser.add_argument("--dataset-quality", default="data/eval80/quality_report.json")
    parser.add_argument("--output-md", default="artifacts/dinov3-mini-vlm/eval80/leaderboard.md")
    parser.add_argument("--output-json", default="artifacts/dinov3-mini-vlm/eval80/leaderboard.json")
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    args = parser.parse_args()

    root = Path(args.root)
    quality_ids = load_quality_issue_ids(Path(args.dataset_quality))
    runs = [load_run(root, run_name) for run_name in args.runs if (root / run_name / "benchmark_summary.json").exists()]
    union = calculate_union(root=root, run_names=[run["name"] for run in runs], excluded_ids=set())
    clean_union = calculate_union(root=root, run_names=[run["name"] for run in runs], excluded_ids=quality_ids)
    payload = {
        "runs": runs,
        "oracle_union": union,
        "clean_oracle_union": clean_union,
        "quality_issue_sample_ids": sorted(quality_ids),
    }
    write_json(args.output_json, payload)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(render_markdown(payload), encoding="utf-8")
    print(f"leaderboard: {args.output_md}")
    print(f"json: {args.output_json}")


def load_run(root: Path, name: str) -> dict[str, Any]:
    summary = json.loads((root / name / "benchmark_summary.json").read_text(encoding="utf-8"))
    return {
        "name": name,
        "correct_count": summary["correct_count"],
        "sample_count": summary["sample_count"],
        "accuracy": summary["accuracy"],
        "target_reached": summary["target_reached"],
        "by_task": summary.get("by_task", {}),
        "dashboard": str(root / name / "dashboard.html"),
    }


def calculate_union(*, root: Path, run_names: list[str], excluded_ids: set[str]) -> dict[str, Any]:
    rows_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    all_ids: set[str] = set()
    for run_name in run_names:
        predictions = read_jsonl(root / run_name / "benchmark_scored_predictions.jsonl")
        indexed = {str(row["sample_id"]): row for row in predictions}
        rows_by_run[run_name] = indexed
        all_ids.update(indexed)
    candidate_ids = sorted(all_ids - excluded_ids)
    correct_ids = [
        sample_id
        for sample_id in candidate_ids
        if any(bool(rows_by_run[run_name].get(sample_id, {}).get("benchmark_success")) for run_name in run_names)
    ]
    missed_ids = [sample_id for sample_id in candidate_ids if sample_id not in set(correct_ids)]
    sample_count = len(candidate_ids)
    return {
        "correct_count": len(correct_ids),
        "sample_count": sample_count,
        "accuracy": len(correct_ids) / sample_count if sample_count else 0.0,
        "missed_sample_ids": missed_ids,
    }


def load_quality_issue_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["sample_id"]) for row in payload.get("findings", []) if row.get("sample_id")}


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Eval80 리더보드",
        "",
        "| run | correct | accuracy | target | dashboard |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for run in sorted(payload["runs"], key=lambda item: item["accuracy"], reverse=True):
        lines.append(
            "| {name} | {correct}/{total} | {accuracy:.1%} | {target} | `{dashboard}` |".format(
                name=run["name"],
                correct=run["correct_count"],
                total=run["sample_count"],
                accuracy=run["accuracy"],
                target="달성" if run["target_reached"] else "미달",
                dashboard=run["dashboard"],
            )
        )
    union = payload["oracle_union"]
    clean_union = payload["clean_oracle_union"]
    lines.extend(
        [
            "",
            "## Oracle Union",
            "",
            f"- 전체 Eval80: {union['correct_count']}/{union['sample_count']} ({union['accuracy']:.1%})",
            f"- 품질 이슈 제외: {clean_union['correct_count']}/{clean_union['sample_count']} ({clean_union['accuracy']:.1%})",
            f"- 품질 이슈 sample_id: {', '.join(payload['quality_issue_sample_ids']) or '없음'}",
            "",
            "## 해석",
            "",
            "- 단일 모델 최고 성능과 여러 모델이 하나라도 맞힌 oracle union을 분리해서 본다.",
            "- oracle union은 자동 선택기가 없는 상한선이므로 실제 서비스 성능으로 간주하지 않는다.",
            "- 품질 이슈가 있는 라벨은 90% 목표 판단에서 별도 관리해야 한다.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
