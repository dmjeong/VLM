from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_vlm.evaluation.benchmark import (
    index_samples_by_id,
    read_jsonl,
    summarize_predictions,
    summary_to_dict,
    write_json,
    write_jsonl,
)
from mini_vlm.evaluation.dashboard import write_evaluation_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="생성 prediction JSONL을 Eval80 기준으로 채점")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", default="data/eval80/test.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-accuracy", type=float, default=0.9)
    parser.add_argument("--summary-json", help="기존 test_summary.json을 benchmark 결과로 갱신")
    parser.add_argument("--image-root", help="대시보드 이미지 root")
    parser.add_argument("--dashboard", help="갱신할 dashboard.html 경로")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    samples = read_jsonl(args.dataset)
    predictions = read_jsonl(args.predictions)
    summary, scored_rows = summarize_predictions(
        predictions=predictions,
        samples_by_id=index_samples_by_id(samples),
        target_accuracy=args.target_accuracy,
    )
    write_json(output_dir / "benchmark_summary.json", summary_to_dict(summary))
    write_jsonl(output_dir / "benchmark_scored_predictions.jsonl", scored_rows)
    if args.summary_json:
        summary_path = Path(args.summary_json)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload.update(
            {
                "benchmark_accuracy": summary.accuracy,
                "benchmark_correct_count": summary.correct_count,
                "benchmark_target_reached": summary.target_reached,
            }
        )
        write_json(summary_path, payload)
    if args.dashboard:
        if not args.summary_json or not args.image_root:
            raise SystemExit("--dashboard 사용 시 --summary-json과 --image-root가 필요합니다.")
        write_evaluation_dashboard(
            summary_path=args.summary_json,
            predictions_path=args.predictions,
            dataset_path=args.dataset,
            image_root=args.image_root,
            output_path=args.dashboard,
        )
    print(
        "benchmark: "
        f"{summary.correct_count}/{summary.sample_count} "
        f"({summary.accuracy * 100:.1f}%), "
        f"target={summary.target_correct_count}/{summary.sample_count}, "
        f"reached={summary.target_reached}"
    )
    print(f"summary: {output_dir / 'benchmark_summary.json'}")
    print(f"scored: {output_dir / 'benchmark_scored_predictions.jsonl'}")
    if not summary.target_reached:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
