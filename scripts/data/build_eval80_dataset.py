from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from mini_vlm.evaluation.benchmark import read_jsonl, write_json, write_jsonl


SHORT_LVIS_TASKS = {"visual-question-answering", "counting", "spatial-reasoning"}


def main() -> None:
    parser = argparse.ArgumentParser(description="External VLM test split에서 고정 Eval80 holdout 생성")
    parser.add_argument("--source", default="data/external_vlm_10k/test.jsonl")
    parser.add_argument("--output", default="data/eval80/test.jsonl")
    parser.add_argument("--manifest", default="data/eval80/manifest.json")
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--lvis-count", type=int, default=40)
    parser.add_argument("--mmbench-count", type=int, default=40)
    args = parser.parse_args()

    rows = read_jsonl(args.source)
    selected = build_eval80(rows, seed=args.seed, lvis_count=args.lvis_count, mmbench_count=args.mmbench_count)
    write_jsonl(args.output, selected)
    write_json(
        args.manifest,
        {
            "name": "eval80-v1",
            "source": args.source,
            "output": args.output,
            "seed": args.seed,
            "sample_count": len(selected),
            "target_accuracy": 0.9,
            "target_correct_count": 72,
            "selection_policy": [
                "기존 external_vlm_10k/test.jsonl holdout에서만 선택한다.",
                "긴 detailed-caption은 제외하고 short VQA, counting, spatial reasoning, MMBench 객관식 중심으로 구성한다.",
                "가능하면 같은 image를 한 번만 선택하되, LVIS short-answer 후보의 unique image가 부족하면 sample 중복 없이 같은 image의 다른 질문을 허용한다.",
                "선택된 80개 sample/image는 이후 학습 데이터에 넣지 않는다.",
            ],
            "source_counts": Counter(str(row.get("metadata", {}).get("source") or "unknown") for row in selected),
            "task_counts": Counter(str(row.get("task") or "unknown") for row in selected),
            "image_count": len({row.get("image") for row in selected}),
        },
    )
    print(f"Eval80 생성: {args.output}")
    print(f"manifest: {args.manifest}")
    print(f"samples: {len(selected)}")


def build_eval80(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    lvis_count: int,
    mmbench_count: int,
) -> list[dict[str, Any]]:
    lvis_candidates = [
        row
        for row in rows
        if row.get("metadata", {}).get("source") == "lvis-instruct4v"
        and row.get("task") in SHORT_LVIS_TASKS
        and is_short_answer(str(row.get("answer") or ""))
    ]
    mmbench_candidates = [
        row
        for row in rows
        if row.get("metadata", {}).get("source") == "mmbench-dev-en"
        and row.get("task") == "multiple-choice-vqa"
    ]
    selected = pick_diverse_samples(lvis_candidates, count=lvis_count, seed=seed, salt="lvis")
    selected.extend(pick_unique_images(mmbench_candidates, count=mmbench_count, seed=seed, salt="mmbench"))
    if len(selected) != lvis_count + mmbench_count:
        raise RuntimeError(f"Eval80 샘플 수가 부족합니다: selected={len(selected)}")
    return sorted(selected, key=lambda row: stable_rank(row, seed=seed, salt="final"))


def is_short_answer(answer: str) -> bool:
    tokens = answer.split()
    return 1 <= len(tokens) <= 24


def pick_unique_images(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    salt: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for row in sorted(rows, key=lambda item: stable_rank(item, seed=seed, salt=salt)):
        image = str(row.get("image") or "")
        if image in seen_images:
            continue
        selected.append(row)
        seen_images.add(image)
        if len(selected) == count:
            return selected
    return selected


def pick_diverse_samples(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    salt: str,
) -> list[dict[str, Any]]:
    selected = pick_unique_images(rows, count=count, seed=seed, salt=salt)
    if len(selected) == count:
        return selected

    selected_ids = {str(row.get("sample_id")) for row in selected}
    for row in sorted(rows, key=lambda item: stable_rank(item, seed=seed, salt=f"{salt}-fallback")):
        sample_id = str(row.get("sample_id"))
        if sample_id in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(sample_id)
        if len(selected) == count:
            return selected
    return selected


def stable_rank(row: dict[str, Any], *, seed: int, salt: str) -> str:
    key = f"{seed}:{salt}:{row.get('sample_id')}:{row.get('image')}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
