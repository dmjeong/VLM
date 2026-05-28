from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM JSONL을 image 기준으로 train/validation/test로 분리")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl_many([Path(path) for path in args.input])
    split = split_rows_by_image(
        rows=rows,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "all.jsonl", rows)
    write_jsonl(output_root / "train.jsonl", split["train"])
    write_jsonl(output_root / "validation.jsonl", split["validation"])
    write_jsonl(output_root / "test.jsonl", split["test"])
    write_manifest(output_root / "split_manifest.json", split=split, seed=args.seed)
    print(
        "완료: "
        f"all={len(rows)} train={len(split['train'])} "
        f"validation={len(split['validation'])} test={len(split['test'])}",
        flush=True,
    )


def read_jsonl_many(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sample_id = str(row.get("sample_id") or "")
                if not sample_id:
                    raise ValueError(f"{path}:{line_number} sample_id가 없습니다.")
                if sample_id in seen_sample_ids:
                    continue
                seen_sample_ids.add(sample_id)
                rows.append(row)
    return rows


def split_rows_by_image(
    *,
    rows: list[dict[str, Any]],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("split ratio 합은 0보다 커야 합니다.")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        image = str(row.get("image") or "")
        if not image:
            raise ValueError(f"image가 없는 row가 있습니다: {row.get('sample_id')}")
        groups[image].append(row)

    image_keys = sorted(groups)
    random.Random(seed).shuffle(image_keys)
    image_count = len(image_keys)
    train_count = round(image_count * train_ratio / ratio_sum)
    validation_count = round(image_count * validation_ratio / ratio_sum)
    train_count = min(max(1, train_count), image_count)
    validation_count = min(max(1, validation_count), max(0, image_count - train_count))
    test_count = image_count - train_count - validation_count
    if test_count <= 0 and image_count >= 3:
        test_count = 1
        if train_count >= validation_count and train_count > 1:
            train_count -= 1
        elif validation_count > 1:
            validation_count -= 1

    train_images = set(image_keys[:train_count])
    validation_images = set(image_keys[train_count : train_count + validation_count])
    test_images = set(image_keys[train_count + validation_count :])

    return {
        "train": flatten_groups(groups, train_images),
        "validation": flatten_groups(groups, validation_images),
        "test": flatten_groups(groups, test_images),
    }


def flatten_groups(groups: dict[str, list[dict[str, Any]]], image_keys: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image in sorted(image_keys):
        rows.extend(sorted(groups[image], key=lambda row: str(row.get("sample_id"))))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_manifest(path: Path, *, split: dict[str, list[dict[str, Any]]], seed: int) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "split_unit": "image",
        "counts": {
            name: {
                "samples": len(rows),
                "images": len({str(row["image"]) for row in rows}),
            }
            for name, rows in split.items()
        },
        "images": {
            name: sorted({str(row["image"]) for row in rows})
            for name, rows in split.items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
