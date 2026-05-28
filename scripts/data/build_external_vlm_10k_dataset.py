from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import random
import re
import shutil
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SHAREGPT4V_REPO = "Lin-Chen/ShareGPT4V"
SHAREGPT4V_FILE = "sharegpt4v_instruct_gpt4-vision_cap100k.json"
LVIS_REPO = "X2FD/LVIS-Instruct4V"
LVIS_FILE = "lvis_instruct4v_220k.json"
MMBENCH_DEV_EN_URL = "https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv"
COCO_BASE_URLS = (
    "http://images.cocodataset.org",
    "https://images.cocodataset.org",
)
USER_AGENT = "dinov3-mini-vlm-external-dataset-builder/0.1"


@dataclass
class BuildState:
    output_root: Path
    rng: random.Random
    used_images: set[str] = field(default_factory=set)
    source_counts: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)

    def count_source(self, source: str, amount: int) -> None:
        self.source_counts[source] = self.source_counts.get(source, 0) + amount

    def count_skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="ShareGPT4V/LVIS-Instruct4V/MMBench 기반 10K VLM 데이터셋 생성")
    parser.add_argument("--output-root", default="data/external_vlm_10k")
    parser.add_argument("--target-train", type=int, default=8000)
    parser.add_argument("--target-validation", type=int, default=1000)
    parser.add_argument("--target-test", type=int, default=1000)
    parser.add_argument("--train-lvis", type=int, default=6500)
    parser.add_argument("--train-sharegpt4v", type=int, default=1500)
    parser.add_argument("--validation-lvis", type=int, default=500)
    parser.add_argument("--validation-mmbench", type=int, default=500)
    parser.add_argument("--test-lvis", type=int, default=500)
    parser.add_argument("--test-mmbench", type=int, default=500)
    parser.add_argument("--caption-prompt-variants", type=int, default=3)
    parser.add_argument("--qa-prompt-variants", type=int, default=3)
    parser.add_argument("--lvis-record-limit", type=int, default=4000)
    parser.add_argument("--sharegpt4v-record-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-sleep", type=float, default=0.0)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.reset and output_root.exists() and not args.dry_run:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    state = BuildState(output_root=output_root, rng=random.Random(args.seed))

    print("외부 데이터셋 준비 시작", flush=True)
    print(f"목표: train={args.target_train} validation={args.target_validation} test={args.target_test}", flush=True)
    if args.dry_run:
        print("dry-run: 파일 다운로드와 이미지 저장은 실행하지 않습니다.", flush=True)
        return

    lvis_records = load_hf_json_records(repo_id=LVIS_REPO, filename=LVIS_FILE, max_records=args.lvis_record_limit)
    share_records = load_hf_json_records(
        repo_id=SHAREGPT4V_REPO,
        filename=SHAREGPT4V_FILE,
        max_records=args.sharegpt4v_record_limit,
    )
    state.rng.shuffle(lvis_records)
    state.rng.shuffle(share_records)

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    fill_from_conversation_records(
        target_rows=train_rows,
        records=lvis_records,
        quota=args.train_lvis,
        split="train",
        source="lvis-instruct4v",
        state=state,
        caption_prompt_variants=args.caption_prompt_variants,
        qa_prompt_variants=args.qa_prompt_variants,
        request_sleep=args.request_sleep,
    )
    fill_from_conversation_records(
        target_rows=train_rows,
        records=share_records,
        quota=args.train_sharegpt4v,
        split="train",
        source="sharegpt4v",
        state=state,
        caption_prompt_variants=args.caption_prompt_variants,
        qa_prompt_variants=args.qa_prompt_variants,
        request_sleep=args.request_sleep,
    )
    fill_from_conversation_records(
        target_rows=validation_rows,
        records=lvis_records,
        quota=args.validation_lvis,
        split="validation",
        source="lvis-instruct4v",
        state=state,
        caption_prompt_variants=args.caption_prompt_variants,
        qa_prompt_variants=args.qa_prompt_variants,
        request_sleep=args.request_sleep,
    )
    fill_from_conversation_records(
        target_rows=test_rows,
        records=lvis_records,
        quota=args.test_lvis,
        split="test",
        source="lvis-instruct4v",
        state=state,
        caption_prompt_variants=args.caption_prompt_variants,
        qa_prompt_variants=args.qa_prompt_variants,
        request_sleep=args.request_sleep,
    )

    mmbench_rows = load_mmbench_rows(MMBENCH_DEV_EN_URL)
    state.rng.shuffle(mmbench_rows)
    fill_from_mmbench(
        target_rows=validation_rows,
        rows=mmbench_rows,
        quota=args.validation_mmbench,
        split="validation",
        state=state,
    )
    fill_from_mmbench(
        target_rows=test_rows,
        rows=mmbench_rows,
        quota=args.test_mmbench,
        split="test",
        state=state,
    )

    train_rows = enforce_count(train_rows, args.target_train, "train")
    validation_rows = enforce_count(validation_rows, args.target_validation, "validation")
    test_rows = enforce_count(test_rows, args.target_test, "test")
    all_rows = train_rows + validation_rows + test_rows

    write_jsonl(output_root / "train.jsonl", train_rows)
    write_jsonl(output_root / "validation.jsonl", validation_rows)
    write_jsonl(output_root / "test.jsonl", test_rows)
    write_jsonl(output_root / "all.jsonl", all_rows)
    write_manifest(
        output_root / "split_manifest.json",
        rows_by_split={"train": train_rows, "validation": validation_rows, "test": test_rows},
        state=state,
        seed=args.seed,
    )
    write_readme(output_root)
    print(
        "완료: "
        f"train={len(train_rows)} validation={len(validation_rows)} test={len(test_rows)} "
        f"images={len({str(row['image']) for row in all_rows})}",
        flush=True,
    )


def load_hf_json_records(*, repo_id: str, filename: str, max_records: int) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub가 필요합니다. 현재 .venv에는 설치되어 있어야 합니다.") from exc
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    records: list[dict[str, Any]] = []
    for row in iter_json_array_objects(Path(path)):
        if isinstance(row, dict):
            records.append(row)
        if len(records) >= max_records:
            break
    print(f"annotation loaded: {repo_id}/{filename} records={len(records)}", flush=True)
    return records


def iter_json_array_objects(path: Path):
    """큰 JSON array에서 객체를 하나씩 읽는다.

    의도: ShareGPT4V/LVIS annotation은 수백 MB라 `json.load`로 전체를 올리면 로컬 노트북에서 오래 멈춘다.
    참고: 두 annotation 파일은 JSONL이 아니라 최상위 JSON array 구조다.
    선택 이유: 추가 의존성 없이 필요한 앞부분 record만 읽으면 10K bootstrap 생성에는 충분하다.
    """

    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    with path.open("r", encoding="utf-8") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk and not buffer.strip():
                return
            buffer += chunk
            position = 0
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"JSON array 시작 문자를 찾지 못했습니다: {path}")
                    started = True
                    position += 1
                while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                if position >= len(buffer):
                    break
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                yield value
                position = end
            buffer = buffer[position:]
            if not chunk:
                return


def fill_from_conversation_records(
    *,
    target_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    quota: int,
    split: str,
    source: str,
    state: BuildState,
    caption_prompt_variants: int,
    qa_prompt_variants: int,
    request_sleep: float,
) -> None:
    target_total = len(target_rows) + quota
    next_log = len(target_rows) + 500
    for record in records:
        if len(target_rows) >= target_total:
            return
        image_ref = str(record.get("image") or "")
        if not image_ref or image_ref in state.used_images:
            state.count_skip(f"{source}:duplicate-or-missing-image")
            continue
        local_image = ensure_coco_image(image_ref=image_ref, output_root=state.output_root)
        if local_image is None:
            state.count_skip(f"{source}:image-download-failed")
            continue
        pairs = conversation_pairs(record.get("conversations"))
        if not pairs:
            state.count_skip(f"{source}:empty-conversation")
            continue
        state.used_images.add(image_ref)
        samples = samples_from_pairs(
            pairs=pairs,
            image=local_image,
            source=source,
            source_id=str(record.get("id") or image_ref),
            split=split,
            caption_prompt_variants=caption_prompt_variants,
            qa_prompt_variants=qa_prompt_variants,
            remaining=target_total - len(target_rows),
        )
        target_rows.extend(samples)
        state.count_source(source, len(samples))
        if request_sleep > 0:
            time.sleep(request_sleep)
        if len(target_rows) >= next_log:
            print(f"  {split}/{source}: {len(target_rows)}/{target_total}", flush=True)
            next_log += 500


def ensure_coco_image(*, image_ref: str, output_root: Path) -> str | None:
    match = re.match(r"^coco/(train2017|val2017|test2017)/([^/]+\.jpg)$", image_ref)
    if not match:
        return None
    split_name, filename = match.groups()
    relative_path = Path("images") / "coco" / split_name / filename
    target_path = output_root / relative_path
    if target_path.exists():
        return relative_path.as_posix()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = fetch_coco_image_bytes(split_name=split_name, filename=filename)
    if image_bytes is None:
        return None
    if not looks_like_image(image_bytes):
        return None
    target_path.write_bytes(image_bytes)
    return relative_path.as_posix()


def conversation_pairs(conversations: Any) -> list[tuple[str, str]]:
    if not isinstance(conversations, list):
        return []
    pairs: list[tuple[str, str]] = []
    pending_question: str | None = None
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from") or "").lower()
        value = clean_text(str(turn.get("value") or ""))
        if not value:
            continue
        if role in {"human", "user"}:
            pending_question = clean_question(value)
        elif role in {"gpt", "assistant"} and pending_question:
            pairs.append((pending_question, value))
            pending_question = None
    return pairs


def samples_from_pairs(
    *,
    pairs: list[tuple[str, str]],
    image: str,
    source: str,
    source_id: str,
    split: str,
    caption_prompt_variants: int,
    qa_prompt_variants: int,
    remaining: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_index, (question, answer) in enumerate(pairs):
        questions = qa_questions(question, qa_prompt_variants)
        if len(pairs) == 1 and len(answer) >= 120 and caption_prompt_variants > 1:
            questions = caption_questions(question, caption_prompt_variants)
        for variant_index, variant_question in enumerate(questions):
            if len(rows) >= remaining:
                return rows
            rows.append(
                {
                    "sample_id": stable_sample_id(source, split, source_id, pair_index, variant_index),
                    "image": image,
                    "question": variant_question,
                    "answer": answer,
                    "task": infer_task(variant_question, answer),
                    "metadata": {
                        "source": source,
                        "source_id": source_id,
                        "source_split": split,
                        "conversation_index": pair_index,
                        "variant_index": variant_index,
                    },
                }
            )
    return rows


def caption_questions(original_question: str, count: int) -> list[str]:
    variants = [
        original_question,
        "Describe this image in detail.",
        "Explain the visual content of this image.",
        "What is happening in this image?",
        "Give a detailed caption for the image.",
    ]
    deduped: list[str] = []
    for question in variants:
        if question not in deduped:
            deduped.append(question)
    return deduped[: max(1, count)]


def qa_questions(original_question: str, count: int) -> list[str]:
    variants = [original_question]
    lowered = original_question.lower()
    if not lowered.startswith("based on the image"):
        variants.append(f"Based on the image, {lowercase_first(original_question)}")
    if not lowered.startswith("look at the image"):
        variants.append(f"Look at the image and answer: {original_question}")
    deduped: list[str] = []
    for question in variants:
        cleaned = clean_text(question)
        if cleaned not in deduped:
            deduped.append(cleaned)
    return deduped[: max(1, count)]


def lowercase_first(value: str) -> str:
    if not value:
        return value
    return value[0].lower() + value[1:]


def load_mmbench_rows(url: str) -> list[dict[str, str]]:
    raw_path = Path("data/external_manifests/MMBench_DEV_EN.tsv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        raw_path.write_bytes(fetch_bytes(url, insecure_ssl=True))
    rows: list[dict[str, str]] = []
    with raw_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            rows.append({key: value for key, value in row.items() if key is not None})
    return rows


def fill_from_mmbench(
    *,
    target_rows: list[dict[str, Any]],
    rows: list[dict[str, str]],
    quota: int,
    split: str,
    state: BuildState,
) -> None:
    source = "mmbench-dev-en"
    target_total = len(target_rows) + quota
    next_log = len(target_rows) + 250
    for row in rows:
        if len(target_rows) >= target_total:
            return
        sample = sample_from_mmbench_row(row=row, split=split, output_root=state.output_root)
        image_key = f"mmbench/{row.get('index')}"
        if sample is None or image_key in state.used_images:
            state.count_skip("mmbench:invalid-or-duplicate")
            continue
        state.used_images.add(image_key)
        target_rows.append(sample)
        state.count_source(source, 1)
        if len(target_rows) >= next_log:
            print(f"  {split}/{source}: {len(target_rows)}/{target_total}", flush=True)
            next_log += 250


def sample_from_mmbench_row(*, row: dict[str, str], split: str, output_root: Path) -> dict[str, Any] | None:
    index = str(row.get("index") or "").strip()
    image_base64 = str(row.get("image") or "").strip()
    answer_key = str(row.get("answer") or "").strip()
    question = str(row.get("question") or "").strip()
    if not index or not image_base64 or answer_key not in {"A", "B", "C", "D"} or not question:
        return None
    image_path = output_root / "images" / "mmbench" / f"{index}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        try:
            image_bytes = base64.b64decode(image_base64, validate=False)
        except ValueError:
            return None
        if not looks_like_image(image_bytes):
            return None
        image_path.write_bytes(image_bytes)

    prompt = build_mmbench_prompt(row)
    answer_text = option_answer(row, answer_key)
    return {
        "sample_id": f"mmbench-{split}-{index}",
        "image": str(image_path.relative_to(output_root)),
        "question": prompt,
        "answer": answer_text,
        "task": "multiple-choice-vqa",
        "metadata": {
            "source": "mmbench-dev-en",
            "source_id": index,
            "source_split": split,
            "category": row.get("category", ""),
            "l2_category": row.get("l2-category", ""),
            "answer_key": answer_key,
        },
    }


def build_mmbench_prompt(row: dict[str, str]) -> str:
    parts = []
    hint = str(row.get("hint") or "").strip()
    if hint:
        parts.append(hint)
    parts.append(str(row.get("question") or "").strip())
    option_lines = []
    for key in ("A", "B", "C", "D"):
        value = str(row.get(key) or "").strip()
        if value:
            option_lines.append(f"{key}. {value}")
    if option_lines:
        parts.append("Options:\n" + "\n".join(option_lines))
    parts.append("Answer with the best option letter and text.")
    return "\n\n".join(parts)


def option_answer(row: dict[str, str], answer_key: str) -> str:
    option_text = str(row.get(answer_key) or "").strip()
    return f"{answer_key}. {option_text}" if option_text else answer_key


def clean_question(value: str) -> str:
    cleaned = value.replace("<image>", " ")
    cleaned = clean_text(cleaned)
    return cleaned or "Describe this image."


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def infer_task(question: str, answer: str) -> str:
    q = question.lower()
    if "how many" in q or "number" in q:
        return "counting"
    if "where" in q or "position" in q or "relation" in q:
        return "spatial-reasoning"
    if "why" in q or "explain" in q or len(answer) > 220:
        return "detailed-caption"
    return "visual-question-answering"


def stable_sample_id(source: str, split: str, source_id: str, pair_index: int, variant_index: int) -> str:
    digest = hashlib.sha1(f"{source}:{split}:{source_id}:{pair_index}:{variant_index}".encode("utf-8")).hexdigest()
    return f"{slugify(source)}-{split}-{digest[:14]}"


def enforce_count(rows: list[dict[str, Any]], target: int, split: str) -> list[dict[str, Any]]:
    if len(rows) < target:
        raise RuntimeError(f"{split} 목표 수량을 채우지 못했습니다: {len(rows)}/{target}")
    return rows[:target]


def fetch_bytes(url: str, *, insecure_ssl: bool = False) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    context = ssl._create_unverified_context() if insecure_ssl else None
    with urlopen(request, timeout=30, context=context) as response:
        return response.read()


def fetch_coco_image_bytes(*, split_name: str, filename: str) -> bytes | None:
    for base_url in COCO_BASE_URLS:
        url = f"{base_url}/{split_name}/{filename}"
        try:
            return fetch_bytes(url)
        except (HTTPError, URLError, TimeoutError):
            continue
    return None


def looks_like_image(image_bytes: bytes) -> bool:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return True
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    return False


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_manifest(
    path: Path,
    *,
    rows_by_split: dict[str, list[dict[str, Any]]],
    state: BuildState,
    seed: int,
) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "split_unit": "image",
        "sources": state.source_counts,
        "skipped": state.skipped,
        "counts": {
            split: {
                "samples": len(rows),
                "images": len({str(row["image"]) for row in rows}),
                "sources": source_counts_for(rows),
            }
            for split, rows in rows_by_split.items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_counts_for(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        source = str(metadata.get("source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def write_readme(output_root: Path) -> None:
    text = """# External VLM 10K Dataset

이 데이터셋은 DINOv3 Mini VLM 실험을 위해 ShareGPT4V, LVIS-Instruct4V, MMBench DEV EN에서 일부 샘플을 변환한 bootstrap 데이터셋이다.

## 구성

- `train.jsonl`: ShareGPT4V + LVIS-Instruct4V 학습 샘플
- `validation.jsonl`: LVIS-Instruct4V + MMBench DEV EN 검증 샘플
- `test.jsonl`: LVIS-Instruct4V + MMBench DEV EN 테스트 샘플
- `images/`: COCO 원본 이미지와 MMBench base64 이미지에서 추출한 로컬 이미지
- `split_manifest.json`: split별 sample/image/source 수량

## 주의

- MMBench는 원래 평가용 벤치마크이므로 train split에는 넣지 않는다.
- MME는 별도 평가 벤치마크로 유지하고, 공식 다운로드/라이선스 확인 후 별도 importer로 추가한다.
- ShareGPT4V/LVIS 계열 데이터는 원본 라이선스와 사용 조건을 반드시 확인하고 배포 범위를 제한한다.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def slugify(value: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z가-힣]+", "-", value.strip()).strip("-").lower()
    return normalized or "item"


if __name__ == "__main__":
    main()
