from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "dinov3-mini-vlm-dataset-builder/0.1 (local research; Wikimedia Commons API)"
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
ALLOWED_LICENSE_HINTS = (
    "public domain",
    "cc0",
    "cc by",
    "cc-by",
    "cc by-sa",
    "cc-by-sa",
)
EXCLUDED_RELEVANCE_PHRASES = (
    "world cup",
    "skeleton world cup",
    "football cup",
    "rugby cup",
    "rifle",
    "crooks",
    "smuggling",
    "baby sick",
    "car accident",
    "train wreck",
    "jefferson airplane",
    "dehydration",
    "shoe plaza",
    "pyracantha",
)


@dataclass(frozen=True)
class CrawlClass:
    label: str
    query: str
    category: str


@dataclass(frozen=True)
class ImageRecord:
    label: str
    category: str
    title: str
    image_path: str
    source_url: str
    description_url: str
    license_name: str
    license_url: str
    artist: str
    width: int
    height: int


DEFAULT_CLASSES = [
    CrawlClass("apple", "apple fruit photo filetype:bitmap", "fruit"),
    CrawlClass("banana", "banana fruit photo filetype:bitmap", "fruit"),
    CrawlClass("orange", "orange fruit photo filetype:bitmap", "fruit"),
    CrawlClass("strawberry", "strawberry fruit photo filetype:bitmap", "fruit"),
    CrawlClass("coffee mug", "coffee mug photo filetype:bitmap", "household"),
    CrawlClass("cup", "coffee cup drinking cup photo filetype:bitmap", "household"),
    CrawlClass("bottle", "bottle object photo filetype:bitmap", "household"),
    CrawlClass("pencil", "pencil object photo filetype:bitmap", "stationery"),
    CrawlClass("book", "book object photo filetype:bitmap", "stationery"),
    CrawlClass("laptop", "laptop computer photo filetype:bitmap", "electronics"),
    CrawlClass("keyboard", "computer keyboard photo filetype:bitmap", "electronics"),
    CrawlClass("computer mouse", "computer mouse photo filetype:bitmap", "electronics"),
    CrawlClass("backpack", "backpack photo filetype:bitmap", "object"),
    CrawlClass("chair", "chair furniture photo filetype:bitmap", "furniture"),
    CrawlClass("table", "table furniture photo filetype:bitmap", "furniture"),
    CrawlClass("bicycle", "bicycle photo filetype:bitmap", "vehicle"),
    CrawlClass("car", "car vehicle photo filetype:bitmap", "vehicle"),
    CrawlClass("bus", "bus vehicle photo filetype:bitmap", "vehicle"),
    CrawlClass("train", "train vehicle photo filetype:bitmap", "vehicle"),
    CrawlClass("airplane", "airplane photo filetype:bitmap", "vehicle"),
    CrawlClass("boat", "boat photo filetype:bitmap", "vehicle"),
    CrawlClass("cat", "cat animal photo filetype:bitmap", "animal"),
    CrawlClass("dog", "dog animal photo filetype:bitmap", "animal"),
    CrawlClass("horse", "horse animal photo filetype:bitmap", "animal"),
    CrawlClass("bird", "bird animal photo filetype:bitmap", "animal"),
    CrawlClass("flower", "flower photo filetype:bitmap", "plant"),
    CrawlClass("tree", "tree photo filetype:bitmap", "plant"),
    CrawlClass("leaf", "leaf plant photo filetype:bitmap", "plant"),
    CrawlClass("clock", "clock object photo filetype:bitmap", "object"),
    CrawlClass("shoe", "shoe object photo filetype:bitmap", "object"),
    CrawlClass("hat", "hat object photo filetype:bitmap", "object"),
    CrawlClass("umbrella", "umbrella object photo filetype:bitmap", "object"),
    CrawlClass("guitar", "guitar musical instrument photo filetype:bitmap", "instrument"),
    CrawlClass("camera", "camera object photo filetype:bitmap", "electronics"),
    CrawlClass("mobile phone", "mobile phone photo filetype:bitmap", "electronics"),
    CrawlClass("pizza", "pizza food photo filetype:bitmap", "food"),
    CrawlClass("bread", "bread food photo filetype:bitmap", "food"),
    CrawlClass("cake", "cake food photo filetype:bitmap", "food"),
    CrawlClass("fish", "fish animal photo filetype:bitmap", "animal"),
    CrawlClass("bench", "bench furniture photo filetype:bitmap", "furniture"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Wikimedia Commons에서 mini VLM용 이미지 QA 데이터셋 생성")
    parser.add_argument("--output-root", default="data/wikimedia_commons_1k")
    parser.add_argument("--target-samples", type=int, default=1000)
    parser.add_argument("--samples-per-image", type=int, default=20)
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-sleep", type=float, default=0.05)
    parser.add_argument("--max-api-pages-per-class", type=int, default=6)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.reset and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    image_target = max(1, (args.target_samples + args.samples_per_image - 1) // args.samples_per_image)
    images_per_class = max(1, (image_target + len(DEFAULT_CLASSES) - 1) // len(DEFAULT_CLASSES))

    records: list[ImageRecord] = []
    seen_titles: set[str] = set()
    seen_hashes: set[str] = set()
    for crawl_class in DEFAULT_CLASSES:
        class_records = collect_class_images(
            crawl_class=crawl_class,
            output_root=output_root,
            target_images=images_per_class,
            image_width=args.image_width,
            max_api_pages=args.max_api_pages_per_class,
            request_sleep=args.request_sleep,
            seen_titles=seen_titles,
            seen_hashes=seen_hashes,
        )
        records.extend(class_records)
        print(f"{crawl_class.label}: images={len(class_records)} total_images={len(records)}", flush=True)
        if len(records) >= image_target:
            break

    if not records:
        raise RuntimeError("수집된 이미지가 없습니다. 네트워크/API/검색어를 확인하세요.")

    samples = build_samples(records=records, samples_per_image=args.samples_per_image, target_samples=args.target_samples)
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    validation_count = max(1, int(len(samples) * args.validation_ratio))
    validation_samples = samples[:validation_count]
    train_samples = samples[validation_count:]

    write_jsonl(output_root / "train.jsonl", train_samples)
    write_jsonl(output_root / "validation.jsonl", validation_samples)
    write_sources(output_root=output_root, records=records, sample_count=len(samples))
    print(
        f"완료: images={len(records)} train={len(train_samples)} validation={len(validation_samples)} "
        f"root={output_root}",
        flush=True,
    )


def collect_class_images(
    *,
    crawl_class: CrawlClass,
    output_root: Path,
    target_images: int,
    image_width: int,
    max_api_pages: int,
    request_sleep: float,
    seen_titles: set[str],
    seen_hashes: set[str],
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    continue_token: dict[str, str] = {}
    for _ in range(max_api_pages):
        payload = commons_search(crawl_class.query, image_width=image_width, continue_token=continue_token)
        pages = list(payload.get("query", {}).get("pages", {}).values())
        pages.sort(key=lambda page: int(page.get("index", 0)))
        for page in pages:
            record = maybe_download_image(
                page=page,
                crawl_class=crawl_class,
                output_root=output_root,
                seen_titles=seen_titles,
                seen_hashes=seen_hashes,
            )
            if record is not None:
                records.append(record)
                print(f"  + {crawl_class.label}: {record.title}", flush=True)
            if len(records) >= target_images:
                return records
        continue_payload = payload.get("continue")
        if not isinstance(continue_payload, dict) or "gsroffset" not in continue_payload:
            return records
        continue_token = {"gsroffset": str(continue_payload["gsroffset"])}
        time.sleep(request_sleep)
    return records


def commons_search(query: str, *, image_width: int, continue_token: dict[str, str]) -> dict[str, Any]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrsearch": query,
        "gsrlimit": "50",
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": str(image_width),
    }
    params.update(continue_token)
    return fetch_json(COMMONS_API_URL + "?" + urlencode(params))


def maybe_download_image(
    *,
    page: dict[str, Any],
    crawl_class: CrawlClass,
    output_root: Path,
    seen_titles: set[str],
    seen_hashes: set[str],
) -> ImageRecord | None:
    title = str(page.get("title") or "")
    imageinfo = page.get("imageinfo") or []
    if not title or title in seen_titles or not imageinfo:
        return None
    info = imageinfo[0]
    mime = str(info.get("mime") or "").lower()
    width = int(info.get("width") or 0)
    height = int(info.get("height") or 0)
    if mime not in ALLOWED_MIME_TYPES or width < 256 or height < 256:
        return None
    metadata = info.get("extmetadata") if isinstance(info.get("extmetadata"), dict) else {}
    if not looks_relevant_to_label(crawl_class.label, title=title, metadata=metadata):
        return None
    license_name = metadata_value(metadata, "LicenseShortName") or metadata_value(metadata, "UsageTerms")
    if not is_allowed_license(license_name):
        return None
    source_url = str(info.get("thumburl") or info.get("url") or "")
    description_url = str(info.get("descriptionurl") or "")
    if not source_url or not description_url:
        return None

    try:
        image_bytes = fetch_bytes(source_url)
    except (HTTPError, URLError, TimeoutError):
        return None
    digest = hashlib.sha256(image_bytes).hexdigest()
    if digest in seen_hashes:
        return None

    class_dir = output_root / "images" / slugify(crawl_class.label)
    class_dir.mkdir(parents=True, exist_ok=True)
    extension = extension_for(mime=mime, url=source_url)
    filename = f"{slugify(title.removeprefix('File:'))}-{digest[:10]}{extension}"
    image_path = class_dir / filename
    image_path.write_bytes(image_bytes)

    seen_titles.add(title)
    seen_hashes.add(digest)
    return ImageRecord(
        label=crawl_class.label,
        category=crawl_class.category,
        title=title,
        image_path=str(image_path.relative_to(output_root)),
        source_url=source_url,
        description_url=description_url,
        license_name=clean_text(license_name),
        license_url=clean_text(metadata_value(metadata, "LicenseUrl")),
        artist=clean_text(metadata_value(metadata, "Artist")),
        width=width,
        height=height,
    )


def build_samples(*, records: list[ImageRecord], samples_per_image: int, target_samples: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        label = record.label
        article_label = article_phrase(label)
        capital_label = label[:1].upper() + label[1:]
        templates = [
            ("caption", "Describe this image.", f"A photo of {article_label} is shown."),
            ("object", "What object is shown?", f"{article_label.capitalize()} is shown."),
            ("main-subject", "What is the main subject?", f"The main subject is {article_label}."),
            ("presence", f"Is there {article_label} in the image?", f"Yes, {article_label} is visible."),
            ("short-name", "Name the visible object.", f"{capital_label}."),
            ("visible-item", "What can you see in the image?", f"I can see {article_label}."),
            ("identify", "Identify the visible item.", f"The visible item is {article_label}."),
            ("category", "What kind of thing is shown?", f"It is {article_label}."),
            ("simple-caption", "Give a short caption for the image.", f"{article_label.capitalize()} in a photo."),
            ("answer-briefly", "Answer briefly: what is in the image?", f"{capital_label}."),
            ("visible-object", "What visible object appears in this photo?", f"The photo shows {article_label}."),
            ("label", "Choose the best label for the image.", f"{capital_label}."),
            ("pictured", "What is pictured here?", f"{article_label.capitalize()} is pictured here."),
            ("contains", "What does the image contain?", f"The image contains {article_label}."),
            ("class-name", "What class does this image belong to?", f"The class is {label}."),
            ("main-item", "What is the main item in the image?", f"The main item is {article_label}."),
            ("recognition", "Recognize the object in the image.", f"It is {article_label}."),
            ("visual-question", "What would you call the visible thing?", f"I would call it {article_label}."),
            ("one-word", "Answer with the object name.", f"{label}."),
            ("photo-subject", "What is the subject of the photo?", f"The subject is {article_label}."),
        ]
        for template_index, (task, question, answer) in enumerate(templates[:samples_per_image], start=1):
            samples.append(
                {
                    "sample_id": f"commons-{index:05d}-{template_index:02d}-{slugify(label)}",
                    "image": record.image_path,
                    "question": question,
                    "answer": answer,
                    "task": task,
                    "metadata": {
                        "source": "wikimedia_commons",
                        "object": label,
                        "category": record.category,
                        "file_title": record.title,
                        "description_url": record.description_url,
                        "license": record.license_name,
                        "license_url": record.license_url,
                        "artist": record.artist,
                    },
                }
            )
            if len(samples) >= target_samples:
                return samples
    return samples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_sources(*, output_root: Path, records: list[ImageRecord], sample_count: int) -> None:
    sources_path = output_root / "sources.jsonl"
    with sources_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.__dict__, ensure_ascii=False, allow_nan=False) + "\n")
    readme = output_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Wikimedia Commons 1K Mini VLM Dataset",
                "",
                f"- 생성 시각: {datetime.now(timezone.utc).isoformat()}",
                f"- 이미지 수: {len(records)}",
                f"- 샘플 수: {sample_count}",
                "- 수집 방식: Wikimedia Commons MediaWiki API `query+imageinfo`",
                "- 필터: JPEG/PNG, 256px 이상, Public domain/CC0/CC BY/CC BY-SA 계열 라이선스 메타데이터",
                "- 라벨 방식: 검색 class label 기반 template caption/VQA",
                "",
                "주의: 이 데이터셋은 실험용 bootstrap 데이터다. 실제 VLM 품질 개선용으로는 사람이 검수한 caption/VQA가 더 필요하다.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=15) as response:
        return response.read()


def metadata_value(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key, {})
    if not isinstance(value, dict):
        return ""
    return clean_text(str(value.get("value") or ""))


def is_allowed_license(license_name: str) -> bool:
    normalized = clean_text(license_name).lower()
    return bool(normalized) and any(hint in normalized for hint in ALLOWED_LICENSE_HINTS)


def looks_relevant_to_label(label: str, *, title: str, metadata: dict[str, Any]) -> bool:
    """검색 결과가 class label과 실제로 관련 있는지 1차 필터링한다.

    의도: Commons 검색은 `cup`이 스포츠 World Cup을 반환하는 식의 오탐이 있다. 사람이 검수하지 않는
    bootstrap 데이터에서는 오탐이 곧 잘못된 라벨이 되므로, 제목/설명/카테고리 안에 label token이 실제로
    등장하는 결과만 받는다.
    """

    searchable = " ".join(
        [
            title,
            metadata_value(metadata, "ObjectName"),
            metadata_value(metadata, "ImageDescription"),
            metadata_value(metadata, "Categories"),
        ]
    ).lower()
    if any(phrase in searchable for phrase in EXCLUDED_RELEVANCE_PHRASES):
        return False
    label_tokens = [token for token in re.split(r"[^a-z0-9]+", label.lower()) if token]
    if not label_tokens:
        return False
    searchable_tokens = set(re.split(r"[^a-z0-9]+", searchable))
    return all(token in searchable_tokens for token in label_tokens)


def clean_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", html.unescape(value))
    return re.sub(r"\s+", " ", without_tags).strip()


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return normalized[:80] or "item"


def article_phrase(label: str) -> str:
    article = "an" if label[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {label}"


def extension_for(*, mime: str, url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    if mime == "image/png":
        return ".png"
    return ".jpg"


if __name__ == "__main__":
    main()
