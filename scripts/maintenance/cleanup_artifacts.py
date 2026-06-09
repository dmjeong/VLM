from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path("artifacts/dinov3-mini-vlm")
DEFAULT_ARCHIVE_DIR = ROOT / "_archive" / "legacy-20260528"

ACTIVE_DIRS = (
    ROOT / "vision-ablation",
    ROOT / "dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-stage0",
    ROOT / "dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1",
    ROOT / "dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch3",
    ROOT / "dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage2-lora-epoch1",
    ROOT / "clip-vitb16-qwen-external-10k-qformer-distilbert-itc-stage0",
    ROOT / "clip-vitb16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1",
    ROOT / "siglip-vitb16-qwen-external-10k-qformer-distilbert-itc-stage0",
    ROOT / "siglip-vitb16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1",
)

ARCHIVE_CANDIDATES = (
    ROOT / "dinov3-local-vits16-qwen-external-10k-adapter-stage1",
    ROOT / "dinov3-local-vits16-qwen-external-10k-adapter-stage1-epoch1",
    ROOT / "dinov3-local-vits16-qwen-lora-perceiver",
    ROOT / "dinov3-local-vits16-qwen-qformer-distilbert-itc-smoke",
    ROOT / "dinov3-local-vits16-qwen-qformer-distilbert-itc-to-llm-smoke",
    ROOT / "dinov3-local-vits16-qwen-qformer-itc-smoke",
    ROOT / "dinov3-local-vits16-qwen-qformer-itc-to-llm-smoke",
    ROOT / "dinov3-local-vits16-qwen-qformer-smoke",
    ROOT / "dinov3-local-vits16-qwen-smoke",
    ROOT / "dinov3-local-vits16-qwen-wikimedia-1k-adapter-stage1",
    ROOT / "local-dinov2-smoke",
    ROOT / "lora-perceiver-one-sample-smoke",
    ROOT / "open-vision-qwen-smoke",
    ROOT / "progress-output-one-sample-smoke",
    ROOT / "progress-tracking-smoke",
)


@dataclass(frozen=True)
class CleanupItem:
    source: str
    destination: str
    size_bytes: int
    action: str


def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv3 Mini VLM artifact 폴더 정리")
    parser.add_argument("--mode", choices=("dry-run", "archive", "delete"), default="dry-run")
    parser.add_argument("--archive-dir", default=str(DEFAULT_ARCHIVE_DIR))
    parser.add_argument("--yes", action="store_true", help="archive/delete 작업을 실제로 수행합니다.")
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    items = plan_cleanup(archive_dir=archive_dir)
    total_size = sum(item.size_bytes for item in items)
    print(f"mode: {args.mode}")
    print(f"candidate_count: {len(items)}")
    print(f"total_size: {format_bytes(total_size)}")
    for item in items:
        print(f"- {item.action}: {item.source} -> {item.destination} ({format_bytes(item.size_bytes)})")

    if args.mode == "dry-run":
        print("dry-run: 파일 이동/삭제는 수행하지 않았습니다.")
        return
    if not args.yes:
        raise SystemExit("실제 정리를 하려면 --yes를 함께 지정하세요.")

    if args.mode == "archive":
        archive_items(items=items, archive_dir=archive_dir)
        return
    delete_items(items)


def plan_cleanup(*, archive_dir: Path) -> list[CleanupItem]:
    active = {path.resolve() for path in ACTIVE_DIRS}
    items: list[CleanupItem] = []
    for source in ARCHIVE_CANDIDATES:
        if not source.exists():
            continue
        if source.resolve() in active:
            continue
        destination = archive_dir / source.name
        items.append(
            CleanupItem(
                source=str(source),
                destination=str(destination),
                size_bytes=directory_size(source),
                action="archive",
            )
        )
    return items


def archive_items(*, items: list[CleanupItem], archive_dir: Path) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    moved: list[CleanupItem] = []
    for item in items:
        source = Path(item.source)
        destination = next_available_path(Path(item.destination))
        if not source.exists():
            continue
        shutil.move(str(source), str(destination))
        moved.append(
            CleanupItem(
                source=item.source,
                destination=str(destination),
                size_bytes=item.size_bytes,
                action="archived",
            )
        )
    write_manifest(archive_dir=archive_dir, items=moved)
    print(f"archived_count: {len(moved)}")
    print(f"manifest: {archive_dir / 'manifest.json'}")


def delete_items(items: list[CleanupItem]) -> None:
    removed: list[CleanupItem] = []
    for item in items:
        source = Path(item.source)
        if not source.exists():
            continue
        shutil.rmtree(source)
        removed.append(
            CleanupItem(
                source=item.source,
                destination="",
                size_bytes=item.size_bytes,
                action="deleted",
            )
        )
    print(f"deleted_count: {len(removed)}")


def write_manifest(*, archive_dir: Path, items: list[CleanupItem]) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "archive_dir": str(archive_dir),
        "total_size_bytes": sum(item.size_bytes for item in items),
        "items": [asdict(item) for item in items],
        "restore_note": "필요하면 destination 폴더를 source 경로로 다시 이동하면 됩니다.",
    }
    (archive_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"사용 가능한 archive 경로를 찾지 못했습니다: {path}")


def directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


if __name__ == "__main__":
    main()
