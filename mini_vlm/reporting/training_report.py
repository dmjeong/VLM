from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingReportPaths:
    """학습 종료 후 생성되는 분석 산출물 경로 묶음."""

    csv_path: Path
    svg_path: Path
    markdown_path: Path


def write_training_report(output_dir: str | Path) -> TrainingReportPaths:
    """`training_summary.json`을 표/그래프/문서로 변환한다.

    의도: 학습 결과를 JSON 숫자만으로 남기면 다음 실험에서 변화 추이를 설명하기 어렵다.
    참고: `train_alignment.py`가 저장하는 epoch summary 계약.
    선택 이유: matplotlib 같은 추가 의존성 없이 SVG를 직접 생성하면 로컬/서버 어디서든 같은 산출물을 남길 수 있다.
    """

    target_dir = Path(output_dir)
    summary_path = target_dir / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"training_summary.json을 찾을 수 없습니다: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    if not isinstance(summary, dict):
        raise TypeError("training_summary.json의 최상위 값은 객체여야 합니다.")

    epoch_rows = build_epoch_rows(summary)
    csv_path = target_dir / "loss_curves.csv"
    svg_path = target_dir / "loss_curve.svg"
    markdown_path = target_dir / "training_report.md"

    write_loss_csv(epoch_rows, csv_path)
    write_loss_svg(epoch_rows, svg_path)
    write_markdown_report(summary, epoch_rows, markdown_path)
    return TrainingReportPaths(csv_path=csv_path, svg_path=svg_path, markdown_path=markdown_path)


def build_epoch_rows(summary: dict[str, Any]) -> list[dict[str, float | int | None]]:
    epochs = summary.get("epochs")
    if not isinstance(epochs, list):
        return []

    rows: list[dict[str, float | int | None]] = []
    for fallback_index, raw_epoch in enumerate(epochs, start=1):
        if not isinstance(raw_epoch, dict):
            continue
        epoch_index = _to_int(raw_epoch.get("epoch"), fallback_index - 1)
        rows.append(
            {
                "epoch": epoch_index + 1,
                "train_avg_loss": _to_float(raw_epoch.get("avg_loss")),
                "validation_loss": _to_float(raw_epoch.get("validation_loss")),
                "first_loss": _to_float(raw_epoch.get("first_loss")),
                "last_loss": _to_float(raw_epoch.get("last_loss")),
                "min_loss": _to_float(raw_epoch.get("min_loss")),
                "max_loss": _to_float(raw_epoch.get("max_loss")),
                "optimizer_step": _to_int(raw_epoch.get("optimizer_step"), 0),
            }
        )
    return rows


def write_loss_csv(rows: list[dict[str, float | int | None]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_avg_loss",
        "validation_loss",
        "first_loss",
        "last_loss",
        "min_loss",
        "max_loss",
        "optimizer_step",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_loss_svg(rows: list[dict[str, float | int | None]], path: Path) -> None:
    width = 860
    height = 440
    margin_left = 76
    margin_right = 34
    margin_top = 38
    margin_bottom = 64
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    series = [
        ("Train avg loss", "train_avg_loss", "#2563eb"),
        ("Validation loss", "validation_loss", "#dc2626"),
    ]
    values = [
        float(row[key])
        for row in rows
        for _, key, _ in series
        if isinstance(row.get(key), (int, float))
    ]
    if not values:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
            '<rect width="100%" height="100%" fill="#ffffff"/>\n'
            '<text x="40" y="60" fill="#111827" font-family="Arial" font-size="18">'
            "표시할 loss 값이 없습니다.</text>\n"
            "</svg>\n"
        )
        path.write_text(svg, encoding="utf-8")
        return

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        min_value -= 0.5
        max_value += 0.5
    padding = (max_value - min_value) * 0.08
    min_value -= padding
    max_value += padding

    def x_for(index: int) -> float:
        if len(rows) <= 1:
            return margin_left + plot_width / 2
        return margin_left + (plot_width * index / (len(rows) - 1))

    def y_for(value: float) -> float:
        ratio = (max_value - value) / (max_value - min_value)
        return margin_top + plot_height * ratio

    y_ticks = build_ticks(min_value, max_value, tick_count=5)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="76" y="24" fill="#111827" font-family="Arial" font-size="18" font-weight="700">Loss curve</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#374151" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#374151" stroke-width="1"/>',
    ]
    for tick in y_ticks:
        y = y_for(tick)
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" fill="#4b5563" font-family="Arial" font-size="12">{tick:.3g}</text>'
        )
    for index, row in enumerate(rows):
        x = x_for(index)
        epoch = int(row["epoch"] or index + 1)
        parts.append(
            f'<text x="{x:.2f}" y="{height - 30}" text-anchor="middle" fill="#4b5563" font-family="Arial" font-size="12">{epoch}</text>'
        )
    parts.append(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 8}" text-anchor="middle" fill="#374151" font-family="Arial" font-size="13">epoch</text>'
    )
    parts.append(
        f'<text x="18" y="{margin_top + plot_height / 2:.2f}" transform="rotate(-90 18 {margin_top + plot_height / 2:.2f})" text-anchor="middle" fill="#374151" font-family="Arial" font-size="13">loss</text>'
    )

    for label, key, color in series:
        points = []
        for index, row in enumerate(rows):
            value = row.get(key)
            if isinstance(value, (int, float)):
                points.append((x_for(index), y_for(float(value)), float(value)))
        if not points:
            continue
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{point_text}"/>')
        for x, y, value in points:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"><title>{html.escape(label)}: {value:.4f}</title></circle>')
    legend_x = width - 218
    legend_y = 24
    for offset, (label, _, color) in enumerate(series):
        y = legend_y + offset * 20
        parts.append(f'<rect x="{legend_x}" y="{y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 18}" y="{y}" fill="#374151" font-family="Arial" font-size="13">{html.escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_markdown_report(summary: dict[str, Any], rows: list[dict[str, float | int | None]], path: Path) -> None:
    best_validation = min(
        (row for row in rows if isinstance(row.get("validation_loss"), (int, float))),
        key=lambda row: float(row["validation_loss"]),
        default=None,
    )
    lines = [
        "# 학습 결과 리포트",
        "",
        f"- 실험명: `{summary.get('experiment_name', '')}`",
        f"- epoch 수: `{summary.get('epoch_count', '')}`",
        f"- 학습 샘플 수: `{summary.get('sample_count', '')}`",
        f"- batch/epoch: `{summary.get('batches_per_epoch', '')}`",
        f"- 최종 train avg loss: `{_format_value(summary.get('final_avg_loss'))}`",
    ]
    if best_validation is not None:
        lines.append(
            f"- best validation loss: `{_format_value(best_validation.get('validation_loss'))}` "
            f"(epoch `{best_validation.get('epoch')}`)"
        )
    lines.extend(
        [
            "",
            "## 생성 파일",
            "",
            "- `metrics.jsonl`: batch/epoch 단위 원본 로그",
            "- `training_summary.json`: epoch별 요약 JSON",
            "- `loss_curves.csv`: 그래프/스프레드시트용 loss 표",
            "- `loss_curve.svg`: train/validation loss 그래프",
            "- `training_report.md`: 현재 리포트",
            "",
            "## Epoch별 지표",
            "",
            "| epoch | train avg loss | validation loss | first loss | last loss | min loss | max loss | optimizer step |",
            "|------:|---------------:|----------------:|-----------:|----------:|---------:|---------:|---------------:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row.get('epoch')} | "
            f"{_format_value(row.get('train_avg_loss'))} | "
            f"{_format_value(row.get('validation_loss'))} | "
            f"{_format_value(row.get('first_loss'))} | "
            f"{_format_value(row.get('last_loss'))} | "
            f"{_format_value(row.get('min_loss'))} | "
            f"{_format_value(row.get('max_loss'))} | "
            f"{row.get('optimizer_step')} |"
        )
    lines.extend(
        [
            "",
            "## 해석 메모",
            "",
            "- train loss와 validation loss가 같이 내려가면 adapter가 데이터의 패턴을 학습하고 있을 가능성이 높다.",
            "- train loss만 내려가고 validation loss가 오르면 overfit 또는 validation 데이터 분포 차이를 의심한다.",
            "- validation loss가 가장 낮은 epoch를 다음 checkpoint 정책의 기준으로 삼는다.",
            "- 이 리포트는 loss 기반 분석이므로 실제 답변 품질은 `evaluation/test_report.md`와 함께 확인한다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_ticks(min_value: float, max_value: float, tick_count: int) -> list[float]:
    if tick_count <= 1:
        return [min_value]
    step = (max_value - min_value) / (tick_count - 1)
    return [min_value + step * index for index in range(tick_count)]


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _to_int(value: Any, fallback: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return fallback


def _format_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "N/A"
