from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from mini_vlm.config import load_config
from mini_vlm.evaluation.dashboard import choice_letter_match, prediction_success, write_evaluation_dashboard


@dataclass(frozen=True)
class ExperimentSpec:
    """비전 백본 비교에 필요한 실험 산출물 묶음."""

    name: str
    stage1_dir: str
    stage0_dir: str
    config_path: str


@dataclass(frozen=True)
class ComparisonPaths:
    """비교 리포트 생성 결과."""

    json_path: Path
    report_path: Path
    dashboard_path: Path


DEFAULT_EXPERIMENTS = (
    ExperimentSpec(
        name="DINOv3 ViT-S/16 1 epoch",
        stage1_dir="artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1",
        stage0_dir="artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-stage0",
        config_path="configs/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1.json",
    ),
    ExperimentSpec(
        name="DINOv3 ViT-S/16 epoch2 checkpoint",
        stage1_dir="artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch3",
        stage0_dir="artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-stage0",
        config_path="configs/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch3.json",
    ),
    ExperimentSpec(
        name="DINOv3 ViT-S/16 + Qwen LoRA Stage2 1 epoch",
        stage1_dir="artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage2-lora-epoch1",
        stage0_dir="artifacts/dinov3-mini-vlm/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-stage0",
        config_path="configs/dinov3-local-vits16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage2-lora-epoch1.json",
    ),
    ExperimentSpec(
        name="CLIP ViT-B/16 1 epoch",
        stage1_dir="artifacts/dinov3-mini-vlm/clip-vitb16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1",
        stage0_dir="artifacts/dinov3-mini-vlm/clip-vitb16-qwen-external-10k-qformer-distilbert-itc-stage0",
        config_path="configs/clip-vitb16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1.json",
    ),
    ExperimentSpec(
        name="SigLIP ViT-B/16 1 epoch",
        stage1_dir="artifacts/dinov3-mini-vlm/siglip-vitb16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1",
        stage0_dir="artifacts/dinov3-mini-vlm/siglip-vitb16-qwen-external-10k-qformer-distilbert-itc-stage0",
        config_path="configs/siglip-vitb16-qwen-external-10k-qformer-distilbert-itc-to-llm-stage1-epoch1.json",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision encoder ablation 비교 리포트 생성")
    parser.add_argument("--output-dir", default="artifacts/dinov3-mini-vlm/vision-ablation")
    parser.add_argument(
        "--skip-dashboard-refresh",
        action="store_true",
        help="기존 test summary/predictions로 개별 평가 대시보드를 재생성하지 않습니다.",
    )
    args = parser.parse_args()

    paths = write_comparison(
        experiments=DEFAULT_EXPERIMENTS,
        output_dir=Path(args.output_dir),
        refresh_dashboards=not args.skip_dashboard_refresh,
    )
    print(f"comparison json: {paths.json_path}")
    print(f"comparison report: {paths.report_path}")
    print(f"comparison dashboard: {paths.dashboard_path}")


def write_comparison(
    *,
    experiments: Sequence[ExperimentSpec] = DEFAULT_EXPERIMENTS,
    output_dir: str | Path,
    refresh_dashboards: bool = True,
) -> ComparisonPaths:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = [summarize_experiment(experiment=experiment, refresh_dashboard=refresh_dashboards) for experiment in experiments]
    mirror_dashboards(rows=rows, output_dir=output_path)
    json_path = output_path / "comparison.json"
    md_path = output_path / "comparison.md"
    dashboard_path = output_path / "dashboard.html"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(rows, base_dir=output_path), encoding="utf-8")
    dashboard_path.write_text(render_comparison_dashboard(rows, base_dir=output_path), encoding="utf-8")
    return ComparisonPaths(json_path=json_path, report_path=md_path, dashboard_path=dashboard_path)


def summarize_experiment(*, experiment: ExperimentSpec, refresh_dashboard: bool = True) -> dict[str, Any]:
    stage1_path = Path(experiment.stage1_dir)
    stage0_path = Path(experiment.stage0_dir)
    config_path = Path(experiment.config_path)
    training_summary = read_optional_json(stage1_path / "training_summary.json")
    if not training_summary:
        training_summary = read_training_summary_from_metrics(stage1_path / "metrics.jsonl")
    test_summary = read_optional_json(stage1_path / "evaluation" / "test_summary.json")
    itc_summary = read_optional_json(stage0_path / "itc_summary.json")
    predictions_path = stage1_path / "evaluation" / "test_predictions.jsonl"
    predictions = read_optional_jsonl(predictions_path)
    dashboard_path = stage1_path / "evaluation" / "dashboard.html"
    dashboard_status = refresh_dashboard_if_possible(
        config_path=config_path,
        summary_path=stage1_path / "evaluation" / "test_summary.json",
        predictions_path=predictions_path,
        dashboard_path=dashboard_path,
        test_summary=test_summary,
        refresh_dashboard=refresh_dashboard,
    )
    choice_stats = summarize_choice_predictions(predictions)
    success_stats = summarize_prediction_success(predictions)
    loaded_adapter = (test_summary.get("loaded_files") or {}).get("visual_adapter") if test_summary else None
    return {
        "name": experiment.name,
        "config_path": str(config_path),
        "stage0_dir": str(stage0_path),
        "stage1_dir": str(stage1_path),
        "dashboard_path": str(dashboard_path) if dashboard_path.exists() else "",
        "dashboard_status": dashboard_status,
        "status": resolve_status(training_summary=training_summary, test_summary=test_summary),
        "checkpoint_note": checkpoint_note(loaded_adapter),
        "itc_validation_loss": first_epoch_value(itc_summary, "validation_loss"),
        "train_avg_loss": first_epoch_value(training_summary, "avg_loss"),
        "validation_loss": first_epoch_value(training_summary, "validation_loss"),
        "test_avg_loss": test_summary.get("avg_loss") if test_summary else None,
        "exact_match_rate": test_summary.get("exact_match_rate") if test_summary else None,
        "contains_answer_rate": test_summary.get("contains_answer_rate") if test_summary else None,
        "avg_token_overlap": test_summary.get("avg_token_overlap") if test_summary else None,
        "choice_letter_accuracy": choice_stats["accuracy"],
        "choice_letter_count": choice_stats["count"],
        "generation_success_rate": success_stats["accuracy"],
        "generation_success_count": success_stats["count"],
    }


def refresh_dashboard_if_possible(
    *,
    config_path: Path,
    summary_path: Path,
    predictions_path: Path,
    dashboard_path: Path,
    test_summary: dict[str, Any],
    refresh_dashboard: bool,
) -> str:
    if not refresh_dashboard:
        return "건너뜀"
    if not test_summary or not summary_path.exists() or not predictions_path.exists():
        return "평가 결과 없음"
    if not config_path.exists():
        return "config 없음"
    config = load_config(config_path)
    dataset_path = Path(str(test_summary.get("dataset_path") or config.test_jsonl))
    if not dataset_path.exists():
        return "dataset 없음"
    try:
        write_evaluation_dashboard(
            summary_path=summary_path,
            predictions_path=predictions_path,
            dataset_path=dataset_path,
            image_root=config.image_root,
            output_path=dashboard_path,
        )
    except Exception as exc:  # pragma: no cover - 실패 메시지를 비교표에 남기기 위한 방어막
        return f"실패: {exc}"
    return "업데이트"


def summarize_choice_predictions(predictions: list[dict[str, Any]]) -> dict[str, float | int | None]:
    matches: list[bool] = []
    for prediction in predictions:
        match = choice_letter_match(
            str(prediction.get("expected_answer") or ""),
            str(prediction.get("generated_answer") or ""),
        )
        if match is not None:
            matches.append(match)
    if not matches:
        return {"accuracy": None, "count": 0}
    return {"accuracy": sum(1 for item in matches if item) / len(matches), "count": len(matches)}


def summarize_prediction_success(predictions: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not predictions:
        return {"accuracy": None, "count": 0}
    successes = sum(1 for prediction in predictions if prediction_success(prediction))
    return {"accuracy": successes / len(predictions), "count": len(predictions)}


def resolve_status(*, training_summary: dict[str, Any], test_summary: dict[str, Any]) -> str:
    if training_summary and test_summary:
        return "완료"
    if training_summary:
        return "평가대기"
    return "대기"


def checkpoint_note(loaded_adapter: Any) -> str:
    if not isinstance(loaded_adapter, str):
        return ""
    name = Path(loaded_adapter).name
    if name == "visual_adapter.pt":
        return ""
    return f"평가 adapter: `{name}`"


def first_epoch_value(summary: dict[str, Any], key: str) -> Any:
    epochs = summary.get("epochs") if summary else None
    if isinstance(epochs, list) and epochs:
        return epochs[-1].get(key)
    return None


def read_training_summary_from_metrics(path: Path) -> dict[str, Any]:
    rows = read_optional_jsonl(path)
    epoch_summaries = [row for row in rows if row.get("event") == "epoch_summary"]
    if not epoch_summaries:
        return {}
    return {
        "epoch_count": len(epoch_summaries),
        "final_avg_loss": epoch_summaries[-1].get("avg_loss"),
        "epochs": epoch_summaries,
        "source": str(path),
    }


def render_markdown(rows: list[dict[str, Any]], *, base_dir: Path) -> str:
    lines = [
        "# Vision Encoder Ablation 비교",
        "",
        f"- 통합 대시보드: [dashboard.html]({relative_link(base_dir / 'dashboard.html', base_dir)})",
        "",
        "| 실험 | 상태 | ITC val loss | train loss | val loss | test loss | choice acc | success | overlap | dashboard |",
        "|------|------|--------------|------------|----------|-----------|------------|---------|---------|-----------|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
                    str(row["status"]),
                    format_metric(row["itc_validation_loss"]),
                    format_metric(row["train_avg_loss"]),
                    format_metric(row["validation_loss"]),
                    format_metric(row["test_avg_loss"]),
                    format_percent(row["choice_letter_accuracy"], row["choice_letter_count"]),
                    format_percent(row["generation_success_rate"], row["generation_success_count"]),
                    format_percent(row["avg_token_overlap"], None),
                    format_dashboard_link(row, base_dir),
                ]
            )
            + " |"
        )
    lines.extend(["", "## 체크포인트 메모", ""])
    notes = [row for row in rows if row.get("checkpoint_note")]
    if notes:
        for row in notes:
            lines.append(f"- {row['name']}: {row['checkpoint_note']}")
    else:
        lines.append("- 모든 실험은 최종 `visual_adapter.pt`로 평가했다.")
    lines.extend(
        [
            "",
            "## 대시보드 갱신 상태",
            "",
            *[f"- {row['name']}: {row.get('dashboard_status') or '-'}" for row in rows],
            "",
            "## 해석 기준",
            "",
            "- `test loss`는 정답 token 예측 손실이다. 낮을수록 좋다.",
            "- `choice acc`는 MMBench처럼 `A. ...` 형식 정답이 있는 생성 샘플에서 letter가 맞은 비율이다.",
            "- `success`는 대시보드의 answer match 기준이다. exact/contains뿐 아니라 choice letter, yes/no 극성, count, 짧은 사실형 답변, object hit, 높은 token overlap을 분리해서 판단한다.",
            "- `overlap`은 정답 핵심 단어 회수율에 가까운 보조 지표다.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_comparison_dashboard(rows: list[dict[str, Any]], *, base_dir: Path) -> str:
    best_loss = best_row(rows, "test_avg_loss", lower_is_better=True)
    best_choice = best_row(rows, "choice_letter_accuracy", lower_is_better=False)
    best_success = best_row(rows, "generation_success_rate", lower_is_better=False)
    table_rows = "\n".join(render_dashboard_table_row(row, base_dir=base_dir) for row in rows)
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vision Encoder Ablation 비교</title>
<style>
:root {{ color-scheme: light; --bg: #f6f7f9; --ink: #111827; --muted: #667085; --line: #d8dee9; --panel: #fff; --blue: #2563eb; --green: #15803d; --red: #dc2626; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
main {{ width: min(1280px, calc(100% - 40px)); margin: 0 auto; padding: 30px 0 48px; }}
.hero {{ display: flex; justify-content: space-between; gap: 20px; align-items: flex-end; border-bottom: 1px solid var(--line); padding-bottom: 20px; }}
h1 {{ margin: 0 0 8px; font-size: 34px; letter-spacing: 0; }}
p {{ color: var(--muted); }}
.cards {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
.card, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
.card {{ padding: 16px; min-height: 120px; }}
.card span {{ color: var(--muted); font-size: 13px; }}
.card strong {{ display: block; margin-top: 8px; font-size: 22px; }}
.card em {{ display: block; margin-top: 6px; color: var(--muted); font-style: normal; }}
.panel {{ padding: 18px; overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; min-width: 980px; }}
th, td {{ padding: 12px 10px; border-bottom: 1px solid var(--line); text-align: left; font-size: 14px; }}
th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
a {{ color: var(--blue); text-decoration: none; font-weight: 700; }}
.status {{ display: inline-flex; padding: 3px 8px; border-radius: 999px; background: #eef2ff; color: #3730a3; font-size: 12px; font-weight: 700; }}
.note {{ margin-top: 16px; display: grid; gap: 8px; }}
.note div {{ border-left: 4px solid var(--blue); background: #eff6ff; padding: 10px 12px; border-radius: 6px; color: #1e3a8a; }}
@media (max-width: 760px) {{ main {{ width: calc(100% - 24px); }} .hero {{ display: block; }} .cards {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
<section class="hero">
<div>
<h1>Vision Encoder Ablation 비교</h1>
<p>DINOv3, CLIP, SigLIP 평가 결과와 개별 이미지 대시보드를 한곳에서 본다.</p>
</div>
<a href="{html.escape(relative_link(base_dir / 'comparison.md', base_dir), quote=True)}">comparison.md</a>
</section>
<section class="cards">
{summary_card("Best test loss", best_loss, "test_avg_loss", lower_is_better=True)}
{summary_card("Best choice acc", best_choice, "choice_letter_accuracy", lower_is_better=False)}
{summary_card("Best success", best_success, "generation_success_rate", lower_is_better=False)}
</section>
<section class="panel">
<table>
<thead>
<tr>
<th>실험</th><th>상태</th><th>ITC val</th><th>Train</th><th>Val</th><th>Test</th><th>Choice</th><th>Success</th><th>Overlap</th><th>Dashboard</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
</section>
<section class="note">
{render_dashboard_notes(rows)}
</section>
</main>
</body>
</html>
"""


def render_dashboard_table_row(row: dict[str, Any], *, base_dir: Path) -> str:
    return (
        "<tr>"
        f"<td>{escape(row.get('name'))}</td>"
        f'<td><span class="status">{escape(row.get("status"))}</span></td>'
        f"<td>{escape(format_metric(row.get('itc_validation_loss')))}</td>"
        f"<td>{escape(format_metric(row.get('train_avg_loss')))}</td>"
        f"<td>{escape(format_metric(row.get('validation_loss')))}</td>"
        f"<td>{escape(format_metric(row.get('test_avg_loss')))}</td>"
        f"<td>{escape(format_percent(row.get('choice_letter_accuracy'), row.get('choice_letter_count')))}</td>"
        f"<td>{escape(format_percent(row.get('generation_success_rate'), row.get('generation_success_count')))}</td>"
        f"<td>{escape(format_percent(row.get('avg_token_overlap'), None))}</td>"
        f"<td>{dashboard_html_link(row, base_dir)}</td>"
        "</tr>"
    )


def render_dashboard_notes(rows: list[dict[str, Any]]) -> str:
    notes: list[str] = []
    for row in rows:
        if row.get("checkpoint_note"):
            notes.append(f"{row['name']}: {row['checkpoint_note']}")
        if row.get("dashboard_status") != "업데이트":
            notes.append(f"{row['name']}: 대시보드 갱신 상태 {row.get('dashboard_status')}")
    if not notes:
        notes.append("모든 개별 평가 대시보드를 최신 summary/predictions 기준으로 갱신했다.")
    return "\n".join(f"<div>{escape(note)}</div>" for note in notes)


def summary_card(label: str, row: dict[str, Any] | None, key: str, *, lower_is_better: bool) -> str:
    if not row:
        return f'<article class="card"><span>{escape(label)}</span><strong>-</strong><em>결과 없음</em></article>'
    value = format_metric(row.get(key)) if key.endswith("loss") else format_percent(row.get(key), None)
    direction = "낮을수록 좋음" if lower_is_better else "높을수록 좋음"
    return (
        '<article class="card">'
        f"<span>{escape(label)}</span>"
        f"<strong>{escape(value)}</strong>"
        f"<em>{escape(row.get('name'))} · {escape(direction)}</em>"
        "</article>"
    )


def best_row(rows: list[dict[str, Any]], key: str, *, lower_is_better: bool) -> dict[str, Any] | None:
    candidates = [row for row in rows if isinstance(row.get(key), (int, float))]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row[key])) if lower_is_better else max(
        candidates, key=lambda row: float(row[key])
    )


def format_dashboard_link(row: dict[str, Any], base_dir: Path) -> str:
    path = str(row.get("comparison_dashboard_path") or row.get("dashboard_path") or "")
    if not path:
        return "-"
    return f"[열기]({relative_link(Path(path), base_dir)})"


def dashboard_html_link(row: dict[str, Any], base_dir: Path) -> str:
    path = str(row.get("comparison_dashboard_path") or row.get("dashboard_path") or "")
    if not path:
        return "-"
    href = relative_link(Path(path), base_dir)
    return f'<a href="{html.escape(href, quote=True)}" target="_blank" rel="noreferrer">열기</a>'


def mirror_dashboards(*, rows: list[dict[str, Any]], output_dir: Path) -> None:
    mirror_root = output_dir / "experiments"
    if mirror_root.exists():
        shutil.rmtree(mirror_root)
    mirror_root.mkdir(parents=True, exist_ok=True)
    used_slugs: set[str] = set()
    for row in rows:
        source = Path(str(row.get("dashboard_path") or ""))
        if not source.exists():
            continue
        slug = unique_slug(slugify(str(row.get("name") or "experiment")), used_slugs)
        target_dir = mirror_root / slug
        target_dir.mkdir(parents=True, exist_ok=True)
        target_dashboard = target_dir / "dashboard.html"
        shutil.copy2(source, target_dashboard)
        source_assets = source.parent / "dashboard_assets"
        if source_assets.exists():
            shutil.copytree(source_assets, target_dir / "dashboard_assets")
        row["comparison_dashboard_path"] = str(target_dashboard)


def slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z가-힣._-]+", "-", value.strip()).strip("-").lower()
    return slug or "experiment"


def unique_slug(slug: str, used_slugs: set[str]) -> str:
    if slug not in used_slugs:
        used_slugs.add(slug)
        return slug
    for index in range(2, 1000):
        candidate = f"{slug}-{index}"
        if candidate not in used_slugs:
            used_slugs.add(candidate)
            return candidate
    raise RuntimeError(f"사용 가능한 dashboard slug를 찾지 못했습니다: {slug}")


def relative_link(target: Path, base_dir: Path) -> str:
    return os.path.relpath(target, start=base_dir).replace(os.sep, "/")


def read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else {}


def read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "-"


def format_percent(value: Any, count: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    suffix = f" ({count})" if isinstance(count, int) and count > 0 else ""
    return f"{float(value) * 100:.1f}%{suffix}"


def escape(value: Any) -> str:
    return html.escape(str(value or ""))


if __name__ == "__main__":
    main()
