from __future__ import annotations

import html
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "from",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "its",
    "near",
    "of",
    "on",
    "or",
    "that",
    "the",
    "there",
    "this",
    "to",
    "what",
    "with",
    "you",
}
GENERIC_VISUAL_TOKENS = {
    "answer",
    "appear",
    "appears",
    "based",
    "describe",
    "depict",
    "depicts",
    "directly",
    "image",
    "interior",
    "look",
    "main",
    "object",
    "placed",
    "photo",
    "picture",
    "positioned",
    "primarily",
    "see",
    "seen",
    "shows",
    "slightly",
    "subject",
    "surface",
    "visible",
}
CANONICAL_TOKEN_MAP = {
    "aeroplane": "plane",
    "airplane": "plane",
    "beneath": "below",
    "bridal": "wedding",
    "gliding": "swimming",
    "gown": "dress",
    "under": "below",
    "underneath": "below",
}
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


@dataclass(frozen=True)
class DashboardImageGroup:
    """대시보드의 이미지 단위 평가 묶음."""

    image: str
    image_uri: str
    object_label: str
    label_kind: str
    category: str
    file_title: str
    source_url: str
    sample_count: int
    exact_rate: float
    contains_rate: float
    answer_match_rate: float
    object_hit_rate: float | None
    choice_letter_rate: float | None
    assessment_warning_rate: float
    avg_token_overlap: float
    severity: str
    issues: list[str]
    predictions: list[dict[str, Any]]


def write_evaluation_dashboard(
    *,
    summary_path: str | Path,
    predictions_path: str | Path,
    dataset_path: str | Path,
    image_root: str | Path,
    output_path: str | Path,
) -> Path:
    """평가 결과를 이미지 중심 HTML 대시보드로 만든다.

    의도: JSONL은 분석에는 좋지만, 어떤 이미지에서 어떤 질문이 실패했는지 한눈에 보기 어렵다.
    참고: `evaluate_cli.py`가 저장하는 `*_summary.json`과 `*_predictions.jsonl`.
    선택 이유: 학습 산출물 폴더 안에서 바로 열 수 있는 정적 HTML이면 별도 서버 없이 실험 리뷰가 가능하다.
    """

    target = Path(output_path)
    summary = read_json(summary_path)
    predictions = read_jsonl(predictions_path)
    samples_by_id = {str(sample.get("sample_id")): sample for sample in read_jsonl(dataset_path)}
    asset_dir = target.parent / "dashboard_assets" / "images"
    groups = build_image_groups(
        predictions=predictions,
        samples_by_id=samples_by_id,
        image_root=Path(image_root),
        html_dir=target.parent,
        asset_dir=asset_dir,
    )
    dashboard = render_dashboard(summary=summary, groups=groups, predictions=predictions)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dashboard, encoding="utf-8")
    return target


def build_image_groups(
    *,
    predictions: list[dict[str, Any]],
    samples_by_id: dict[str, dict[str, Any]],
    image_root: Path,
    html_dir: Path | None = None,
    asset_dir: Path | None = None,
) -> list[DashboardImageGroup]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for prediction in predictions:
        sample = samples_by_id.get(str(prediction.get("sample_id")), {})
        merged = merge_prediction_with_sample(prediction, sample)
        grouped.setdefault(str(merged.get("image") or ""), []).append(merged)

    image_groups: list[DashboardImageGroup] = []
    for image, rows in sorted(grouped.items()):
        metadata = first_metadata(rows)
        object_label = str(metadata.get("object") or "").strip()
        category = str(metadata.get("category") or first_row_value(rows, "task") or metadata.get("source") or "unknown")
        label_kind = "object" if object_label else ("category" if metadata.get("category") else "task")
        display_label = object_label or category or Path(image).stem
        file_title = str(metadata.get("file_title") or image)
        source_url = str(metadata.get("description_url") or "")
        exact_rate = mean_bool(rows, "exact_match")
        contains_rate = mean_bool(rows, "contains_answer")
        answer_match_rate = mean_bool(rows, "answer_match")
        object_hit_rate = mean_optional_bool(rows, "object_hit")
        choice_letter_rate = mean_optional_bool(rows, "choice_letter_match")
        assessment_warning_rate = mean_bool(rows, "assessment_warning")
        avg_overlap = mean_float(rows, "token_overlap")
        issues = describe_group_issues(
            label=display_label,
            exact_rate=exact_rate,
            contains_rate=contains_rate,
            answer_match_rate=answer_match_rate,
            object_hit_rate=object_hit_rate,
            choice_letter_rate=choice_letter_rate,
            assessment_warning_rate=assessment_warning_rate,
            avg_overlap=avg_overlap,
        )
        image_path = resolve_image_path(image_root=image_root, image=image)
        image_groups.append(
            DashboardImageGroup(
                image=image,
                image_uri=build_image_uri(image_path=image_path, image=image, html_dir=html_dir, asset_dir=asset_dir),
                object_label=display_label,
                label_kind=label_kind,
                category=category,
                file_title=file_title,
                source_url=source_url,
                sample_count=len(rows),
                exact_rate=exact_rate,
                contains_rate=contains_rate,
                answer_match_rate=answer_match_rate,
                object_hit_rate=object_hit_rate,
                choice_letter_rate=choice_letter_rate,
                assessment_warning_rate=assessment_warning_rate,
                avg_token_overlap=avg_overlap,
                severity=classify_severity(
                    answer_match_rate=answer_match_rate,
                    contains_rate=contains_rate,
                    object_hit_rate=object_hit_rate,
                    choice_letter_rate=choice_letter_rate,
                    assessment_warning_rate=assessment_warning_rate,
                    avg_overlap=avg_overlap,
                ),
                issues=issues,
                predictions=rows,
            )
        )
    return image_groups


def merge_prediction_with_sample(prediction: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    object_label = str(metadata.get("object") or "")
    generated_answer = str(prediction.get("generated_answer") or "")
    expected_answer = str(prediction.get("expected_answer") or sample.get("answer") or "")
    merged = {
        **prediction,
        "task": sample.get("task") or "",
        "metadata": metadata,
        "object_hit": object_label_hits_generation(object_label, generated_answer) if object_label else None,
        "choice_letter_match": choice_letter_match(expected_answer, generated_answer),
    }
    if not merged.get("image") and sample.get("image"):
        merged["image"] = sample.get("image")
    if not merged.get("question") and sample.get("question"):
        merged["question"] = sample.get("question")
    if not merged.get("expected_answer") and sample.get("answer"):
        merged["expected_answer"] = sample.get("answer")
    merged.update(assess_prediction_answer(merged))
    return merged


def render_dashboard(
    *,
    summary: dict[str, Any],
    groups: list[DashboardImageGroup],
    predictions: list[dict[str, Any]],
) -> str:
    coverage = coverage_rate(summary)
    issues = describe_global_issues(summary=summary, groups=groups, coverage=coverage)
    worst_groups = sorted(groups, key=weakness_sort_key)[:5]
    generated_count = int(summary.get("generation_sample_count") or len(predictions))
    answer_match_rate = overall_answer_match_rate(groups)

    body = [
        "<!doctype html>",
        '<html lang="ko">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>DINOv3 Mini VLM 평가 대시보드</title>",
        "<style>",
        stylesheet(),
        "</style>",
        "</head>",
        "<body>",
        '<main class="page">',
        '<section class="hero">',
        "<div>",
        '<p class="eyebrow">DINOv3 Mini VLM Evaluation</p>',
        "<h1>이미지별 평가 대시보드</h1>",
        f"<p>{escape(summary.get('checkpoint'))}</p>",
        "</div>",
        f'<div class="run-meta"><span>split</span><strong>{escape(summary.get("split"))}</strong><span>dataset</span><strong>{escape(summary.get("dataset_path"))}</strong></div>',
        "</section>",
        '<section class="metric-grid">',
        metric_card("Test loss", format_metric(summary.get("avg_loss")), "정답 토큰 예측 손실"),
        metric_card("Exact match", format_percent(summary.get("exact_match_rate")), "생성 답변 완전 일치"),
        metric_card("Contains answer", format_percent(summary.get("contains_answer_rate")), "정답 문장 포함 비율"),
        metric_card("Answer match", format_percent(answer_match_rate), "대시보드 의미 기준 통과율"),
        metric_card("Token overlap", format_percent(summary.get("avg_token_overlap")), "정답 단어 회수율"),
        metric_card("Coverage", format_percent(coverage), f"{generated_count}/{summary.get('sample_count')} samples"),
        metric_card("Images", str(len(groups)), "생성 평가가 있는 이미지 수"),
        "</section>",
        '<section class="panel">',
        "<h2>문제 요약</h2>",
        '<div class="issue-list">',
        "".join(f'<div class="issue">{escape(issue)}</div>' for issue in issues),
        "</div>",
        "</section>",
        '<section class="panel">',
        "<h2>가장 취약한 이미지 Top 5</h2>",
        f"<p>이 섹션은 전체 평가 이미지 {len(groups)}개 중 취약도가 높은 5개만 보여주는 요약입니다.</p>",
        '<div class="weak-grid">',
        "".join(render_weak_group(group) for group in worst_groups),
        "</div>",
        "</section>",
        '<section class="panel section-intro">',
        "<h2>전체 테스트 이미지</h2>",
        f"<p>아래 목록이 생성 평가가 수행된 이미지 {len(groups)}개 전체입니다. 검색과 필터를 적용하면 보이는 이미지 수가 함께 바뀝니다.</p>",
        "</section>",
        '<section class="toolbar" aria-label="대시보드 필터">',
        '<input id="searchInput" type="search" placeholder="이미지, 분류, 질문, 답변 검색">',
        '<select id="severityFilter"><option value="all">전체 상태</option><option value="심각">심각</option><option value="주의">주의</option><option value="양호">양호</option></select>',
        '<select id="objectFilter"><option value="all">전체 분류</option>',
        "".join(f'<option value="{escape(group.object_label)}">{escape(group.object_label)}</option>' for group in unique_groups_by_object(groups)),
        "</select>",
        f'<output id="visibleCount" class="visible-count">{len(groups)} / {len(groups)} images</output>',
        "</section>",
        '<section class="image-grid" id="imageGrid">',
        "".join(render_image_group(group) for group in groups),
        "</section>",
        "</main>",
        "<script>",
        script(),
        "</script>",
        "</body>",
        "</html>",
    ]
    return "\n".join(body)


def metric_card(label: str, value: str, caption: str) -> str:
    return (
        '<article class="metric-card">'
        f"<span>{escape(label)}</span>"
        f"<strong>{escape(value)}</strong>"
        f"<p>{escape(caption)}</p>"
        "</article>"
    )


def render_weak_group(group: DashboardImageGroup) -> str:
    return (
        f'<article class="weak-card severity-{escape(group.severity)}">'
        f"<strong>{escape(group.object_label)}</strong>"
        f"<span>{escape(group.file_title)}</span>"
        f"<p>{escape(primary_metric_label(group))} {format_percent(primary_success_rate(group))} · overlap {format_percent(group.avg_token_overlap)}</p>"
        "</article>"
    )


def render_image_group(group: DashboardImageGroup) -> str:
    issue_html = "".join(f"<li>{escape(issue)}</li>" for issue in group.issues)
    prediction_rows = "".join(render_prediction_row(row) for row in group.predictions)
    metric_items = [
        ("samples", str(group.sample_count)),
        ("answer", format_percent(group.answer_match_rate)),
        ("exact", format_percent(group.exact_rate)),
        ("contains", format_percent(group.contains_rate)),
    ]
    if group.choice_letter_rate is not None:
        metric_items.append(("choice", format_percent(group.choice_letter_rate)))
    if group.object_hit_rate is not None:
        metric_items.append(("object", format_percent(group.object_hit_rate)))
    metric_items.append(("overlap", format_percent(group.avg_token_overlap)))
    metric_html = "".join(
        f"<span>{escape(label)} <strong>{escape(value)}</strong></span>" for label, value in metric_items
    )
    source_link = (
        f'<a href="{escape_attr(group.source_url)}" target="_blank" rel="noreferrer">Wikimedia 원본</a>'
        if group.source_url
        else ""
    )
    return (
        f'<article class="image-card" data-severity="{escape_attr(group.severity)}" '
        f'data-object="{escape_attr(group.object_label)}" data-search="{escape_attr(search_blob_for_group(group))}">'
        '<div class="image-card-header">'
        f'<img src="{escape_attr(group.image_uri)}" alt="{escape_attr(group.object_label)}">'
        "<div>"
        f'<span class="badge severity-{escape_attr(group.severity)}">{escape(group.severity)}</span>'
        f"<h3>{escape(group.object_label)}</h3>"
        f'<p class="label-kind">{escape(group.label_kind)}</p>'
        f"<p>{escape(group.file_title)}</p>"
        f"{source_link}"
        "</div>"
        "</div>"
        '<div class="mini-metrics">'
        f"{metric_html}"
        "</div>"
        f'<ul class="issues">{issue_html}</ul>'
        '<details class="predictions">'
        "<summary>질문별 결과</summary>"
        f"{prediction_rows}"
        "</details>"
        "</article>"
    )


def render_prediction_row(row: dict[str, Any]) -> str:
    state = "ok" if prediction_success(row) else "bad"
    score_parts = [
        f"answer {format_bool(row.get('answer_match'))}",
        f"exact {format_bool(row.get('exact_match'))}",
        f"contains {format_bool(row.get('contains_answer'))}",
    ]
    if row.get("answer_match_reason"):
        score_parts.append(f"reason {row.get('answer_match_reason')}")
    if row.get("choice_letter_match") is not None:
        score_parts.append(f"choice {format_bool(row.get('choice_letter_match'))}")
    if row.get("choice_explanation_warning"):
        score_parts.append("choice text warning")
    if row.get("yes_no_match") is not None:
        score_parts.append(f"yes/no {format_bool(row.get('yes_no_match'))}")
    if row.get("count_match") is not None:
        score_parts.append(f"count {format_bool(row.get('count_match'))}")
    if row.get("object_hit") is not None:
        score_parts.append(f"object {format_bool(row.get('object_hit'))}")
    score_parts.append(f"overlap {format_percent(row.get('token_overlap'))}")
    return (
        f'<div class="prediction-row {state}">'
        "<div>"
        f"<span>{escape(row.get('task') or row.get('sample_id'))}</span>"
        f"<strong>{escape(row.get('question'))}</strong>"
        "</div>"
        f"<p><b>정답</b>{escape(row.get('expected_answer'))}</p>"
        f"<p><b>생성</b>{escape(row.get('generated_answer'))}</p>"
        f"<em>{escape(' · '.join(score_parts))}</em>"
        "</div>"
    )


def stylesheet() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --ink: #111827;
  --muted: #667085;
  --line: #d9dee8;
  --panel: #ffffff;
  --blue: #2563eb;
  --green: #15803d;
  --amber: #b45309;
  --red: #dc2626;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.page { width: min(1440px, calc(100% - 40px)); margin: 0 auto; padding: 28px 0 48px; }
.hero { display: flex; justify-content: space-between; gap: 24px; align-items: flex-end; padding: 10px 0 22px; border-bottom: 1px solid var(--line); }
.hero h1 { margin: 4px 0 8px; font-size: 34px; line-height: 1.1; letter-spacing: 0; }
.hero p { margin: 0; color: var(--muted); overflow-wrap: anywhere; }
.eyebrow { color: var(--blue) !important; font-weight: 700; font-size: 13px; }
.run-meta { display: grid; grid-template-columns: auto 1fr; gap: 6px 10px; min-width: min(420px, 100%); font-size: 13px; }
.run-meta span { color: var(--muted); }
.run-meta strong { overflow-wrap: anywhere; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 18px 0; }
.metric-card, .panel, .image-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
.metric-card { padding: 14px; min-height: 112px; }
.metric-card span { display: block; color: var(--muted); font-size: 13px; }
.metric-card strong { display: block; margin-top: 10px; font-size: 28px; letter-spacing: 0; }
.metric-card p { margin: 8px 0 0; color: var(--muted); font-size: 13px; }
.panel { padding: 18px; margin: 16px 0; }
.panel h2 { margin: 0 0 12px; font-size: 18px; }
.panel p { margin: 0; color: var(--muted); overflow-wrap: anywhere; }
.section-intro { border-left: 4px solid var(--blue); }
.issue-list { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.issue { border-left: 4px solid var(--red); background: #fff5f5; padding: 10px 12px; border-radius: 6px; color: #7f1d1d; }
.weak-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; }
.weak-card { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fff; }
.weak-card strong, .weak-card span { display: block; overflow-wrap: anywhere; }
.weak-card span { margin-top: 5px; color: var(--muted); font-size: 12px; }
.weak-card p { margin: 8px 0 0; font-size: 12px; }
.toolbar { position: sticky; top: 0; z-index: 3; display: grid; grid-template-columns: 1fr 170px 180px 140px; gap: 10px; padding: 12px 0; background: rgba(247, 248, 250, 0.95); backdrop-filter: blur(8px); }
input, select { width: 100%; border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; font: inherit; background: #fff; color: var(--ink); }
.visible-count { display: flex; align-items: center; justify-content: center; border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; background: #fff; color: var(--muted); font-size: 13px; white-space: nowrap; }
.image-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.image-card { padding: 14px; min-width: 0; }
.image-card-header { display: grid; grid-template-columns: 168px 1fr; gap: 14px; align-items: start; }
.image-card-header > div { min-width: 0; }
.image-card img { width: 168px; max-width: 100%; aspect-ratio: 4 / 3; object-fit: cover; border-radius: 6px; border: 1px solid var(--line); background: #eef2f7; }
.image-card h3 { margin: 8px 0 4px; font-size: 22px; overflow-wrap: anywhere; }
.image-card p { margin: 0; color: var(--muted); overflow-wrap: anywhere; }
.label-kind { display: inline-block; margin: 0 0 6px !important; padding: 2px 7px; border-radius: 999px; background: #eef2ff; color: #3730a3 !important; font-size: 12px; font-weight: 700; }
.image-card a { display: inline-block; margin-top: 8px; color: var(--blue); text-decoration: none; font-size: 13px; }
.badge { display: inline-flex; align-items: center; min-height: 24px; padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }
.severity-심각 { border-color: #fecaca; background: #fef2f2; color: var(--red); }
.severity-주의 { border-color: #fed7aa; background: #fff7ed; color: var(--amber); }
.severity-양호 { border-color: #bbf7d0; background: #f0fdf4; color: var(--green); }
.mini-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(88px, 1fr)); gap: 8px; margin: 14px 0; }
.mini-metrics span { border: 1px solid var(--line); border-radius: 6px; padding: 8px; color: var(--muted); font-size: 12px; }
.mini-metrics strong { display: block; color: var(--ink); margin-top: 3px; font-size: 15px; }
.issues { margin: 0 0 12px; padding-left: 18px; color: #7f1d1d; }
.predictions { border-top: 1px solid var(--line); padding-top: 10px; }
.predictions summary { cursor: pointer; font-weight: 700; }
.prediction-row { margin-top: 10px; padding: 10px; border-radius: 6px; border: 1px solid var(--line); background: #fff; }
.prediction-row.bad { border-left: 4px solid var(--red); }
.prediction-row.ok { border-left: 4px solid var(--green); }
.prediction-row span { color: var(--muted); font-size: 12px; }
.prediction-row strong { display: block; margin-top: 3px; }
.prediction-row p { display: grid; grid-template-columns: 44px 1fr; gap: 8px; margin: 8px 0 0; color: var(--ink); }
.prediction-row b { color: var(--muted); }
.prediction-row em { display: block; margin-top: 8px; color: var(--muted); font-style: normal; font-size: 12px; }
@media (max-width: 1100px) {
  .metric-grid, .weak-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .image-grid { grid-template-columns: 1fr; }
}
@media (max-width: 760px) {
  .page { width: min(100% - 24px, 1440px); padding-top: 18px; }
  .hero { display: block; }
  .run-meta { margin-top: 16px; }
  .metric-grid, .issue-list, .weak-grid, .toolbar { grid-template-columns: 1fr; }
  .image-card-header { grid-template-columns: 1fr; }
  .image-card img { width: 100%; max-height: 280px; }
  .mini-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""


def script() -> str:
    return """
const searchInput = document.getElementById('searchInput');
const severityFilter = document.getElementById('severityFilter');
const objectFilter = document.getElementById('objectFilter');
const visibleCount = document.getElementById('visibleCount');
const cards = Array.from(document.querySelectorAll('.image-card'));

function applyFilters() {
  const query = searchInput.value.trim().toLowerCase();
  const severity = severityFilter.value;
  const objectName = objectFilter.value;
  let shown = 0;
  cards.forEach((card) => {
    const matchesQuery = !query || card.dataset.search.includes(query);
    const matchesSeverity = severity === 'all' || card.dataset.severity === severity;
    const matchesObject = objectName === 'all' || card.dataset.object === objectName;
    const isVisible = matchesQuery && matchesSeverity && matchesObject;
    card.style.display = isVisible ? '' : 'none';
    shown += isVisible ? 1 : 0;
  });
  visibleCount.textContent = `${shown} / ${cards.length} images`;
}

[searchInput, severityFilter, objectFilter].forEach((node) => node.addEventListener('input', applyFilters));
"""


def describe_global_issues(
    *,
    summary: dict[str, Any],
    groups: list[DashboardImageGroup],
    coverage: float,
) -> list[str]:
    issues: list[str] = []
    answer_match_rate = overall_answer_match_rate(groups)
    if coverage < 1.0:
        issues.append("생성 평가는 전체 test sample 중 일부만 수행되어 이미지별 판단 범위가 제한된다.")
    if groups and answer_match_rate < 0.3:
        issues.append("의미 기준 answer match가 낮다. 단순 문장 일치보다 실제 답변 방향이 맞았는지 우선 확인해야 한다.")
    if float(summary.get("exact_match_rate") or 0.0) < 0.1:
        issues.append("Exact match가 매우 낮다. 현재 모델은 정답 문장 형식을 안정적으로 재현하지 못한다.")
    if float(summary.get("contains_answer_rate") or 0.0) < 0.5:
        issues.append("Contains answer 비율이 낮다. 생성 답변이 정답 표현을 충분히 포함하지 못한다.")
    if float(summary.get("avg_token_overlap") or 0.0) < 0.5:
        issues.append("Token overlap이 낮다. 정답 핵심 단어를 놓치는 샘플이 많다.")
    severe_count = sum(1 for group in groups if group.severity == "심각")
    if severe_count:
        issues.append(f"이미지 {severe_count}개가 심각 상태다. 생성 답변과 정답의 차이가 큰 이미지부터 확인해야 한다.")
    if groups and all(group.object_hit_rate is None for group in groups):
        issues.append("현재 평가 split에는 `metadata.object`가 없어 object hit 지표 대신 task/category와 선택지 정답률을 표시한다.")
    if not issues:
        issues.append("주요 지표상 큰 경고는 없지만, 생성 답변 원문 검토는 계속 필요하다.")
    return issues


def describe_group_issues(
    *,
    label: str,
    exact_rate: float,
    contains_rate: float,
    answer_match_rate: float,
    object_hit_rate: float | None,
    choice_letter_rate: float | None,
    assessment_warning_rate: float,
    avg_overlap: float,
) -> list[str]:
    issues: list[str] = []
    if answer_match_rate == 0:
        issues.append(f"`{label}` 의미 기준으로 맞은 답변이 없음")
    elif answer_match_rate < 0.5:
        issues.append(f"`{label}` 의미 기준으로 맞은 답변이 일부뿐임")
    if choice_letter_rate is not None and choice_letter_rate < 0.5:
        issues.append(f"`{label}` 선택지 letter 정답률이 낮음")
    if assessment_warning_rate > 0:
        issues.append("선택지 letter는 맞았지만 생성 설명이 정답 텍스트와 어긋나는 행이 있음")
    if object_hit_rate is not None and object_hit_rate < 0.5:
        issues.append(f"`{label}` 객체 단어를 생성 답변에 충분히 포함하지 못함")
    if contains_rate == 0 and answer_match_rate < 0.5:
        issues.append("정답 문장을 그대로 포함한 답변이 없음")
    if exact_rate == 0 and answer_match_rate < 0.5:
        issues.append("완전 일치 답변이 없음")
    if avg_overlap < 0.4 and answer_match_rate < 0.5:
        issues.append("정답 단어 overlap이 낮음")
    if not issues:
        issues.append("현재 샘플 기준 큰 문제 없음")
    return issues


def classify_severity(
    *,
    answer_match_rate: float,
    contains_rate: float,
    object_hit_rate: float | None,
    choice_letter_rate: float | None,
    assessment_warning_rate: float,
    avg_overlap: float,
) -> str:
    if answer_match_rate == 0 and avg_overlap < 0.35:
        return "심각"
    if choice_letter_rate is not None and choice_letter_rate < 0.4 and answer_match_rate < 0.4:
        return "심각"
    if object_hit_rate is not None and object_hit_rate < 0.4 and answer_match_rate < 0.4:
        return "심각"
    if answer_match_rate == 0:
        return "심각"
    if answer_match_rate < 0.5:
        return "주의"
    if assessment_warning_rate > 0:
        return "주의"
    if choice_letter_rate is not None and choice_letter_rate < 0.7:
        return "주의"
    if object_hit_rate is not None and object_hit_rate < 0.8:
        return "주의"
    return "양호"


def weakness_sort_key(group: DashboardImageGroup) -> tuple[int, float, float]:
    severity_rank = {"심각": 0, "주의": 1, "양호": 2}.get(group.severity, 3)
    return (severity_rank, primary_success_rate(group), group.avg_token_overlap)


def primary_success_rate(group: DashboardImageGroup) -> float:
    return group.answer_match_rate


def primary_metric_label(group: DashboardImageGroup) -> str:
    return "answer"


def prediction_success(row: dict[str, Any]) -> bool:
    if row.get("answer_match") is not None:
        return row.get("answer_match") is True
    assessed = assess_prediction_answer(row)
    return assessed.get("answer_match") is True


def overall_answer_match_rate(groups: list[DashboardImageGroup]) -> float:
    rows = [row for group in groups for row in group.predictions]
    if not rows:
        return 0.0
    return sum(1 for row in rows if prediction_success(row)) / len(rows)


def assess_prediction_answer(row: dict[str, Any]) -> dict[str, Any]:
    expected_answer = str(row.get("expected_answer") or "")
    generated_answer = str(row.get("generated_answer") or "")
    question = str(row.get("question") or "")
    task = str(row.get("task") or "")
    overlap = float(row.get("token_overlap") or token_overlap(expected_answer, generated_answer))
    choice_match = row.get("choice_letter_match")
    if choice_match is None:
        choice_match = choice_letter_match(expected_answer, generated_answer)
    yes_no = yes_no_answer_match(expected_answer=expected_answer, generated_answer=generated_answer, question=question)
    count = count_answer_match(
        expected_answer=expected_answer,
        generated_answer=generated_answer,
        question=question,
        task=task,
    )
    choice_warning = bool(choice_match is True and choice_explanation_warning(expected_answer, generated_answer))

    if bool(row.get("exact_match")):
        answer_match = True
        reason = "exact"
    elif bool(row.get("contains_answer")):
        answer_match = True
        reason = "contains"
    elif choice_match is not None:
        answer_match = choice_match is True
        reason = "choice-letter"
    elif count is not None:
        answer_match = count is True
        reason = "count"
    elif yes_no is not None:
        answer_match = yes_no is True
        reason = "yes-no"
    elif row.get("object_hit") is True:
        answer_match = True
        reason = "object"
    elif has_numeric_conflict(expected_answer, generated_answer):
        answer_match = False
        reason = "numeric-conflict"
    else:
        short_match = short_fact_answer_match(expected_answer, generated_answer, question)
        if short_match is not None:
            answer_match = short_match
            reason = "short-fact"
        elif overlap >= 0.72:
            answer_match = True
            reason = "token-overlap"
        else:
            answer_match = False
            reason = "low-overlap"

    return {
        "answer_match": answer_match,
        "answer_match_reason": reason,
        "choice_letter_match": choice_match,
        "choice_explanation_warning": choice_warning,
        "yes_no_match": yes_no,
        "count_match": count,
        "assessment_warning": choice_warning,
    }


def yes_no_answer_match(*, expected_answer: str, generated_answer: str, question: str) -> bool | None:
    if not is_yes_no_question(question):
        return None
    expected_polarity = detect_answer_polarity(expected_answer)
    generated_polarity = detect_answer_polarity(generated_answer)
    if expected_polarity is None and generated_polarity is not None:
        expected_polarity = infer_descriptive_yes_no_polarity(expected_answer)
    if expected_polarity is None or generated_polarity is None:
        return None
    return expected_polarity == generated_polarity


def infer_descriptive_yes_no_polarity(answer: str) -> bool | None:
    """yes/no 질문의 서술형 정답이 긍정을 암시하는지 보수적으로 판정한다.

    의도: 일부 데이터는 "Yes" 대신 "The bell peppers appear..."처럼 대상 설명만 제공한다.
    참고: LVIS-Instruct-4V 변환 데이터의 yes/no 질문 패턴.
    선택 이유: 생성 답변이 "yes"처럼 짧을 때 정답 서술이 명시적 부정이 아니면 실제 의미상 긍정인 경우가 많다.
    """

    normalized = normalize_semantic_text(answer)
    if not normalized:
        return None
    if detect_answer_polarity(answer) is False:
        return False
    negative_markers = (" no ", " not ", " without ", " none ", " nothing ")
    padded = f" {normalized} "
    if any(marker in padded for marker in negative_markers):
        return False
    return True


def is_yes_no_question(question: str) -> bool:
    normalized = normalize_semantic_text(question)
    normalized = re.sub(r"^(based on the image|look at the image and answer|in the image)\s+", "", normalized)
    non_binary_prefixes = ("can you describe", "can you count", "can you identify")
    if normalized.startswith(non_binary_prefixes):
        return False
    if question_contains_answer_choices(normalized) and not normalized.startswith(
        ("are there", "is there", "are any", "is any", "do you see", "can you see", "could you see")
    ):
        return False
    return normalized.startswith(
        (
            "are there",
            "is there",
            "are the",
            "is the",
            "are any",
            "is any",
            "do you see",
            "does the",
            "do the",
            "did the",
            "has the",
            "have the",
            "was the",
            "were the",
            "can you see",
            "could you see",
            "would you say",
        )
    )


def detect_answer_polarity(answer: str) -> bool | None:
    normalized = normalize_semantic_text(answer)
    tokens = normalized.split()
    if not tokens:
        return None
    if tokens[0] == "yes":
        return True
    if tokens[0] == "no":
        return False
    negative_patterns = (
        r"\bnot exactly\b",
        r"\bthere (?:is|are|was|were) not\b",
        r"\bthere (?:is|are|was|were) no\b",
        r"\bnot (?:any|visible|present)\b",
        r"\bno (?:visible|people|humans|animals|water|source|pond|door|doors)\b",
        r"\bwithout\b",
        r"\bnothing\b",
        r"\bnone\b",
    )
    if any(re.search(pattern, normalized) for pattern in negative_patterns):
        return False
    positive_patterns = (
        r"\bthere (?:is|are|was|were)\b",
        r"\bvisible\b",
        r"\bpresent\b",
    )
    if any(re.search(pattern, normalized) for pattern in positive_patterns):
        return True
    return None


def count_answer_match(*, expected_answer: str, generated_answer: str, question: str, task: str) -> bool | None:
    if "count" not in normalize_semantic_text(task) and not is_count_question(question):
        return None
    expected_numbers = extract_numbers(expected_answer)
    generated_numbers = extract_numbers(generated_answer)
    if not expected_numbers:
        return None
    if not generated_numbers:
        return False
    return bool(set(expected_numbers) & set(generated_numbers))


def is_count_question(question: str) -> bool:
    normalized = normalize_semantic_text(question)
    return "how many" in normalized or "number of" in normalized or "can you count" in normalized


def has_numeric_conflict(expected_answer: str, generated_answer: str) -> bool:
    expected_numbers = extract_numbers(expected_answer)
    generated_numbers = extract_numbers(generated_answer)
    return bool(expected_numbers and generated_numbers and not set(expected_numbers) & set(generated_numbers))


def extract_numbers(text: str) -> list[int]:
    normalized = normalize_semantic_text(text)
    numbers = [int(match.group(0)) for match in re.finditer(r"\b\d+\b", normalized)]
    for token in normalized.split():
        if token in NUMBER_WORDS:
            numbers.append(NUMBER_WORDS[token])
    return numbers


def short_fact_answer_match(expected_answer: str, generated_answer: str, question: str = "") -> bool | None:
    expected_tokens = canonical_informative_tokens(expected_answer)
    if not expected_tokens:
        return None
    generated_tokens_list = canonical_informative_tokens(generated_answer)
    generated_tokens = set(generated_tokens_list)
    question_tokens = set(canonical_informative_tokens(question))
    if short_generated_fact_is_supported(
        expected_tokens=expected_tokens,
        generated_tokens=generated_tokens,
        generated_tokens_list=generated_tokens_list,
        question_tokens=question_tokens,
        allow_question_token_answers=question_contains_answer_choices(question),
    ):
        return True
    if len(expected_tokens) > 8:
        return None
    critical_tokens = [
        token
        for token in expected_tokens
        if token not in GENERIC_VISUAL_TOKENS and token not in question_tokens
    ]
    if not critical_tokens:
        return None
    hit_count = sum(1 for token in critical_tokens if token in generated_tokens)
    if len(critical_tokens) <= 2:
        return hit_count == len(critical_tokens)
    return (hit_count / len(critical_tokens)) >= 0.5


def short_generated_fact_is_supported(
    *,
    expected_tokens: list[str],
    generated_tokens: set[str],
    generated_tokens_list: list[str],
    question_tokens: set[str],
    allow_question_token_answers: bool = False,
) -> bool:
    """짧은 생성 단답이 정답 문장의 핵심 일부와 일치하면 통과시킨다.

    의도: "bathroom" vs "bathroom interior", "in air" vs 긴 위치 설명처럼 단답이 더 정확한 경우가 있다.
    선택 이유: 모든 정답 토큰을 요구하면 긴 설명형 정답에서 실제 정답인 짧은 명사구가 과도하게 실패한다.
    """

    if not generated_tokens or len(generated_tokens_list) > 4:
        return False
    expected_token_set = set(expected_tokens)
    if not generated_tokens <= expected_token_set:
        return False
    content_tokens = [
        token
        for token in generated_tokens
        if token not in GENERIC_VISUAL_TOKENS
        and (allow_question_token_answers or token not in question_tokens)
    ]
    return bool(content_tokens)


def question_contains_answer_choices(question: str) -> bool:
    normalized = normalize_semantic_text(question)
    return " or " in f" {normalized} "


def choice_explanation_warning(expected_answer: str, generated_answer: str) -> bool:
    expected_text = expected_choice_text(expected_answer)
    if not expected_text:
        return False
    generated_tokens = informative_tokens(generated_answer)
    if len(generated_tokens) <= 1:
        return False
    expected_tokens = set(informative_tokens(expected_text))
    if not expected_tokens:
        return False
    if expected_tokens & set(generated_tokens):
        return False
    return token_overlap(expected_text, generated_answer) < 0.5


def expected_choice_text(expected_answer: str) -> str:
    return re.sub(r"^\s*[A-Da-d]\s*[\.\)]\s*", "", str(expected_answer or "")).strip()


def token_overlap(expected_answer: str, generated_answer: str) -> float:
    expected_tokens = set(normalize_search_text(expected_answer).split())
    generated_tokens = set(normalize_search_text(generated_answer).split())
    if not expected_tokens:
        return 0.0
    return len(expected_tokens & generated_tokens) / len(expected_tokens)


def informative_tokens(text: str) -> list[str]:
    return [
        token
        for token in normalize_search_text(text).split()
        if len(token) > 1 and token not in STOP_WORDS
    ]


def canonical_informative_tokens(text: str) -> list[str]:
    return [CANONICAL_TOKEN_MAP.get(token, token) for token in informative_tokens(text)]


def normalize_semantic_text(value: str) -> str:
    lowered = str(value or "").lower()
    lowered = re.sub(r"n['’]t\b", " not", lowered)
    lowered = re.sub(r"[^0-9a-z가-힣]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def coverage_rate(summary: dict[str, Any]) -> float:
    sample_count = int(summary.get("sample_count") or 0)
    generation_count = int(summary.get("generation_sample_count") or 0)
    if sample_count <= 0:
        return 0.0
    return generation_count / sample_count


def unique_groups_by_object(groups: list[DashboardImageGroup]) -> list[DashboardImageGroup]:
    seen: set[str] = set()
    unique: list[DashboardImageGroup] = []
    for group in sorted(groups, key=lambda item: item.object_label):
        if group.object_label in seen:
            continue
        seen.add(group.object_label)
        unique.append(group)
    return unique


def search_blob_for_group(group: DashboardImageGroup) -> str:
    parts = [group.image, group.object_label, group.label_kind, group.category, group.file_title, group.severity]
    for prediction in group.predictions:
        parts.extend(
            [
                str(prediction.get("question") or ""),
                str(prediction.get("expected_answer") or ""),
                str(prediction.get("generated_answer") or ""),
            ]
        )
    return normalize_search_text(" ".join(parts))


def object_label_hits_generation(object_label: str, generated_answer: str) -> bool:
    normalized_object = normalize_search_text(object_label)
    normalized_generated = normalize_search_text(generated_answer)
    return bool(normalized_object and normalized_object in normalized_generated)


def choice_letter_match(expected_answer: str, generated_answer: str) -> bool | None:
    expected_letter = extract_expected_choice_letter(expected_answer)
    if expected_letter is None:
        return None
    generated_letter = extract_choice_letter(generated_answer)
    if generated_letter is None:
        return False
    return expected_letter == generated_letter


def extract_expected_choice_letter(value: str) -> str | None:
    text = str(value or "").strip()
    expected_style = re.match(r"^\s*([A-Da-d])\s*[\.\)]", text)
    if expected_style:
        return expected_style.group(1).upper()
    return None


def extract_choice_letter(value: str) -> str | None:
    text = str(value or "").strip()
    expected_style = re.match(r"^\s*([A-Da-d])\s*[\.\)]", text)
    if expected_style:
        return expected_style.group(1).upper()
    boxed = re.search(r"\\boxed\{?\s*([A-Da-d])\s*\}?", text)
    if boxed:
        return boxed.group(1).upper()
    standalone = re.search(r"\b([A-Da-d])\b", text)
    if standalone:
        return standalone.group(1).upper()
    return None


def normalize_search_text(value: str) -> str:
    lowered = str(value).lower()
    lowered = re.sub(r"[^0-9a-z가-힣]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def first_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            return metadata
    return {}


def first_row_value(rows: list[dict[str, Any]], key: str) -> str:
    for row in rows:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def resolve_image_path(*, image_root: Path, image: str) -> Path:
    image_path = Path(image)
    if image_path.is_absolute():
        return image_path
    return image_root / image_path


def build_image_uri(
    *,
    image_path: Path,
    image: str,
    html_dir: Path | None,
    asset_dir: Path | None,
) -> str:
    """대시보드 HTML에서 안정적으로 열 수 있는 이미지 경로를 만든다.

    의도: `file://` 절대경로는 브라우저/IDE preview 환경에 따라 차단될 수 있다. 그래서 실제 대시보드 생성
    시점에는 이미지를 대시보드 옆 asset 폴더로 복사하고 상대경로로 참조한다.
    """

    if not image_path.exists():
        return ""
    if html_dir is None or asset_dir is None:
        return image_path.resolve().as_uri()

    asset_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(image).encode("utf-8")).hexdigest()[:10]
    object_slug = slugify(Path(image).parent.name or "image")
    suffix = image_path.suffix.lower() or ".img"
    copied_path = asset_dir / f"{object_slug}-{digest}{suffix}"
    if not copied_path.exists():
        shutil.copy2(image_path, copied_path)
    return relative_uri(from_dir=html_dir, to_path=copied_path)


def relative_uri(*, from_dir: Path, to_path: Path) -> str:
    return Path(os.path.relpath(to_path, start=from_dir)).as_posix()


def slugify(value: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z가-힣]+", "-", value.strip()).strip("-").lower()
    return normalized or "image"


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if bool(row.get(key))) / len(rows)


def mean_optional_bool(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(1 for value in values if bool(value)) / len(values)


def mean_float(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key) or 0.0) for row in rows]
    if not values:
        return 0.0
    return sum(values) / len(values)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 객체가 필요합니다: {path}")
    return payload


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


def format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "N/A"


def format_percent(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.1f}%"
    return "N/A"


def format_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def escape(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def escape_attr(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)
