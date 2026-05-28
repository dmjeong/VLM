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


@dataclass(frozen=True)
class DashboardImageGroup:
    """대시보드의 이미지 단위 평가 묶음."""

    image: str
    image_uri: str
    object_label: str
    category: str
    file_title: str
    source_url: str
    sample_count: int
    exact_rate: float
    contains_rate: float
    object_hit_rate: float
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
        object_label = str(metadata.get("object") or "unknown")
        category = str(metadata.get("category") or "unknown")
        file_title = str(metadata.get("file_title") or image)
        source_url = str(metadata.get("description_url") or "")
        exact_rate = mean_bool(rows, "exact_match")
        contains_rate = mean_bool(rows, "contains_answer")
        object_hit_rate = mean_bool(rows, "object_hit")
        avg_overlap = mean_float(rows, "token_overlap")
        issues = describe_group_issues(
            object_label=object_label,
            exact_rate=exact_rate,
            contains_rate=contains_rate,
            object_hit_rate=object_hit_rate,
            avg_overlap=avg_overlap,
        )
        image_path = resolve_image_path(image_root=image_root, image=image)
        image_groups.append(
            DashboardImageGroup(
                image=image,
                image_uri=build_image_uri(image_path=image_path, image=image, html_dir=html_dir, asset_dir=asset_dir),
                object_label=object_label,
                category=category,
                file_title=file_title,
                source_url=source_url,
                sample_count=len(rows),
                exact_rate=exact_rate,
                contains_rate=contains_rate,
                object_hit_rate=object_hit_rate,
                avg_token_overlap=avg_overlap,
                severity=classify_severity(contains_rate=contains_rate, object_hit_rate=object_hit_rate, avg_overlap=avg_overlap),
                issues=issues,
                predictions=rows,
            )
        )
    return image_groups


def merge_prediction_with_sample(prediction: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    object_label = str(metadata.get("object") or "")
    generated_answer = str(prediction.get("generated_answer") or "")
    merged = {
        **prediction,
        "task": sample.get("task") or "",
        "metadata": metadata,
        "object_hit": object_label_hits_generation(object_label, generated_answer),
    }
    if not merged.get("image") and sample.get("image"):
        merged["image"] = sample.get("image")
    if not merged.get("question") and sample.get("question"):
        merged["question"] = sample.get("question")
    if not merged.get("expected_answer") and sample.get("answer"):
        merged["expected_answer"] = sample.get("answer")
    return merged


def render_dashboard(
    *,
    summary: dict[str, Any],
    groups: list[DashboardImageGroup],
    predictions: list[dict[str, Any]],
) -> str:
    coverage = coverage_rate(summary)
    issues = describe_global_issues(summary=summary, groups=groups, coverage=coverage)
    worst_groups = sorted(groups, key=lambda group: (group.object_hit_rate, group.avg_token_overlap))[:5]
    generated_count = int(summary.get("generation_sample_count") or len(predictions))

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
        "<h2>가장 취약한 이미지</h2>",
        '<div class="weak-grid">',
        "".join(render_weak_group(group) for group in worst_groups),
        "</div>",
        "</section>",
        '<section class="toolbar" aria-label="대시보드 필터">',
        '<input id="searchInput" type="search" placeholder="이미지, 객체, 질문, 답변 검색">',
        '<select id="severityFilter"><option value="all">전체 상태</option><option value="심각">심각</option><option value="주의">주의</option><option value="양호">양호</option></select>',
        '<select id="objectFilter"><option value="all">전체 객체</option>',
        "".join(f'<option value="{escape(group.object_label)}">{escape(group.object_label)}</option>' for group in unique_groups_by_object(groups)),
        "</select>",
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
        f"<p>object hit {format_percent(group.object_hit_rate)} · overlap {format_percent(group.avg_token_overlap)}</p>"
        "</article>"
    )


def render_image_group(group: DashboardImageGroup) -> str:
    issue_html = "".join(f"<li>{escape(issue)}</li>" for issue in group.issues)
    prediction_rows = "".join(render_prediction_row(row) for row in group.predictions)
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
        f"<p>{escape(group.file_title)}</p>"
        f"{source_link}"
        "</div>"
        "</div>"
        '<div class="mini-metrics">'
        f"<span>samples <strong>{group.sample_count}</strong></span>"
        f"<span>exact <strong>{format_percent(group.exact_rate)}</strong></span>"
        f"<span>contains <strong>{format_percent(group.contains_rate)}</strong></span>"
        f"<span>object hit <strong>{format_percent(group.object_hit_rate)}</strong></span>"
        f"<span>overlap <strong>{format_percent(group.avg_token_overlap)}</strong></span>"
        "</div>"
        f'<ul class="issues">{issue_html}</ul>'
        '<details class="predictions" open>'
        "<summary>질문별 결과</summary>"
        f"{prediction_rows}"
        "</details>"
        "</article>"
    )


def render_prediction_row(row: dict[str, Any]) -> str:
    state = "ok" if row.get("object_hit") else "bad"
    return (
        f'<div class="prediction-row {state}">'
        "<div>"
        f"<span>{escape(row.get('task') or row.get('sample_id'))}</span>"
        f"<strong>{escape(row.get('question'))}</strong>"
        "</div>"
        f"<p><b>정답</b>{escape(row.get('expected_answer'))}</p>"
        f"<p><b>생성</b>{escape(row.get('generated_answer'))}</p>"
        f"<em>exact {format_bool(row.get('exact_match'))} · contains {format_bool(row.get('contains_answer'))} · "
        f"object {format_bool(row.get('object_hit'))} · overlap {format_percent(row.get('token_overlap'))}</em>"
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
.metric-grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }
.metric-card, .panel, .image-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
.metric-card { padding: 14px; min-height: 112px; }
.metric-card span { display: block; color: var(--muted); font-size: 13px; }
.metric-card strong { display: block; margin-top: 10px; font-size: 28px; letter-spacing: 0; }
.metric-card p { margin: 8px 0 0; color: var(--muted); font-size: 13px; }
.panel { padding: 18px; margin: 16px 0; }
.panel h2 { margin: 0 0 12px; font-size: 18px; }
.issue-list { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.issue { border-left: 4px solid var(--red); background: #fff5f5; padding: 10px 12px; border-radius: 6px; color: #7f1d1d; }
.weak-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; }
.weak-card { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fff; }
.weak-card strong, .weak-card span { display: block; overflow-wrap: anywhere; }
.weak-card span { margin-top: 5px; color: var(--muted); font-size: 12px; }
.weak-card p { margin: 8px 0 0; font-size: 12px; }
.toolbar { position: sticky; top: 0; z-index: 3; display: grid; grid-template-columns: 1fr 170px 180px; gap: 10px; padding: 12px 0; background: rgba(247, 248, 250, 0.95); backdrop-filter: blur(8px); }
input, select { width: 100%; border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; font: inherit; background: #fff; color: var(--ink); }
.image-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.image-card { padding: 14px; min-width: 0; }
.image-card-header { display: grid; grid-template-columns: 168px 1fr; gap: 14px; align-items: start; }
.image-card img { width: 168px; aspect-ratio: 4 / 3; object-fit: cover; border-radius: 6px; border: 1px solid var(--line); background: #eef2f7; }
.image-card h3 { margin: 8px 0 4px; font-size: 22px; }
.image-card p { margin: 0; color: var(--muted); overflow-wrap: anywhere; }
.image-card a { display: inline-block; margin-top: 8px; color: var(--blue); text-decoration: none; font-size: 13px; }
.badge { display: inline-flex; align-items: center; min-height: 24px; padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }
.severity-심각 { border-color: #fecaca; background: #fef2f2; color: var(--red); }
.severity-주의 { border-color: #fed7aa; background: #fff7ed; color: var(--amber); }
.severity-양호 { border-color: #bbf7d0; background: #f0fdf4; color: var(--green); }
.mini-metrics { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin: 14px 0; }
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
const cards = Array.from(document.querySelectorAll('.image-card'));

function applyFilters() {
  const query = searchInput.value.trim().toLowerCase();
  const severity = severityFilter.value;
  const objectName = objectFilter.value;
  cards.forEach((card) => {
    const matchesQuery = !query || card.dataset.search.includes(query);
    const matchesSeverity = severity === 'all' || card.dataset.severity === severity;
    const matchesObject = objectName === 'all' || card.dataset.object === objectName;
    card.style.display = matchesQuery && matchesSeverity && matchesObject ? '' : 'none';
  });
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
    if coverage < 1.0:
        issues.append("생성 평가는 전체 test sample 중 일부만 수행되어 이미지별 판단 범위가 제한된다.")
    if float(summary.get("exact_match_rate") or 0.0) < 0.1:
        issues.append("Exact match가 매우 낮다. 현재 모델은 정답 문장 형식을 안정적으로 재현하지 못한다.")
    if float(summary.get("contains_answer_rate") or 0.0) < 0.5:
        issues.append("Contains answer 비율이 낮다. 생성 답변이 정답 표현을 충분히 포함하지 못한다.")
    if float(summary.get("avg_token_overlap") or 0.0) < 0.5:
        issues.append("Token overlap이 낮다. 정답 핵심 단어를 놓치는 샘플이 많다.")
    severe_count = sum(1 for group in groups if group.severity == "심각")
    if severe_count:
        issues.append(f"이미지 {severe_count}개가 심각 상태다. 객체명 자체를 못 맞히는 이미지부터 데이터/라벨을 확인해야 한다.")
    if not issues:
        issues.append("주요 지표상 큰 경고는 없지만, 생성 답변 원문 검토는 계속 필요하다.")
    return issues


def describe_group_issues(
    *,
    object_label: str,
    exact_rate: float,
    contains_rate: float,
    object_hit_rate: float,
    avg_overlap: float,
) -> list[str]:
    issues: list[str] = []
    if object_hit_rate < 0.5:
        issues.append(f"`{object_label}` 객체 단어를 생성 답변에 충분히 포함하지 못함")
    if contains_rate == 0:
        issues.append("정답 문장을 그대로 포함한 답변이 없음")
    if exact_rate == 0:
        issues.append("완전 일치 답변이 없음")
    if avg_overlap < 0.4:
        issues.append("정답 단어 overlap이 낮음")
    if not issues:
        issues.append("현재 샘플 기준 큰 문제 없음")
    return issues


def classify_severity(*, contains_rate: float, object_hit_rate: float, avg_overlap: float) -> str:
    if object_hit_rate < 0.4 or (contains_rate == 0 and avg_overlap < 0.35):
        return "심각"
    if object_hit_rate < 0.8 or contains_rate < 0.3 or avg_overlap < 0.6:
        return "주의"
    return "양호"


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
    parts = [group.image, group.object_label, group.category, group.file_title, group.severity]
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
