from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_DASHBOARDS = (
    Path("artifacts/dinov3-mini-vlm/vision-ablation/dashboard.html"),
    Path("artifacts/dinov3-mini-vlm/vision-ablation/experiments/dinov3-vit-s-16-qwen-lora-stage2-1-epoch/dashboard.html"),
)
DEFAULT_OUTPUT_DIR = Path("artifacts/dinov3-mini-vlm/vision-ablation/render-check")
VIEWPORTS = {
    "desktop": {"width": 1440, "height": 1000},
    "mobile": {"width": 390, "height": 844},
}


@dataclass(frozen=True)
class ViewportCheck:
    dashboard: str
    viewport: str
    screenshot: str
    title: str
    h1: str
    image_count: int
    broken_image_count: int
    card_count: int
    visible_count: str
    console_errors: list[str]
    page_errors: list[str]
    horizontal_overflow_px: int
    assertions: dict[str, bool]

    @property
    def passed(self) -> bool:
        return (
            all(self.assertions.values())
            and not self.console_errors
            and not self.page_errors
            and self.broken_image_count == 0
            and self.horizontal_overflow_px <= 4
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="HTML 평가 대시보드를 Playwright로 렌더링 검증")
    parser.add_argument(
        "--dashboard",
        action="append",
        default=[],
        help="검증할 dashboard.html 경로. 여러 번 지정할 수 있습니다.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    dashboards = [Path(value) for value in args.dashboard] or list(DEFAULT_DASHBOARDS)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = verify_dashboards(dashboards=dashboards, output_dir=output_dir)
    report_path = output_dir / "render_check.json"
    report_path.write_text(
        json.dumps(
            {
                "passed": all(result.passed for result in results),
                "result_count": len(results),
                "results": [asdict(result) | {"passed": result.passed} for result in results],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status} {result.viewport} {result.dashboard} "
            f"cards={result.card_count} images={result.image_count} broken={result.broken_image_count} "
            f"screenshot={result.screenshot}"
        )
    print(f"report: {report_path}")
    if not all(result.passed for result in results):
        raise SystemExit(1)


def verify_dashboards(*, dashboards: list[Path], output_dir: Path) -> list[ViewportCheck]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit("Playwright가 필요합니다. `.venv/bin/python -m pip install '.[qa]'` 후 실행하세요.") from exc

    results: list[ViewportCheck] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for dashboard in dashboards:
                if not dashboard.exists():
                    raise FileNotFoundError(f"대시보드 파일이 없습니다: {dashboard}")
                for viewport_name, viewport in VIEWPORTS.items():
                    results.append(
                        inspect_dashboard(
                            browser=browser,
                            dashboard=dashboard,
                            viewport_name=viewport_name,
                            viewport=viewport,
                            output_dir=output_dir,
                        )
                    )
        finally:
            browser.close()
    return results


def inspect_dashboard(
    *,
    browser: Any,
    dashboard: Path,
    viewport_name: str,
    viewport: dict[str, int],
    output_dir: Path,
) -> ViewportCheck:
    console_errors: list[str] = []
    page_errors: list[str] = []
    page = browser.new_page(viewport=viewport)
    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.goto(dashboard.resolve().as_uri(), wait_until="networkidle")

    title = page.title()
    h1 = text_or_empty(page, "h1")
    image_count = page.locator("img").count()
    card_count = page.locator(".image-card").count()
    visible_count = text_or_empty(page, "#visibleCount")
    broken_images = page.evaluate(
        """
        () => Array.from(document.images)
          .filter((image) => !image.complete || image.naturalWidth === 0)
          .map((image) => image.getAttribute('src') || image.src)
        """
    )
    horizontal_overflow_px = int(
        page.evaluate(
            "() => Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth)"
        )
    )
    assertions = {
        "has_title": bool(title.strip()),
        "has_h1": bool(h1.strip()),
        "has_expected_dashboard_content": has_expected_dashboard_content(page),
        "links_resolve": dashboard_links_resolve(page=page, dashboard=dashboard),
        "filter_works": filter_works(page) if card_count else True,
        "answer_match_visible": answer_match_visible(page) if card_count else True,
    }
    screenshot_path = output_dir / f"{slugify(dashboard)}-{viewport_name}.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    page.close()
    return ViewportCheck(
        dashboard=str(dashboard),
        viewport=viewport_name,
        screenshot=str(screenshot_path),
        title=title,
        h1=h1,
        image_count=image_count,
        broken_image_count=len(broken_images),
        card_count=card_count,
        visible_count=visible_count,
        console_errors=console_errors,
        page_errors=page_errors,
        horizontal_overflow_px=horizontal_overflow_px,
        assertions=assertions,
    )


def has_expected_dashboard_content(page: Any) -> bool:
    body = page.locator("body").inner_text(timeout=5000)
    if "Vision Encoder Ablation 비교" in body:
        return page.locator("table tbody tr").count() > 0 and page.locator("a", has_text="열기").count() > 0
    if "이미지별 평가 대시보드" in body:
        return page.locator(".image-card").count() > 0 and "Answer match" in body
    return False


def dashboard_links_resolve(*, page: Any, dashboard: Path) -> bool:
    hrefs = page.locator("a").evaluate_all(
        "(links) => links.map((link) => link.getAttribute('href')).filter(Boolean)"
    )
    local_links = [href for href in hrefs if not str(href).startswith(("http://", "https://", "#"))]
    for href in local_links:
        if href == "comparison.md":
            continue
        target = (dashboard.parent / str(href)).resolve()
        if not target.exists():
            return False
    return True


def filter_works(page: Any) -> bool:
    search = page.locator("#searchInput")
    if search.count() == 0:
        return True
    total_cards = page.locator(".image-card").count()
    if total_cards == 0:
        return False
    search.fill("humans or animals")
    page.wait_for_timeout(100)
    visible_cards = page.locator(".image-card").evaluate_all(
        "(cards) => cards.filter((card) => getComputedStyle(card).display !== 'none').length"
    )
    search.fill("")
    return 0 < int(visible_cards) < total_cards


def answer_match_visible(page: Any) -> bool:
    body = page.locator("body").evaluate("(body) => body.textContent || ''", timeout=5000)
    return "Answer match" in body and "answer yes" in body and "reason yes-no" in body


def text_or_empty(page: Any, selector: str) -> str:
    locator = page.locator(selector)
    if locator.count() == 0:
        return ""
    return locator.first.inner_text(timeout=5000).strip()


def slugify(path: Path) -> str:
    parts = [part for part in path.with_suffix("").parts[-4:] if part not in {"artifacts", "dinov3-mini-vlm"}]
    return re.sub(r"[^0-9A-Za-z가-힣._-]+", "-", "-".join(parts)).strip("-").lower()


if __name__ == "__main__":
    main()
