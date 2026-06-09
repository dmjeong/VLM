from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

from mini_vlm.evaluation.benchmark import read_jsonl, write_json
from mini_vlm.evaluation.dashboard import extract_numbers, is_count_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval 데이터셋 라벨 품질 감사")
    parser.add_argument("--dataset", default="data/eval80/test.jsonl")
    parser.add_argument("--output", default="data/eval80/quality_report.json")
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    findings = [finding for row in rows for finding in audit_sample(row)]
    report = {
        "dataset": args.dataset,
        "sample_count": len(rows),
        "finding_count": len(findings),
        "by_reason": Counter(finding["reason"] for finding in findings),
        "findings": findings,
    }
    write_json(args.output, report)
    print(f"quality findings: {len(findings)}")
    print(f"report: {args.output}")


def audit_sample(sample: dict[str, Any]) -> list[dict[str, Any]]:
    answer = str(sample.get("answer") or "").strip()
    question = str(sample.get("question") or "").strip()
    task = str(sample.get("task") or "")
    findings: list[dict[str, Any]] = []
    if answer.endswith(":"):
        findings.append(make_finding(sample, "incomplete-answer", "정답이 콜론으로 끝나 실제 답변 내용이 누락된 것으로 보임"))
    if task == "counting" or is_count_question(question):
        if not extract_numbers(answer):
            findings.append(make_finding(sample, "count-answer-without-number", "counting 질문인데 정답에 숫자가 없음"))
        elif contains_vague_count(answer):
            findings.append(make_finding(sample, "vague-count-answer", "counting 정답에 few/multiple 같은 모호한 수량 표현이 섞여 있음"))
    if len(answer.split()) <= 2 and task not in {"multiple-choice-vqa", "counting"}:
        findings.append(make_finding(sample, "very-short-answer", "서술형 VQA 정답이 매우 짧아 의미 채점 안정성이 낮을 수 있음"))
    return findings


def contains_vague_count(answer: str) -> bool:
    normalized = re.sub(r"[^0-9a-z]+", " ", answer.lower())
    return bool(re.search(r"\b(few|several|multiple|many|some)\b", normalized))


def make_finding(sample: dict[str, Any], reason: str, message: str) -> dict[str, Any]:
    return {
        "sample_id": sample.get("sample_id"),
        "task": sample.get("task"),
        "image": sample.get("image"),
        "question": sample.get("question"),
        "answer": sample.get("answer"),
        "reason": reason,
        "message": message,
    }


if __name__ == "__main__":
    main()
