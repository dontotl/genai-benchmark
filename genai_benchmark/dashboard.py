from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an HTML dashboard from benchmark JSON reports.")
    parser.add_argument(
        "--runs-dir",
        default=str(RUNS_DIR),
        help="Directory containing benchmark JSON reports.",
    )
    parser.add_argument(
        "--output",
        default=str(RUNS_DIR / "dashboard.html"),
        help="Output HTML file path.",
    )
    return parser.parse_args()


def load_reports(runs_dir: Path) -> list[dict]:
    reports = []
    for path in sorted(runs_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "summary" not in data or "results" not in data:
            continue
        reports.append(
            {
                "path": path,
                "name": path.stem,
                "summary": data.get("summary", []),
                "results": data.get("results", []),
                "selected_models": data.get("selected_models", []),
                "skipped": data.get("skipped", []),
                "benchmark_config": data.get("benchmark_config", {}),
                "generated_at": data.get("generated_at", ""),
            }
        )
    return reports


def clean_markdown_cell(value: str) -> str:
    value = value.strip()
    if value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    link_match = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", value)
    if link_match:
        return link_match.group(2)
    return value


def parse_seconds(value: str) -> float | None:
    value = clean_markdown_cell(value)
    if value == "-":
        return None
    if value.endswith("s"):
        value = value[:-1]
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str) -> int:
    return int(clean_markdown_cell(value))


def parse_table_row(line: str) -> list[str]:
    return [clean_markdown_cell(cell) for cell in line.strip().strip("|").split("|")]


def parse_suite_summary(path: Path) -> dict | None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("# ") or not lines[0].endswith(" Suite Summary"):
        return None

    name = lines[0].removeprefix("# ").removesuffix(" Suite Summary").strip()
    summary = {
        "path": path,
        "name": name,
        "generated_at": "",
        "target_regions": [],
        "attempts": 0,
        "successes": 0,
        "failures": 0,
        "sources": [],
        "target_latency": [],
    }

    section = ""
    for line in lines[1:]:
        if line.startswith("- Generated At:"):
            summary["generated_at"] = clean_markdown_cell(line.split(":", 1)[1].strip())
        elif line.startswith("- Target Regions:"):
            summary["target_regions"] = re.findall(r"`([^`]+)`", line)
        elif line.startswith("- Attempts:"):
            summary["attempts"] = parse_int(line.split(":", 1)[1].strip())
        elif line.startswith("- Successes:"):
            summary["successes"] = parse_int(line.split(":", 1)[1].strip())
        elif line.startswith("- Failures:"):
            summary["failures"] = parse_int(line.split(":", 1)[1].strip())
        elif line == "## Source Runner Summary":
            section = "sources"
        elif line == "## Target Region Average Latency":
            section = "target_latency"
        elif line.startswith("|") and "---" not in line and not line.startswith("| Source |"):
            cells = parse_table_row(line)
            if section == "sources" and len(cells) >= 8:
                summary["sources"].append(
                    {
                        "source": cells[0],
                        "status": cells[1],
                        "attempts": parse_int(cells[2]),
                        "successes": parse_int(cells[3]),
                        "failures": parse_int(cells[4]),
                        "avg_latency": parse_seconds(cells[5]),
                        "json": cells[6],
                        "markdown": cells[7],
                    }
                )
            elif section == "target_latency" and len(cells) >= 3:
                summary["target_latency"].append(
                    {
                        "source": cells[0],
                        "target_region": cells[1],
                        "avg_latency": parse_seconds(cells[2]),
                    }
                )

    if not summary["sources"] or not summary["target_latency"]:
        return None
    return summary


def load_suite_summaries(runs_dir: Path) -> list[dict]:
    summaries = []
    for path in sorted(runs_dir.glob("*-summary.md")):
        summary = parse_suite_summary(path)
        if summary:
            summaries.append(summary)
    return summaries


def detect_timestamp(results: list[dict]) -> str:
    if not results:
        return ""
    return max(result.get("iteration", 0) for result in results).__str__()


def aggregate_family_metrics(reports: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str], dict] = {}
    for report in reports:
        for item in report["summary"]:
            key = (report["name"], item["family"])
            bucket = buckets.setdefault(
                key,
                {
                    "report": report["name"],
                    "family": item["family"],
                    "attempts": 0,
                    "successes": 0,
                    "failures": 0,
                    "latencies": [],
                    "tokens_per_second": [],
                },
            )
            bucket["attempts"] += item["attempts"]
            bucket["successes"] += item["successes"]
            bucket["failures"] += item["failures"]
            if item["avg_latency_seconds"] is not None:
                bucket["latencies"].append(item["avg_latency_seconds"])
            if item.get("avg_output_tokens_per_second") is not None:
                bucket["tokens_per_second"].append(item["avg_output_tokens_per_second"])

    rows = []
    for (_, _), bucket in sorted(buckets.items()):
        attempts = bucket["attempts"]
        successes = bucket["successes"]
        success_rate = (successes / attempts) * 100 if attempts else 0.0
        avg_latency = (
            sum(bucket["latencies"]) / len(bucket["latencies"]) if bucket["latencies"] else None
        )
        avg_tokens_per_second = (
            sum(bucket["tokens_per_second"]) / len(bucket["tokens_per_second"])
            if bucket["tokens_per_second"]
            else None
        )
        rows.append(
            {
                "report": bucket["report"],
                "family": bucket["family"],
                "attempts": attempts,
                "successes": successes,
                "failures": bucket["failures"],
                "success_rate": success_rate,
                "avg_latency": avg_latency,
                "avg_tokens_per_second": avg_tokens_per_second,
            }
        )
    return rows


def aggregate_case_metrics(reports: list[dict]) -> list[dict]:
    rows = []
    for report in reports:
        for item in report["summary"]:
            attempts = item["attempts"]
            successes = item["successes"]
            rows.append(
                {
                    "report": report["name"],
                    "region": item["region"],
                    "family": item["family"],
                    "model": item["model"],
                    "case_id": item["case_id"],
                    "concurrency": item.get("concurrency", 1),
                    "streaming": item.get("streaming", False),
                    "attempts": attempts,
                    "successes": successes,
                    "failures": item["failures"],
                    "success_rate": (successes / attempts) * 100 if attempts else 0.0,
                    "avg_latency": item["avg_latency_seconds"],
                    "p95_latency": item["p95_latency_seconds"],
                    "p99_latency": item.get("p99_latency_seconds"),
                    "avg_ttft": item.get("avg_ttft_seconds"),
                    "avg_tokens": item["avg_total_tokens"],
                    "avg_tokens_per_second": item.get("avg_output_tokens_per_second"),
                    "avg_post_ttft_tokens_per_second": item.get("avg_post_ttft_output_tokens_per_second"),
                }
            )
    return rows


def collect_failures(reports: list[dict]) -> list[dict]:
    failures = []
    for report in reports:
        for item in report["results"]:
            if item.get("error"):
                failures.append(
                    {
                        "report": report["name"],
                        "region": item["region"],
                        "family": item["family"],
                        "model": item["model"],
                        "case_id": item["case_id"],
                        "concurrency": item.get("concurrency", 1),
                        "streaming": item.get("streaming", False),
                        "iteration": item["iteration"],
                        "latency": item["latency_seconds"],
                        "error": item["error"],
                        "error_type": item.get("error_type"),
                        "http_status": item.get("http_status"),
                        "response_body_preview": item.get("response_body_preview"),
                        "request_id": item.get("request_id"),
                    }
                )
    return failures


def collect_filter_options(rows: list[dict]) -> dict[str, list[str]]:
    return {
        "families": sorted({row["family"] for row in rows if row.get("family")}),
        "regions": sorted({row["region"] for row in rows if row.get("region")}),
        "models": sorted({row["model"] for row in rows if row.get("model")}),
        "concurrency": sorted({str(row["concurrency"]) for row in rows}, key=lambda value: int(value)),
    }


def collect_skipped_combinations(reports: list[dict]) -> list[dict]:
    rows = []
    for report in reports:
        for item in report.get("skipped", []):
            rows.append(
                {
                    "report": report["name"],
                    "region": item.get("region", ""),
                    "family": item.get("family", ""),
                    "model": item.get("model", ""),
                    "reason": item.get("reason") or "지원 리전 아님",
                }
            )
    return rows


def collect_failure_summary(failures: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    for failure in failures:
        key = (
            failure["report"],
            failure["family"],
            failure["model"],
            str(failure["http_status"] or "-"),
            failure["error_type"] or "-",
        )
        buckets[key] += 1
    return [
        {
            "report": report,
            "family": family,
            "model": model,
            "http_status": http_status,
            "error_type": error_type,
            "count": count,
        }
        for (report, family, model, http_status, error_type), count in sorted(
            buckets.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def css() -> str:
    return """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 0;
  background: #f5f3ee;
  color: #1d1b18;
}
.page {
  max-width: 1180px;
  margin: 0 auto;
  padding: 32px 24px 64px;
}
h1, h2 {
  margin: 0 0 16px;
}
.section-title {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.info-tooltip {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  border: 1px solid #b8ad9f;
  border-radius: 999px;
  color: #4d463f;
  background: #fffaf2;
  font-size: 12px;
  font-weight: 800;
  line-height: 1;
  cursor: help;
}
.tooltip-box {
  position: absolute;
  left: 50%;
  bottom: calc(100% + 8px);
  transform: translateX(-50%);
  z-index: 20;
  display: none;
  width: min(320px, 78vw);
  padding: 9px 10px;
  border: 1px solid #cfc5b6;
  border-radius: 8px;
  background: #1d1b18;
  color: #fffdf9;
  font-size: 12px;
  font-weight: 500;
  line-height: 1.45;
  box-shadow: 0 8px 24px rgba(46, 39, 31, 0.18);
}
.info-tooltip:hover .tooltip-box,
.info-tooltip:focus .tooltip-box {
  display: block;
}
.lede {
  color: #4d463f;
  margin-bottom: 28px;
}
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 14px;
  margin-bottom: 28px;
}
.card {
  background: #fffdf9;
  border: 1px solid #ddd4c7;
  border-radius: 14px;
  padding: 16px;
  box-shadow: 0 4px 16px rgba(46, 39, 31, 0.05);
}
.metric {
  font-size: 28px;
  font-weight: 700;
  margin: 6px 0 4px;
}
.muted {
  color: #6b635b;
  font-size: 14px;
}
.section {
  margin-top: 34px;
}
.filters {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 12px;
  margin: 20px 0 28px;
  padding: 14px;
  background: #fffdf9;
  border: 1px solid #ddd4c7;
  border-radius: 12px;
}
.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.field label {
  color: #6b635b;
  font-size: 13px;
  font-weight: 600;
}
.field select {
  border: 1px solid #cfc5b6;
  border-radius: 8px;
  background: #fffaf2;
  color: #1d1b18;
  font: inherit;
  padding: 8px 10px;
}
.chart-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
}
.panel {
  background: #fffdf9;
  border: 1px solid #ddd4c7;
  border-radius: 14px;
  padding: 16px;
}
.bar-row {
  margin: 14px 0;
}
.filtered-out {
  display: none;
}
.bar-label {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 14px;
  margin-bottom: 6px;
}
.bar-track {
  width: 100%;
  height: 14px;
  background: #ece5d8;
  border-radius: 999px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  border-radius: 999px;
}
.bar-fill.openai { background: linear-gradient(90deg, #0f766e, #14b8a6); }
.bar-fill.gemini { background: linear-gradient(90deg, #b45309, #f59e0b); }
.bar-fill.grok { background: linear-gradient(90deg, #1d4ed8, #60a5fa); }
.bar-fill.meta { background: linear-gradient(90deg, #7c3aed, #c084fc); }
.bar-fill.cohere { background: linear-gradient(90deg, #be123c, #fb7185); }
table {
  width: 100%;
  border-collapse: collapse;
  background: #fffdf9;
  border-radius: 14px;
  overflow: hidden;
}
th, td {
  padding: 10px 12px;
  border-bottom: 1px solid #e7dfd4;
  text-align: left;
  font-size: 14px;
  vertical-align: top;
}
th {
  background: #f1eadf;
}
code {
  background: #f3eee6;
  padding: 1px 5px;
  border-radius: 6px;
}
.pill {
  display: inline-block;
  padding: 3px 8px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
}
.pill.openai { background: #ccfbf1; color: #115e59; }
.pill.gemini { background: #fef3c7; color: #92400e; }
.pill.grok { background: #dbeafe; color: #1d4ed8; }
.pill.meta { background: #ede9fe; color: #6d28d9; }
.pill.cohere { background: #ffe4e6; color: #be123c; }
a { color: #8b5e34; text-decoration: none; }
a:hover { text-decoration: underline; }
.scatter {
  width: 100%;
  height: auto;
  background: linear-gradient(180deg, #fffaf2, #fffdf9);
  border-radius: 12px;
}
.axis {
  stroke: #a89f93;
  stroke-width: 1;
}
.axis-label {
  fill: #6b635b;
  font-size: 11px;
}
.axis-title {
  fill: #4d463f;
  font-size: 12px;
  font-weight: 600;
}
.dot.openai { fill: #14b8a6; }
.dot.gemini { fill: #f59e0b; }
.dot.grok { fill: #60a5fa; }
.dot.meta { fill: #c084fc; }
.dot.cohere { fill: #fb7185; }
.dot-label {
  fill: #4d463f;
  font-size: 10px;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 10px;
  font-size: 13px;
  color: #4d463f;
}
.legend span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  display: inline-block;
}
.legend-dot.openai { background: #14b8a6; }
.legend-dot.gemini { background: #f59e0b; }
.legend-dot.grok { background: #60a5fa; }
.legend-dot.meta { background: #c084fc; }
.legend-dot.cohere { background: #fb7185; }
.tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 10px 0 14px;
}
.tab-button {
  border: 1px solid #cfc5b6;
  border-radius: 8px;
  background: #fffaf2;
  color: #4d463f;
  font: inherit;
  font-weight: 700;
  padding: 7px 10px;
  cursor: pointer;
}
.tab-button.active {
  background: #1d1b18;
  border-color: #1d1b18;
  color: #fffdf9;
}
.tab-panel.hidden {
  display: none;
}
.matrix-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.9fr);
  gap: 18px;
}
.heatmap {
  display: grid;
  gap: 8px;
  align-items: stretch;
}
.heat-cell {
  min-height: 62px;
  border: 1px solid rgba(46, 39, 31, 0.12);
  border-radius: 8px;
  padding: 10px;
}
.heat-cell.header {
  min-height: auto;
  background: #f1eadf;
  color: #4d463f;
  font-size: 12px;
  font-weight: 700;
}
.heat-cell.source {
  background: #fffaf2;
  font-weight: 700;
}
.heat-value {
  font-size: 20px;
  font-weight: 800;
}
.heat-note {
  color: #4d463f;
  font-size: 12px;
  margin-top: 4px;
}
.runner-tiles {
  display: grid;
  gap: 12px;
}
.runner-tile {
  background: #fffaf2;
  border: 1px solid #ddd4c7;
  border-radius: 8px;
  padding: 14px;
}
.runner-tile h3 {
  margin: 0 0 10px;
  font-size: 16px;
}
.tile-metric {
  font-size: 26px;
  font-weight: 800;
  margin-bottom: 8px;
}
.tile-line {
  color: #4d463f;
  font-size: 13px;
  margin-top: 4px;
}
.context-grid {
  display: grid;
  grid-template-columns: minmax(0, 0.95fr) minmax(0, 1.05fr);
  gap: 18px;
  margin-top: 18px;
}
.context-list {
  margin: 0;
  padding-left: 18px;
}
.context-list li {
  margin: 8px 0;
}
.workload-grid {
  display: grid;
  gap: 10px;
}
.workload-item {
  background: #fffaf2;
  border: 1px solid #e5ddcf;
  border-radius: 8px;
  padding: 12px;
}
.workload-item strong {
  display: block;
  margin-bottom: 5px;
}
.workload-detail {
  overflow-x: auto;
}
.workload-detail h3 {
  margin: 0 0 8px;
}
.workload-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 16px;
  color: #4d463f;
  font-size: 13px;
  margin: 10px 0 14px;
}
details {
  margin-top: 12px;
}
details.debug-details {
  background: #fffdf9;
  border: 1px solid #ddd4c7;
  border-radius: 14px;
  padding: 16px;
}
summary {
  cursor: pointer;
  color: #4d463f;
  font-weight: 700;
}
.prompt-block {
  margin-top: 10px;
}
pre {
  white-space: pre-wrap;
  word-break: break-word;
  background: #f3eee6;
  border-radius: 8px;
  padding: 10px;
  margin: 6px 0 0;
  font-size: 12px;
  line-height: 1.45;
}
@media (max-width: 900px) {
  .chart-grid { grid-template-columns: 1fr; }
  .matrix-grid { grid-template-columns: 1fr; }
  .context-grid { grid-template-columns: 1fr; }
}
"""


def info_icon(description: str) -> str:
    text = escape(description)
    return (
        "<span class='info-tooltip' tabindex='0' aria-label='설명' title='마우스를 올리면 설명이 표시됩니다'>"
        "i"
        f"<span class='tooltip-box'>{text}</span>"
        "</span>"
    )


def title_html(level: int, title: str, description: str) -> str:
    return f"<h{level} class='section-title'>{escape(title)}{info_icon(description)}</h{level}>"


WORKLOAD_DESCRIPTIONS = {
    "summary-ko": "긴 내용을 한국어로 짧게 요약하게 하는 테스트입니다. 사용자가 빠르게 핵심만 읽을 수 있는 답을 만드는 속도를 봅니다.",
    "table-en": "작은 모델과 큰 모델의 장단점을 영어 표로 정리하게 하는 테스트입니다. 단순 문장뿐 아니라 표처럼 구조가 있는 답을 만드는 속도를 봅니다.",
    "ops-checklist": "운영자가 점검표를 만들듯 확인 항목을 한국어 목록으로 정리하게 하는 테스트입니다. 여러 항목을 빠짐없이 정리하는 속도를 봅니다.",
    "chat-helpdesk": "사용자 문의에 상담원처럼 답하게 하는 테스트입니다. 원인 후보와 먼저 확인할 일을 쉽게 정리하는 속도를 봅니다.",
    "code-debug": "짧은 코드의 문제를 찾고 수정 방향을 설명하게 하는 테스트입니다. 코드 이해와 실용적인 설명 속도를 봅니다.",
    "reasoning-choice": "여러 조건을 비교해서 더 나은 선택을 고르게 하는 테스트입니다. 간단한 판단과 근거 설명 속도를 봅니다.",
    "agentic-plan": "실제 도구를 실행하지 않고 작업 순서와 확인 항목을 계획하게 하는 테스트입니다. 실행 전 계획 수립 능력과 속도를 봅니다.",
    "nl2sql-sales-analytics": "사람이 말로 한 매출 분석 질문을 SQL SELECT query로 바꾸게 하는 테스트입니다. 실제 DB 실행 없이 SQL을 만드는 속도를 봅니다.",
}


WORKLOAD_DETAILS = {
    "summary-ko": ("Summarization", "짧은 한국어 요약"),
    "table-en": ("Structured writing", "영어 markdown table"),
    "ops-checklist": ("Operational checklist", "한국어 점검 목록"),
    "chat-helpdesk": ("Chat / support", "원인 후보와 다음 확인 단계"),
    "code-debug": ("Code reasoning", "버그 설명과 수정 방향"),
    "reasoning-choice": ("Reasoning", "선택지와 짧은 근거"),
    "agentic-plan": ("Agentic planning", "실행 전 작업 순서와 확인 항목"),
    "nl2sql-sales-analytics": ("NL2SQL", "안전한 SELECT SQL query"),
}


def latest_suite_summary(summaries: list[dict]) -> dict | None:
    if not summaries:
        return None
    return max(summaries, key=lambda item: (item.get("generated_at") or "", item["name"]))


def average(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def percentile_value(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def format_latency(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}s"


def format_number(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}%"


def latency_grade(value: float | None) -> str:
    if value is None:
        return "No data"
    if value < 2.0:
        return "Good"
    if value < 4.0:
        return "Watch"
    return "Slow"


def heat_color(value: float | None, minimum: float, maximum: float) -> str:
    if value is None:
        return "#f3eee6"
    if maximum <= minimum:
        return "#d1fae5"
    midpoint = minimum + (maximum - minimum) / 2
    if value <= midpoint:
        ratio = (value - minimum) / (midpoint - minimum) if midpoint > minimum else 0.0
        start = (209, 250, 229)
        end = (254, 243, 199)
    else:
        ratio = (value - midpoint) / (maximum - midpoint) if maximum > midpoint else 0.0
        start = (254, 243, 199)
        end = (254, 226, 226)
    rgb = [round(start[index] + (end[index] - start[index]) * ratio) for index in range(3)]
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


def suite_reports(summary: dict, reports: list[dict]) -> list[dict]:
    report_names = {
        Path(row["json"]).stem
        for row in summary.get("sources", [])
        if row.get("json")
    }
    selected = [report for report in reports if report["name"] in report_names]
    return selected or reports


def focus_reports(summaries: list[dict], reports: list[dict]) -> list[dict]:
    summary = latest_suite_summary(summaries)
    return suite_reports(summary, reports) if summary else reports


def format_config_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "-"
    return str(value)


def report_source_label(report: dict) -> str:
    return report.get("benchmark_config", {}).get("source_label") or report["name"]


def render_runner_context(summary: dict, reports: list[dict]) -> str:
    selected_reports = suite_reports(summary, reports)
    model_labels = sorted(
        {
            model.get("label") or model.get("model_id") or ""
            for report in selected_reports
            for model in report.get("selected_models", [])
            if model.get("label") or model.get("model_id")
        }
    )
    case_ids = sorted(
        {
            item.get("case_id") or ""
            for report in selected_reports
            for item in report.get("summary", [])
            if item.get("case_id")
        }
    )
    config = next((report.get("benchmark_config", {}) for report in selected_reports if report.get("benchmark_config")), {})
    repeats = config.get("repeats", "-")
    concurrency = format_config_value(config.get("concurrency_levels"))
    streaming = "-" if "streaming" not in config else ("yes" if config.get("streaming") else "no")
    temperature = config.get("temperature", "-")
    max_tokens = config.get("max_tokens", "-")

    model_items = "".join(f"<li><code>{escape(label)}</code></li>" for label in model_labels) or "<li>-</li>"
    return (
        "<div class='section'>"
        "<div class='panel'>"
        + title_html(2, "Test Context", "벤치마크 실행 조건, 선택 모델, workload, 동시성, 생성 설정을 한곳에서 확인합니다.")
        + "<p class='muted'>각 source runner가 같은 workload를 target region으로 보내고 latency, 처리량, 성공률을 비교합니다. 모델별 파라미터 수와 아키텍처는 다르므로, 이 결과는 동일 파라미터급 품질 비교가 아니라 동일 실행 조건에서의 managed serving 성능 비교입니다.</p>"
        "<ul class='context-list'>"
        f"<li>Models tested:<ul>{model_items}</ul></li>"
        f"<li>Workloads tested: <code>{escape(', '.join(case_ids) if case_ids else '-')}</code></li>"
        f"<li>Execution: repeats <code>{escape(str(repeats))}</code>, concurrency <code>{escape(concurrency)}</code>, streaming <code>{streaming}</code></li>"
        f"<li>Generation settings: temperature <code>{escape(str(temperature))}</code>, max tokens <code>{escape(str(max_tokens))}</code></li>"
        "<li>해석 기준: latency는 낮을수록, output tokens/sec는 높을수록 좋습니다. 이 값은 품질 평가가 아니라 속도와 안정성 중심의 실행 지표입니다.</li>"
        "</ul>"
        "</div></div>"
    )


def metric_row_from_values(values: list[dict]) -> dict:
    attempts = sum(item["attempts"] for item in values)
    successes = sum(item["successes"] for item in values)
    latency_values = [item["avg_latency"] for item in values if item.get("avg_latency") is not None]
    p95_values = [item["p95_latency"] for item in values if item.get("p95_latency") is not None]
    ttft_values = [item["avg_ttft"] for item in values if item.get("avg_ttft") is not None]
    token_values = [item["avg_tokens"] for item in values if item.get("avg_tokens") is not None]
    throughput_values = [
        item["avg_tokens_per_second"]
        for item in values
        if item.get("avg_tokens_per_second") is not None
    ]
    return {
        "attempts": attempts,
        "successes": successes,
        "failures": sum(item["failures"] for item in values),
        "success_rate": (successes / attempts * 100) if attempts else None,
        "avg_latency": average(latency_values),
        "p95_latency": average(p95_values),
        "avg_ttft": average(ttft_values),
        "avg_tokens": average(token_values),
        "avg_tokens_per_second": average(throughput_values),
    }


def aggregate_concurrency_metrics(case_rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in case_rows:
        grouped[int(row.get("concurrency", 1))].append(row)
    return [
        {"concurrency": concurrency, **metric_row_from_values(values)}
        for concurrency, values in sorted(grouped.items())
    ]


def aggregate_model_concurrency_metrics(case_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in case_rows:
        grouped[(row["family"], row["model"], int(row.get("concurrency", 1)))].append(row)
    rows = []
    for (family, model, concurrency), values in sorted(grouped.items()):
        rows.append(
            {
                "family": family,
                "model": model,
                "concurrency": concurrency,
                **metric_row_from_values(values),
            }
        )
    return rows


def c50_or_max_concurrency(case_rows: list[dict]) -> int | None:
    levels = sorted({int(row.get("concurrency", 1)) for row in case_rows})
    if not levels:
        return None
    return 50 if 50 in levels else levels[-1]


def render_load_summary(case_rows: list[dict]) -> str:
    rows = aggregate_concurrency_metrics(case_rows)
    if not rows:
        return ""
    by_concurrency = {row["concurrency"]: row for row in rows}
    levels = [row["concurrency"] for row in rows]
    level_text = ", ".join(f"C{level}" for level in levels)
    max_concurrency = max(levels)
    c1 = by_concurrency.get(1)
    max_row = by_concurrency.get(max_concurrency)
    multiplier = None
    if c1 and max_row and max_concurrency != 1 and c1.get("p95_latency") and max_row.get("p95_latency"):
        multiplier = max_row["p95_latency"] / c1["p95_latency"]

    body = []
    for row in rows:
        label = f"C{row['concurrency']}"
        body.append(
            "<tr>"
            f"<td><strong>{escape(label)}</strong></td>"
            f"<td>{row['successes']}/{row['attempts']} ({escape(format_percent(row['success_rate']))})</td>"
            f"<td>{escape(format_latency(row['p95_latency']))}</td>"
            f"<td>{escape(format_number(row['avg_tokens_per_second']))}</td>"
            f"<td>{escape(format_latency(row['avg_ttft']))}</td>"
            "</tr>"
        )

    multiplier_text = f"{multiplier:.2f}x" if multiplier is not None else "-"
    return (
        "<div class='section'>"
        + title_html(2, "Load Summary", "동시성 수준별로 성공률, P95 latency, tokens/sec, TTFT를 요약합니다.")
        + f"<p class='muted'>{escape(level_text)} 기준으로 success rate, P95 latency, output tokens/sec, TTFT를 요약합니다. C{max_concurrency}/C1 latency multiplier는 부하가 커질 때 P95가 얼마나 증가했는지 보여줍니다.</p>"
        "<table><thead><tr><th>Concurrency</th><th>Success</th><th>P95 Latency</th><th>Tok/sec</th><th>TTFT</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
        f"<p class='muted'><strong>C{max_concurrency}/C1 latency multiplier:</strong> {escape(multiplier_text)}</p>"
        "</div>"
    )


def render_c50_ranking(case_rows: list[dict]) -> str:
    target_concurrency = c50_or_max_concurrency(case_rows)
    if target_concurrency is None:
        return ""
    rows = [
        row
        for row in aggregate_model_concurrency_metrics(case_rows)
        if row["concurrency"] == target_concurrency
    ]
    rows.sort(
        key=lambda row: (
            -(row["success_rate"] or 0),
            row["p95_latency"] if row["p95_latency"] is not None else float("inf"),
            -(row["avg_tokens_per_second"] or 0),
            row["family"],
            row["model"],
        )
    )
    body = []
    for index, row in enumerate(rows, start=1):
        body.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td><span class='pill {escape(row['family'])}'>{escape(row['family'])}</span></td>"
            f"<td><code>{escape(row['model'])}</code></td>"
            f"<td>{row['successes']}/{row['attempts']} ({escape(format_percent(row['success_rate']))})</td>"
            f"<td>{escape(format_latency(row['p95_latency']))}</td>"
            f"<td>{escape(format_number(row['avg_tokens_per_second']))}</td>"
            f"<td>{escape(format_latency(row['avg_ttft']))}</td>"
            "</tr>"
        )
    return (
        "<div class='section'>"
        + title_html(2, f"C{target_concurrency} Ranking", "가장 높은 동시성 기준으로 모델 순위를 비교합니다. 성공률을 먼저 보고, 그 다음 P95 latency와 tokens/sec를 봅니다.")
        + "<p class='muted'>기본 정렬은 success rate를 최우선으로 보고, 그 다음 P95 latency, output tokens/sec 순서로 비교합니다.</p>"
        "<table><thead><tr><th>Rank</th><th>Family</th><th>Model</th><th>Success</th><th>P95 Latency</th><th>Tok/sec</th><th>TTFT</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def render_load_sensitivity(case_rows: list[dict]) -> str:
    rows = aggregate_model_concurrency_metrics(case_rows)
    if not rows:
        return ""
    by_model: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for row in rows:
        by_model[(row["family"], row["model"])][row["concurrency"]] = row
    levels = sorted({row["concurrency"] for row in rows})
    headers = "".join(f"<th>C{level}</th>" for level in levels)
    body = []
    for (family, model), values in sorted(by_model.items()):
        cells = []
        for level in levels:
            row = values.get(level)
            if row is None:
                cells.append("<td>-</td>")
            else:
                cells.append(
                    "<td>"
                    f"P95 {escape(format_latency(row['p95_latency']))}<br>"
                    f"Success {escape(format_percent(row['success_rate']))}<br>"
                    f"TTFT {escape(format_latency(row['avg_ttft']))}"
                    "</td>"
                )
        body.append(
            "<tr>"
            f"<td><span class='pill {escape(family)}'>{escape(family)}</span></td>"
            f"<td><code>{escape(model)}</code></td>"
            + "".join(cells)
            + "</tr>"
        )
    return (
        "<div class='section'>"
        + title_html(2, "Load Sensitivity", "모델별로 부하 단계가 올라갈 때 latency와 성공률이 어떻게 변하는지 보여줍니다.")
        + f"<p class='muted'>모델별 {' -> '.join(escape(f'C{level}') for level in levels)} 변화를 통해 동시성이 커질 때 latency 증가나 성공률 하락이 어디서 발생하는지 확인합니다.</p>"
        "<table><thead><tr><th>Family</th><th>Model</th>"
        + headers
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def source_target_metrics(summary: dict, reports: list[dict], concurrency: int | None = None) -> dict[tuple[str, str], dict]:
    selected_reports = suite_reports(summary, reports)
    source_by_report = {
        Path(row["json"]).stem: row["source"]
        for row in summary.get("sources", [])
        if row.get("json")
    }
    rows_by_source_target: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for report in selected_reports:
        source = source_by_report.get(report["name"]) or report_source_label(report)
        for item in aggregate_case_metrics([report]):
            if concurrency is not None and item.get("concurrency", 1) != concurrency:
                continue
            rows_by_source_target[(source, item["region"])].append(item)
    return {
        key: metric_row_from_values(values)
        for key, values in rows_by_source_target.items()
    }


def prompt_paths_from_reports(reports: list[dict]) -> list[Path]:
    paths = []
    for report in reports:
        config = report.get("benchmark_config", {})
        prompt_path = config.get("prompt_file") or config.get("prompts")
        if not prompt_path:
            continue
        path = Path(prompt_path)
        paths.append(path if path.is_absolute() else PROJECT_ROOT / path)
    paths.extend(
        [
            PROJECT_ROOT / "prompts" / "sample_prompts.jsonl",
            PROJECT_ROOT / "prompts" / "chat_nl2sql_workloads.jsonl",
        ]
    )
    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def load_workload_prompts(reports: list[dict]) -> dict[str, list[dict[str, str]]]:
    workloads = {}
    for path in prompt_paths_from_reports(reports):
        if not path.exists():
            continue
        for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = payload.get("id") or f"case-{lineno}"
            if case_id in workloads:
                continue
            if "messages" in payload and isinstance(payload["messages"], list):
                workloads[case_id] = [
                    {
                        "role": str(message.get("role", "user")),
                        "content": str(message.get("content", "")),
                    }
                    for message in payload["messages"]
                    if isinstance(message, dict)
                ]
            elif "prompt" in payload:
                workloads[case_id] = [{"role": "user", "content": str(payload["prompt"])}]
    return workloads


def render_prompt_messages(messages: list[dict[str, str]]) -> str:
    if not messages:
        return "<p class='muted'>이 workload의 prompt 원문을 찾지 못했습니다.</p>"
    blocks = []
    for message in messages:
        blocks.append(
            "<div class='prompt-block'>"
            f"<div class='muted'><code>{escape(message['role'])}</code></div>"
            f"<pre>{escape(message['content'])}</pre>"
            "</div>"
        )
    return "".join(blocks)


def render_workload_metric_table(rows: list[dict]) -> str:
    if not rows:
        return "<p class='muted'>이 workload에 대한 측정 결과가 없습니다.</p>"
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td><code>{escape(row['report'])}</code></td>"
            f"<td><code>{escape(row['region'])}</code></td>"
            f"<td><span class='pill {escape(row['family'])}'>{escape(row['family'])}</span></td>"
            f"<td><code>{escape(row['model'])}</code></td>"
            f"<td>{row['successes']}/{row['attempts']} ({row['success_rate']:.1f}%)</td>"
            f"<td>{escape(format_latency(row['avg_latency']))}</td>"
            f"<td>{escape(format_latency(row['p95_latency']))}</td>"
            f"<td>{escape(format_latency(row['p99_latency']))}</td>"
            f"<td>{escape(format_number(row['avg_tokens']))}</td>"
            f"<td>{escape(format_number(row['avg_tokens_per_second']))}</td>"
            f"<td>{escape(format_latency(row['avg_ttft']))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Report</th><th>Region</th><th>Family</th><th>Model</th><th>Success</th>"
        "<th>Avg Latency</th><th>P95</th><th>P99</th><th>Avg Tokens</th><th>Tok/sec</th><th>TTFT</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def render_workload_details(case_rows: list[dict], reports: list[dict]) -> str:
    if not case_rows:
        return ""
    prompts = load_workload_prompts(reports)
    case_ids = sorted({row["case_id"] for row in case_rows})
    cards = []
    for case_id in case_ids:
        kind, expected = WORKLOAD_DETAILS.get(case_id, ("Benchmark workload", "모델 응답"))
        description = WORKLOAD_DESCRIPTIONS.get(
            case_id,
            "이 workload는 같은 입력을 여러 모델과 리전에 보내 응답 속도와 처리량을 비교하는 테스트입니다.",
        )
        rows = [row for row in case_rows if row["case_id"] == case_id]
        cards.append(
            "<div class='workload-item workload-detail'>"
            + title_html(3, case_id, description)
            + f"<p class='muted'>{escape(description)}</p>"
            "<div class='workload-meta'>"
            f"<span>Type: <code>{escape(kind)}</code></span>"
            f"<span>Expected output: {escape(expected)}</span>"
            "</div>"
            f"{render_workload_metric_table(rows)}"
            "<details>"
            "<summary>Prompt 원문 보기</summary>"
            f"{render_prompt_messages(prompts.get(case_id, []))}"
            "</details>"
            "</div>"
        )
    return (
        "<div class='section'>"
        + title_html(2, "Workload Details", "각 workload의 목적, 기대 출력, prompt 원문, 모델별 측정치를 확인합니다.")
        + "<p class='muted'>각 테스트 케이스가 무엇을 시키는지와, workload별 latency/token/sec/TTFT 결과를 함께 보여줍니다.</p>"
        "<div class='workload-grid'>"
        f"{''.join(cards)}"
        "</div></div>"
    )


def render_runner_matrix(summaries: list[dict], reports: list[dict]) -> str:
    summary = latest_suite_summary(summaries)
    if not summary:
        return ""

    sources = [row["source"] for row in summary["sources"]]
    targets = summary["target_regions"] or sorted({row["target_region"] for row in summary["target_latency"]})
    latency_by_pair = {
        (row["source"], row["target_region"]): row["avg_latency"]
        for row in summary["target_latency"]
    }
    selected_reports = suite_reports(summary, reports)
    concurrency_levels = sorted(
        {
            int(item.get("concurrency", 1))
            for report in selected_reports
            for item in report.get("summary", [])
        }
    ) or [1]
    latencies = [value for value in latency_by_pair.values() if value is not None]
    minimum = min(latencies) if latencies else 0.0
    maximum = max(latencies) if latencies else 0.0
    source_best = {}
    target_best = {}
    for source in sources:
        values = [
            (target, latency_by_pair.get((source, target)))
            for target in targets
            if latency_by_pair.get((source, target)) is not None
        ]
        if values:
            source_best[source] = min(values, key=lambda item: item[1])
    for target in targets:
        values = [
            (source, latency_by_pair.get((source, target)))
            for source in sources
            if latency_by_pair.get((source, target)) is not None
        ]
        if values:
            target_best[target] = min(values, key=lambda item: item[1])

    grid_columns = "150px " + " ".join("minmax(130px, 1fr)" for _ in targets)
    tab_buttons = []
    panels = []
    for index, concurrency in enumerate(concurrency_levels):
        active = index == 0
        label = f"C{concurrency}"
        tab_buttons.append(
            "<button class='tab-button{active}' type='button' data-tab-target='heatmap-{concurrency}'>{label}</button>".format(
                active=" active" if active else "",
                concurrency=concurrency,
                label=escape(label),
            )
        )
        metrics_by_pair = source_target_metrics(summary, reports, concurrency)
        heat_values = [
            metrics.get("p95_latency") or metrics.get("avg_latency")
            for metrics in metrics_by_pair.values()
            if metrics.get("p95_latency") is not None or metrics.get("avg_latency") is not None
        ] or latencies
        heat_minimum = min(heat_values) if heat_values else minimum
        heat_maximum = max(heat_values) if heat_values else maximum
        cells = [
            f"<div id='heatmap-{concurrency}' class='tab-panel{' hidden' if not active else ''}'>",
            f"<div class='heatmap' style='grid-template-columns:{grid_columns}'>",
            "<div class='heat-cell header'>Source / Target</div>",
        ]
        for target in targets:
            cells.append(f"<div class='heat-cell header'><code>{escape(target)}</code></div>")
        for source in sources:
            cells.append(f"<div class='heat-cell source'><code>{escape(source)}</code></div>")
            for target in targets:
                metrics = metrics_by_pair.get((source, target), {})
                latency = metrics.get("p95_latency") or metrics.get("avg_latency") or latency_by_pair.get((source, target))
                labels = []
                if source_best.get(source, (None, None))[0] == target:
                    labels.append("source best")
                if target_best.get(target, (None, None))[0] == source:
                    labels.append("target best")
                note = ", ".join(labels) if labels else "avg latency"
                cells.append(
                    "<div class='heat-cell' "
                    f"style='background:{heat_color(latency, heat_minimum, heat_maximum)}'>"
                    f"<div class='heat-value'>{escape(format_latency(latency))}</div>"
                    f"<div class='heat-note'>{escape(note)} · {escape(latency_grade(latency))}</div>"
                    f"<div class='heat-note'>P95 {escape(format_latency(metrics.get('p95_latency')))}</div>"
                    f"<div class='heat-note'>Tok/sec {escape(format_number(metrics.get('avg_tokens_per_second')))}</div>"
                    f"<div class='heat-note'>Success {escape(format_percent(metrics.get('success_rate')))}</div>"
                    f"<div class='heat-note'>TTFT {escape(format_latency(metrics.get('avg_ttft')))}</div>"
                    "</div>"
                )
        cells.append("</div></div>")
        panels.append("".join(cells))
    heatmap_html = "<div class='tabs'>" + "".join(tab_buttons) + "</div>" + "".join(panels)

    source_rows = {row["source"]: row for row in summary["sources"]}
    tiles = ["<div class='runner-tiles'>"]
    for source in sources:
        row = source_rows[source]
        values = [
            (target, latency_by_pair.get((source, target)))
            for target in targets
            if latency_by_pair.get((source, target)) is not None
        ]
        best = min(values, key=lambda item: item[1]) if values else ("-", None)
        worst = max(values, key=lambda item: item[1]) if values else ("-", None)
        success_rate = (row["successes"] / row["attempts"] * 100) if row["attempts"] else 0.0
        tiles.append(
            "<div class='runner-tile'>"
            + title_html(3, source, "이 source runner에서 실행된 전체 요청의 평균 latency, fastest target, slowest target, success rate입니다.")
            + f"<div class='tile-metric'>{escape(format_latency(row['avg_latency']))}</div>"
            f"<div class='tile-line'>Best target: <code>{escape(best[0])}</code> ({escape(format_latency(best[1]))})</div>"
            f"<div class='tile-line'>Worst target: <code>{escape(worst[0])}</code> ({escape(format_latency(worst[1]))})</div>"
            f"<div class='tile-line'>Success: {row['successes']}/{row['attempts']} ({success_rate:.1f}%)</div>"
            "</div>"
        )
    tiles.append("</div>")

    return (
        "<div class='section'>"
        + title_html(2, f"Region-to-Region Performance: {summary['name']}", "source runner와 target region 조합별 성능을 heatmap으로 비교합니다.")
        + "<p class='muted'>이 heatmap은 suite latency 값에 각 runner JSON report의 P95, success rate, tokens/sec, TTFT를 함께 표시합니다. latency는 낮을수록, tokens/sec는 높을수록 좋습니다.</p>"
        + "<div class='matrix-grid'>"
        f"<div class='panel'>{heatmap_html}</div>"
        "<div class='panel'>"
        + title_html(2, "Metric Guide", "heatmap 셀에 표시되는 latency 등급과 보조 지표의 의미를 설명합니다.")
        + "<p class='muted'>Latency grade는 Good &lt; 2s, Watch 2-4s, Slow &gt;= 4s 기준입니다. TTFT는 streaming으로 첫 토큰 시각이 잡힌 실행에서만 표시되고, 없으면 '-'로 표시됩니다.</p>"
        + title_html(2, "Runner Summary", "source runner별 전체 평균 latency와 가장 빠른/느린 target, 성공률을 요약합니다.")
        + f"<p class='muted'>각 tile은 source runner 하나의 전체 평균 latency, fastest target, slowest target, success rate를 요약합니다.</p>{''.join(tiles)}</div>"
        "</div>"
        f"{render_runner_context(summary, reports)}"
        "</div>"
    )


def render_bars(rows: list[dict], metric: str, title: str, value_suffix: str) -> str:
    max_value = max((row[metric] or 0) for row in rows) if rows else 0
    chunks = [
        "<div class='panel'>"
        + title_html(2, title, "report와 model family 단위로 집계된 bar chart입니다.")
    ]
    for row in rows:
        raw_value = row[metric] or 0
        width = (raw_value / max_value * 100) if max_value else 0
        label = f"{row['report']} / {row['family']}"
        shown = f"{raw_value:.1f}{value_suffix}" if isinstance(raw_value, float) else f"{raw_value}{value_suffix}"
        chunks.append(
            "<div class='bar-row'>"
            f"<div class='bar-label'><span>{escape(label)}</span><span>{escape(shown)}</span></div>"
            "<div class='bar-track'>"
            f"<div class='bar-fill {escape(row['family'])}' style='width:{width:.1f}%'></div>"
            "</div></div>"
        )
    chunks.append("</div>")
    return "".join(chunks)


def render_case_bars(rows: list[dict], metric: str, title: str, value_suffix: str) -> str:
    max_value = max((row[metric] or 0) for row in rows) if rows else 0
    chunks = [
        "<div class='panel'>"
        + title_html(2, title, "case 단위로 성공률, latency, tokens/sec가 어디서 달라지는지 보여줍니다.")
    ]
    for row in rows:
        raw_value = row[metric] or 0
        width = (raw_value / max_value * 100) if max_value else 0
        label = f"{row['report']} / {row['family']} / {row['case_id']} / c{row['concurrency']}"
        shown = f"{raw_value:.1f}{value_suffix}" if isinstance(raw_value, float) else f"{raw_value}{value_suffix}"
        chunks.append(
            "<div class='bar-row filterable' "
            f"data-family='{escape(row['family'])}' data-region='{escape(row['region'])}' data-model='{escape(row['model'])}' "
            f"data-concurrency='{row['concurrency']}'>"
            f"<div class='bar-label'><span>{escape(label)}</span><span>{escape(shown)}</span></div>"
            "<div class='bar-track'>"
            f"<div class='bar-fill {escape(row['family'])}' style='width:{width:.1f}%'></div>"
            "</div></div>"
        )
    chunks.append("</div>")
    return "".join(chunks)


def render_reports_table(reports: list[dict]) -> str:
    rows = []
    for report in reports:
        attempts = sum(item["attempts"] for item in report["summary"])
        successes = sum(item["successes"] for item in report["summary"])
        failures = sum(item["failures"] for item in report["summary"])
        md_name = f"{report['name']}.md"
        json_name = f"{report['name']}.json"
        rows.append(
            "<tr>"
            f"<td><code>{escape(report['name'])}</code></td>"
            f"<td>{attempts}</td>"
            f"<td>{successes}</td>"
            f"<td>{failures}</td>"
            f"<td><a href='{escape(md_name)}'>{escape(md_name)}</a><br><a href='{escape(json_name)}'>{escape(json_name)}</a></td>"
            "</tr>"
        )
    return (
        "<div class='section'>"
        + title_html(2, "Reports", "dashboard에 포함된 benchmark JSON/Markdown report 파일 목록입니다.")
        + "<table><thead><tr>"
        "<th>Report</th><th>Attempts</th><th>Successes</th><th>Failures</th><th>Files</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def render_case_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        latency = f"{row['avg_latency']:.3f}s" if row["avg_latency"] is not None else "-"
        p95 = f"{row['p95_latency']:.3f}s" if row["p95_latency"] is not None else "-"
        p99 = f"{row['p99_latency']:.3f}s" if row["p99_latency"] is not None else "-"
        tokens = f"{row['avg_tokens']:.1f}" if row["avg_tokens"] is not None else "-"
        ttft = f"{row['avg_ttft']:.3f}s" if row["avg_ttft"] is not None else "-"
        tokens_per_second = (
            f"{row['avg_tokens_per_second']:.3f}" if row["avg_tokens_per_second"] is not None else "-"
        )
        post_ttft_tokens_per_second = (
            f"{row['avg_post_ttft_tokens_per_second']:.3f}"
            if row["avg_post_ttft_tokens_per_second"] is not None
            else "-"
        )
        body.append(
            "<tr class='filterable sortable-row' "
            f"data-family='{escape(row['family'])}' data-region='{escape(row['region'])}' data-model='{escape(row['model'])}' "
            f"data-concurrency='{row['concurrency']}' data-success-rate='{row['success_rate']:.6f}' "
            f"data-avg-latency='{row['avg_latency'] if row['avg_latency'] is not None else ''}' "
            f"data-p99-latency='{row['p99_latency'] if row['p99_latency'] is not None else ''}' "
            f"data-throughput='{row['avg_tokens_per_second'] if row['avg_tokens_per_second'] is not None else ''}'>"
            f"<td><code>{escape(row['report'])}</code></td>"
            f"<td><code>{escape(row['region'])}</code></td>"
            f"<td><span class='pill {escape(row['family'])}'>{escape(row['family'])}</span></td>"
            f"<td><code>{escape(row['model'])}</code></td>"
            f"<td><code>{escape(row['case_id'])}</code></td>"
            f"<td>{row['concurrency']}</td>"
            f"<td>{'yes' if row['streaming'] else 'no'}</td>"
            f"<td>{row['successes']}/{row['attempts']}</td>"
            f"<td>{row['success_rate']:.1f}%</td>"
            f"<td>{latency}</td>"
            f"<td>{p95}</td>"
            f"<td>{p99}</td>"
            f"<td>{ttft}</td>"
            f"<td>{tokens}</td>"
            f"<td>{tokens_per_second}</td>"
            f"<td>{post_ttft_tokens_per_second}</td>"
            "</tr>"
        )
    return (
        "<div class='section'>"
        + title_html(2, "Case Detail", "region, family, model, case, concurrency 조합별 상세 측정 결과입니다.")
        + "<table><thead><tr>"
        "<th>Report</th><th>Region</th><th>Family</th><th>Model</th><th>Case</th><th>Concurrency</th><th>Streaming</th><th>Success</th><th>Success Rate</th><th>Avg Latency</th><th>P95</th><th>P99</th><th>Avg TTFT</th><th>Avg Tokens</th><th>Avg E2E Output Tokens/sec</th><th>Avg Post-TTFT Output Tokens/sec</th>"
        "</tr></thead><tbody id='case-detail-body'>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def render_scatter(rows: list[dict], title: str) -> str:
    plotted = [row for row in rows if row["avg_tokens"] is not None and row["avg_latency"] is not None]
    if not plotted:
        return (
            "<div class='panel'>"
            + title_html(2, title, "생성 token 수와 latency의 관계를 scatter plot으로 보여줍니다.")
            + "<p class='muted'>latency와 token 데이터가 모두 있는 성공 row가 없습니다.</p></div>"
        )

    max_tokens = max(row["avg_tokens"] for row in plotted) or 1
    max_latency = max(row["avg_latency"] for row in plotted) or 1
    width = 520
    height = 320
    pad_left = 56
    pad_right = 18
    pad_top = 18
    pad_bottom = 38
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom

    points = []
    for row in plotted:
        x = pad_left + (row["avg_tokens"] / max_tokens) * plot_width
        y = pad_top + plot_height - (row["avg_latency"] / max_latency) * plot_height
        label = (
            f"{row['report']} / {row['family']} / {row['case_id']} / c{row['concurrency']} "
            f"(tokens={row['avg_tokens']:.1f}, latency={row['avg_latency']:.3f}s)"
        )
        points.append(
            "<g class='filterable' "
            f"data-family='{escape(row['family'])}' data-region='{escape(row['region'])}' data-model='{escape(row['model'])}' "
            f"data-concurrency='{row['concurrency']}'>"
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='6' class='dot {escape(row['family'])}'>"
            f"<title>{escape(label)}</title></circle>"
            f"<text x='{x + 8:.1f}' y='{y - 8:.1f}' class='dot-label'>{escape(row['family'])}:{escape(row['case_id'])}:c{row['concurrency']}</text>"
            "</g>"
        )

    ticks = []
    for i in range(5):
        frac = i / 4
        tx = pad_left + frac * plot_width
        token_value = max_tokens * frac
        ticks.append(
            f"<line x1='{tx:.1f}' y1='{pad_top + plot_height:.1f}' x2='{tx:.1f}' y2='{pad_top + plot_height + 6:.1f}' class='axis' />"
            f"<text x='{tx:.1f}' y='{height - 10:.1f}' text-anchor='middle' class='axis-label'>{token_value:.0f}</text>"
        )
    for i in range(5):
        frac = i / 4
        ty = pad_top + plot_height - frac * plot_height
        latency_value = max_latency * frac
        ticks.append(
            f"<line x1='{pad_left - 6:.1f}' y1='{ty:.1f}' x2='{pad_left:.1f}' y2='{ty:.1f}' class='axis' />"
            f"<text x='{pad_left - 10:.1f}' y='{ty + 4:.1f}' text-anchor='end' class='axis-label'>{latency_value:.1f}s</text>"
        )

    legend = (
        "<div class='legend'>"
        "<span><span class='legend-dot openai'></span>openai</span>"
        "<span><span class='legend-dot gemini'></span>gemini</span>"
        "<span><span class='legend-dot grok'></span>grok</span>"
        "<span><span class='legend-dot meta'></span>meta</span>"
        "<span><span class='legend-dot cohere'></span>cohere</span>"
        "</div>"
    )

    return (
        "<div class='panel'>"
        + title_html(2, title, "생성 token 수가 많아서 느린 것인지, 같은 token 규모에서도 느린 것인지 확인합니다.")
        + "<p class='muted'>X축은 평균 total tokens, Y축은 평균 latency입니다. 점에 마우스를 올리면 report/case 조합을 확인할 수 있습니다.</p>"
        f"{legend}"
        f"<svg viewBox='0 0 {width} {height}' class='scatter' role='img' aria-label='{escape(title)}'>"
        f"<line x1='{pad_left}' y1='{pad_top + plot_height}' x2='{width - pad_right}' y2='{pad_top + plot_height}' class='axis' />"
        f"<line x1='{pad_left}' y1='{pad_top}' x2='{pad_left}' y2='{pad_top + plot_height}' class='axis' />"
        + "".join(ticks)
        + "".join(points)
        + f"<text x='{pad_left + plot_width / 2:.1f}' y='{height - 2:.1f}' text-anchor='middle' class='axis-title'>Average Tokens</text>"
        + f"<text x='18' y='{pad_top + plot_height / 2:.1f}' transform='rotate(-90 18 {pad_top + plot_height / 2:.1f})' text-anchor='middle' class='axis-title'>Average Latency</text>"
        + "</svg></div>"
    )


def render_failures_table(failures: list[dict]) -> str:
    if not failures:
        return (
            "<div class='section'>"
            + title_html(2, "Failures", "실패한 개별 요청의 오류 유형, HTTP 상태, request id, 응답 본문 일부를 보여줍니다.")
            + "<p class='muted'>기록된 실패가 없습니다.</p></div>"
        )
    rows = []
    for failure in failures:
        status = failure["http_status"] if failure["http_status"] is not None else "-"
        request_id = failure["request_id"] or "-"
        error_type = failure["error_type"] or "-"
        body_preview = failure["response_body_preview"] or "-"
        rows.append(
            "<tr class='filterable' "
            f"data-family='{escape(failure['family'])}' data-region='{escape(failure.get('region') or '')}' data-model='{escape(failure['model'])}' "
            f"data-concurrency='{failure['concurrency']}'>"
            f"<td><code>{escape(failure['report'])}</code></td>"
            f"<td><code>{escape(failure.get('region') or '-')}</code></td>"
            f"<td><span class='pill {escape(failure['family'])}'>{escape(failure['family'])}</span></td>"
            f"<td><code>{escape(failure['model'])}</code></td>"
            f"<td><code>{escape(failure['case_id'])}</code></td>"
            f"<td>{failure['concurrency']}</td>"
            f"<td>{failure['iteration']}</td>"
            f"<td>{failure['latency']:.3f}s</td>"
            f"<td>{escape(error_type)}</td>"
            f"<td>{status}</td>"
            f"<td>{escape(request_id)}</td>"
            f"<td>{escape(failure['error'])}</td>"
            f"<td>{escape(body_preview)}</td>"
            "</tr>"
        )
    return (
        "<div class='section'>"
        + title_html(2, "Failures", "실패한 개별 요청의 오류 유형, HTTP 상태, request id, 응답 본문 일부를 보여줍니다.")
        + "<table><thead><tr>"
        "<th>Report</th><th>Region</th><th>Family</th><th>Model</th><th>Case</th><th>Concurrency</th><th>Iter</th><th>Latency</th><th>Type</th><th>HTTP</th><th>Request ID</th><th>Error</th><th>Body Preview</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def render_failure_summary_table(rows: list[dict]) -> str:
    if not rows:
        return (
            "<div class='section'>"
            + title_html(2, "Failure Summary", "오류를 report, family, model, HTTP status, error type별로 묶어 보여줍니다.")
            + "<p class='muted'>기록된 실패가 없습니다.</p></div>"
        )
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td><code>{escape(row['report'])}</code></td>"
            f"<td><span class='pill {escape(row['family'])}'>{escape(row['family'])}</span></td>"
            f"<td><code>{escape(row['model'])}</code></td>"
            f"<td>{escape(row['http_status'])}</td>"
            f"<td>{escape(row['error_type'])}</td>"
            f"<td>{row['count']}</td>"
            "</tr>"
        )
    return (
        "<div class='section'>"
        + title_html(2, "Failure Summary", "오류를 report, family, model, HTTP status, error type별로 묶어 보여줍니다.")
        + "<table><thead><tr>"
        "<th>Report</th><th>Family</th><th>Model</th><th>HTTP</th><th>Type</th><th>Count</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def render_skipped_table(rows: list[dict]) -> str:
    if not rows:
        return (
            "<div class='section'>"
            + title_html(2, "Skipped Region/Model Combinations", "catalog 기준으로 해당 region에서 지원되지 않아 실행하지 않은 model/region 조합입니다.")
            + "<p class='muted'>skip된 미지원 region/model 조합이 없습니다.</p></div>"
        )
    body = []
    for row in rows:
        body.append(
            "<tr class='filterable' "
            f"data-family='{escape(row['family'])}' data-region='{escape(row['region'])}' data-model='{escape(row['model'])}' "
            "data-concurrency=''>"
            f"<td><code>{escape(row['report'])}</code></td>"
            f"<td><code>{escape(row['region'])}</code></td>"
            f"<td><span class='pill {escape(row['family'])}'>{escape(row['family'])}</span></td>"
            f"<td><code>{escape(row['model'])}</code></td>"
            f"<td>지원 리전 아님</td>"
            f"<td>{escape(row['reason'])}</td>"
            "</tr>"
        )
    return (
        "<div class='section'>"
        + title_html(2, "Skipped Region/Model Combinations", "catalog 기준으로 해당 region에서 지원되지 않아 실행하지 않은 model/region 조합입니다.")
        + "<p class='muted'>미지원 model/region 조합은 runtime failure와 구분되도록 별도 표로 표시합니다.</p>"
        "<table><thead><tr><th>Report</th><th>Region</th><th>Family</th><th>Model</th><th>Status</th><th>Reason</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def render_filter_controls(options: dict[str, list[str]]) -> str:
    def option_tags(values: list[str]) -> str:
        return "<option value=''>All</option>" + "".join(
            f"<option value='{escape(value)}'>{escape(value)}</option>" for value in values
        )

    return (
        "<div class='filters' aria-label='Dashboard filters'>"
        "<div class='field'><label for='family-filter'>Family</label>"
        f"<select id='family-filter' data-filter='family'>{option_tags(options['families'])}</select></div>"
        "<div class='field'><label for='region-filter'>Region</label>"
        f"<select id='region-filter' data-filter='region'>{option_tags(options['regions'])}</select></div>"
        "<div class='field'><label for='model-filter'>Model</label>"
        f"<select id='model-filter' data-filter='model'>{option_tags(options['models'])}</select></div>"
        "<div class='field'><label for='concurrency-filter'>Concurrency</label>"
        f"<select id='concurrency-filter' data-filter='concurrency'>{option_tags(options['concurrency'])}</select></div>"
        "<div class='field'><label for='case-sort'>Case Table Sort</label>"
        "<select id='case-sort'>"
        "<option value='successRate:asc'>Success Rate Asc</option>"
        "<option value='avgLatency:desc'>Avg Latency Desc</option>"
        "<option value='avgLatency:asc'>Avg Latency Asc</option>"
        "<option value='p99Latency:desc'>P99 Latency Desc</option>"
        "<option value='throughput:desc'>E2E Output Tokens/sec Desc</option>"
        "</select></div>"
        "</div>"
    )


def render_raw_debug_details(
    reports: list[dict],
    case_rows: list[dict],
    family_rows: list[dict],
    failure_summary: list[dict],
    failures: list[dict],
) -> str:
    return (
        "<div class='section'>"
        "<details class='debug-details'>"
        "<summary>Raw Data / Debug Details</summary>"
        "<p class='muted'>운영 디버깅을 위해 result file, family-level chart, case-level metric, failure diagnostic을 접어 둔 영역입니다.</p>"
        "<div class='chart-grid'>"
        f"{render_bars(family_rows, 'success_rate', 'Success Rate by Report / Family', '%')}"
        f"{render_bars(family_rows, 'avg_latency', 'Average Latency by Report / Family', 's')}"
        "</div>"
        "<div class='section'>"
        + title_html(2, "Case-Level Charts", "report/family/case 조합별로 실패나 지연이 집중되는 위치를 확인합니다.")
        + "<p class='muted'>각 bar는 report/family/case 조합 하나를 나타내며, 실패나 slowdown이 어디에 몰리는지 확인할 수 있습니다.</p>"
        "</div>"
        "<div class='chart-grid'>"
        f"{render_case_bars(case_rows, 'success_rate', 'Success Rate by Case', '%')}"
        f"{render_case_bars(case_rows, 'avg_latency', 'Average Latency by Case', 's')}"
        "</div>"
        "<div class='chart-grid'>"
        f"{render_case_bars(case_rows, 'avg_tokens', 'Average Tokens by Case', '')}"
        f"{render_case_bars(case_rows, 'avg_tokens_per_second', 'Average E2E Output Tokens/sec by Case', '')}"
        "</div>"
        f"{render_reports_table(reports)}"
        f"{render_case_table(case_rows)}"
        f"{render_failure_summary_table(failure_summary)}"
        f"{render_failures_table(failures)}"
        "</details>"
        "</div>"
    )


def collect_overall_metrics(reports: list[dict], case_rows: list[dict]) -> dict[str, float | int | None]:
    latencies = []
    throughputs = []
    tokens = []
    ttfts = []
    for report in reports:
        for item in report["results"]:
            if item.get("error"):
                continue
            if isinstance(item.get("latency_seconds"), (int, float)):
                latencies.append(float(item["latency_seconds"]))
            if isinstance(item.get("output_tokens_per_second"), (int, float)):
                throughputs.append(float(item["output_tokens_per_second"]))
            if isinstance(item.get("total_tokens"), (int, float)):
                tokens.append(float(item["total_tokens"]))
            if isinstance(item.get("ttft_seconds"), (int, float)):
                ttfts.append(float(item["ttft_seconds"]))
    total_attempts = sum(row["attempts"] for row in case_rows)
    total_successes = sum(row["successes"] for row in case_rows)
    total_failures = sum(row["failures"] for row in case_rows)
    return {
        "reports": len(reports),
        "attempts": total_attempts,
        "successes": total_successes,
        "failures": total_failures,
        "success_rate": (total_successes / total_attempts * 100) if total_attempts else 0.0,
        "avg_latency": average(latencies),
        "p95_latency": percentile_value(latencies, 95),
        "avg_tokens_per_second": average(throughputs),
        "avg_tokens": average(tokens),
        "avg_ttft": average(ttfts),
    }


def javascript() -> str:
    return """
function selectedFilters() {
  return {
    family: document.querySelector('[data-filter="family"]').value,
    region: document.querySelector('[data-filter="region"]').value,
    model: document.querySelector('[data-filter="model"]').value,
    concurrency: document.querySelector('[data-filter="concurrency"]').value
  };
}
function matchesFilters(element, filters) {
  return (!filters.family || element.dataset.family === filters.family)
    && (!filters.region || element.dataset.region === filters.region)
    && (!filters.model || element.dataset.model === filters.model)
    && (!filters.concurrency || element.dataset.concurrency === filters.concurrency);
}
function activateTab(button) {
  const target = button.dataset.tabTarget;
  document.querySelectorAll('.tab-button').forEach((item) => {
    item.classList.toggle('active', item === button);
  });
  document.querySelectorAll('.tab-panel').forEach((panel) => {
    panel.classList.toggle('hidden', panel.id !== target);
  });
}
function applyFilters() {
  const filters = selectedFilters();
  document.querySelectorAll('.filterable').forEach((element) => {
    element.classList.toggle('filtered-out', !matchesFilters(element, filters));
  });
}
function numericValue(row, key) {
  const raw = row.dataset[key] || '';
  if (raw === '') {
    return Number.NEGATIVE_INFINITY;
  }
  return Number(raw);
}
function sortCaseRows() {
  const select = document.getElementById('case-sort');
  const body = document.getElementById('case-detail-body');
  if (!select || !body) {
    return;
  }
  const [key, direction] = select.value.split(':');
  const rows = Array.from(body.querySelectorAll('.sortable-row'));
  rows.sort((left, right) => {
    const delta = numericValue(left, key) - numericValue(right, key);
    return direction === 'asc' ? delta : -delta;
  });
  rows.forEach((row) => body.appendChild(row));
}
document.querySelectorAll('[data-filter]').forEach((control) => {
  control.addEventListener('change', applyFilters);
});
const sortControl = document.getElementById('case-sort');
if (sortControl) {
  sortControl.addEventListener('change', sortCaseRows);
}
document.querySelectorAll('.tab-button').forEach((button) => {
  button.addEventListener('click', () => activateTab(button));
});
sortCaseRows();
applyFilters();
"""


def render_html(reports: list[dict], suite_summaries: list[dict] | None = None) -> str:
    suite_summaries = suite_summaries or []
    family_rows = aggregate_family_metrics(reports)
    case_rows = aggregate_case_metrics(reports)
    workload_reports = focus_reports(suite_summaries, reports)
    workload_rows = aggregate_case_metrics(workload_reports)
    failures = collect_failures(reports)
    failure_summary = collect_failure_summary(failures)
    skipped_rows = collect_skipped_combinations(reports)
    filter_options = collect_filter_options(case_rows)
    overall = collect_overall_metrics(reports, case_rows)

    cards = [
        ("JSON reports", str(overall["reports"]), "dashboard에 로드된 benchmark result set 수"),
        ("Success rate", format_percent(overall["success_rate"]), f"{overall['successes']}/{overall['attempts']} 요청 성공"),
        ("Avg latency", format_latency(overall["avg_latency"]), "성공 요청의 평균 end-to-end 응답 시간"),
        ("P95 latency", format_latency(overall["p95_latency"]), "성공 요청 중 95%가 이 시간 안에 완료됨"),
        ("Avg tok/sec", format_number(overall["avg_tokens_per_second"]), "초당 평균 output token 수"),
        ("Avg TTFT", format_latency(overall["avg_ttft"]), "streaming run에서 첫 토큰까지 걸린 평균 시간"),
        ("Avg tokens", format_number(overall["avg_tokens"]), "성공 요청당 평균 total token 수"),
    ]
    card_html = "".join(
        "<div class='card'>"
        f"<div class='muted'>{escape(title)}</div>"
        f"<div class='metric'>{escape(value)}</div>"
        f"<div class='muted'>{escape(note)}</div>"
        "</div>"
        for title, value, note in cards
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GenAI Benchmark Dashboard</title>
  <style>{css()}</style>
</head>
<body>
  <div class="page">
    {title_html(1, 'GenAI Benchmark Dashboard', 'source runner가 여러 target region의 OCI Generative AI 모델 endpoint로 같은 workload를 보내고, 동시성별 성능과 실패/skip 조합을 보여주는 정적 dashboard입니다.')}
    <p class="lede">이 테스트는 각 source runner VM에서 같은 Chat/NL2SQL workload를 target region의 대표 모델에 설정된 동시성 단계로 호출해 수행합니다. 결과는 end-to-end latency, streaming TTFT, success rate, output tok/sec, 부하 증가 시 지연 배율, 지원되지 않는 region/model skip 조합을 쉽게 비교하도록 보여줍니다.</p>
    <div class="cards">{card_html}</div>
    {render_load_summary(case_rows)}
    {render_c50_ranking(case_rows)}
    {render_load_sensitivity(case_rows)}
    {render_runner_matrix(suite_summaries, reports)}
    {render_skipped_table(skipped_rows)}
    {render_workload_details(workload_rows, workload_reports)}
    {render_filter_controls(filter_options)}
    <div class="section">
      {title_html(2, 'Latency vs Token Volume', '응답이 token을 많이 생성해서 느린 것인지, 비슷한 token 수에서도 상대적으로 느린 것인지 확인합니다.')}
      <p class="muted">응답이 느린 이유가 생성 token 수 때문인지, 같은 token 규모의 다른 결과보다 느린 것인지 비교합니다.</p>
    </div>
    <div class="chart-grid">
      {render_scatter(case_rows, 'Latency vs Generated Tokens')}
      <div class='panel'>
        {title_html(2, 'How to Read', 'scatter plot의 위치를 해석하는 방법입니다. 오른쪽은 token 수가 많고, 위쪽은 latency가 긴 결과입니다.')}
        <p class="muted">오른쪽에 있을수록 더 많은 token을 생성했고, 위쪽에 있을수록 더 오래 걸렸습니다. token 수가 비슷한 두 점에서는 더 위에 있는 점이 상대적으로 느린 결과입니다.</p>
      </div>
    </div>
    {render_raw_debug_details(reports, case_rows, family_rows, failure_summary, failures)}
    <div class="section">
      <div class="muted">Generated at {escape(generated_at)}</div>
    </div>
  </div>
  <script>{javascript()}</script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    output = Path(args.output)
    reports = load_reports(runs_dir)
    if not reports:
        raise SystemExit(f"No JSON reports found in {runs_dir}")
    output.write_text(render_html(reports, load_suite_summaries(runs_dir)), encoding="utf-8")
    print(f"Wrote dashboard to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
