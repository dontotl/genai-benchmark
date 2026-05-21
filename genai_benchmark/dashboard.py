from __future__ import annotations

import argparse
import json
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
                "generated_at": data.get("generated_at", ""),
            }
        )
    return reports


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
        "regions": sorted({row["region"] for row in rows if row.get("region")}),
        "models": sorted({row["model"] for row in rows if row.get("model")}),
        "concurrency": sorted({str(row["concurrency"]) for row in rows}, key=lambda value: int(value)),
    }


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
@media (max-width: 900px) {
  .chart-grid { grid-template-columns: 1fr; }
}
"""


def render_bars(rows: list[dict], metric: str, title: str, value_suffix: str) -> str:
    max_value = max((row[metric] or 0) for row in rows) if rows else 0
    chunks = [f"<div class='panel'><h2>{escape(title)}</h2>"]
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
    chunks = [f"<div class='panel'><h2>{escape(title)}</h2>"]
    for row in rows:
        raw_value = row[metric] or 0
        width = (raw_value / max_value * 100) if max_value else 0
        label = f"{row['report']} / {row['family']} / {row['case_id']} / c{row['concurrency']}"
        shown = f"{raw_value:.1f}{value_suffix}" if isinstance(raw_value, float) else f"{raw_value}{value_suffix}"
        chunks.append(
            "<div class='bar-row filterable' "
            f"data-region='{escape(row['region'])}' data-model='{escape(row['model'])}' "
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
        "<div class='section'><h2>Reports</h2><table><thead><tr>"
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
            f"data-region='{escape(row['region'])}' data-model='{escape(row['model'])}' "
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
        "<div class='section'><h2>Case Detail</h2><table><thead><tr>"
        "<th>Report</th><th>Region</th><th>Family</th><th>Model</th><th>Case</th><th>Concurrency</th><th>Streaming</th><th>Success</th><th>Success Rate</th><th>Avg Latency</th><th>P95</th><th>P99</th><th>Avg TTFT</th><th>Avg Tokens</th><th>Avg E2E Output Tokens/sec</th><th>Avg Post-TTFT Output Tokens/sec</th>"
        "</tr></thead><tbody id='case-detail-body'>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def render_scatter(rows: list[dict], title: str) -> str:
    plotted = [row for row in rows if row["avg_tokens"] is not None and row["avg_latency"] is not None]
    if not plotted:
        return (
            "<div class='panel'><h2>"
            + escape(title)
            + "</h2><p class='muted'>No successful rows with both latency and token data.</p></div>"
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
            f"data-region='{escape(row['region'])}' data-model='{escape(row['model'])}' "
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
        "</div>"
    )

    return (
        "<div class='panel'>"
        f"<h2>{escape(title)}</h2>"
        "<p class='muted'>X axis is average total tokens. Y axis is average latency. Hover a point to see the exact report/case tuple.</p>"
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
        return "<div class='section'><h2>Failures</h2><p class='muted'>No failures recorded.</p></div>"
    rows = []
    for failure in failures:
        status = failure["http_status"] if failure["http_status"] is not None else "-"
        request_id = failure["request_id"] or "-"
        error_type = failure["error_type"] or "-"
        body_preview = failure["response_body_preview"] or "-"
        rows.append(
            "<tr class='filterable' "
            f"data-region='{escape(failure.get('region') or '')}' data-model='{escape(failure['model'])}' "
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
        "<div class='section'><h2>Failures</h2><table><thead><tr>"
        "<th>Report</th><th>Region</th><th>Family</th><th>Model</th><th>Case</th><th>Concurrency</th><th>Iter</th><th>Latency</th><th>Type</th><th>HTTP</th><th>Request ID</th><th>Error</th><th>Body Preview</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def render_failure_summary_table(rows: list[dict]) -> str:
    if not rows:
        return "<div class='section'><h2>Failure Summary</h2><p class='muted'>No failures recorded.</p></div>"
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
        "<div class='section'><h2>Failure Summary</h2><table><thead><tr>"
        "<th>Report</th><th>Family</th><th>Model</th><th>HTTP</th><th>Type</th><th>Count</th>"
        "</tr></thead><tbody>"
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


def javascript() -> str:
    return """
function selectedFilters() {
  return {
    region: document.querySelector('[data-filter="region"]').value,
    model: document.querySelector('[data-filter="model"]').value,
    concurrency: document.querySelector('[data-filter="concurrency"]').value
  };
}
function matchesFilters(element, filters) {
  return (!filters.region || element.dataset.region === filters.region)
    && (!filters.model || element.dataset.model === filters.model)
    && (!filters.concurrency || element.dataset.concurrency === filters.concurrency);
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
sortCaseRows();
applyFilters();
"""


def render_html(reports: list[dict]) -> str:
    family_rows = aggregate_family_metrics(reports)
    case_rows = aggregate_case_metrics(reports)
    failures = collect_failures(reports)
    failure_summary = collect_failure_summary(failures)
    filter_options = collect_filter_options(case_rows)
    total_attempts = sum(row["attempts"] for row in family_rows)
    total_successes = sum(row["successes"] for row in family_rows)
    total_failures = sum(row["failures"] for row in family_rows)
    success_rate = (total_successes / total_attempts * 100) if total_attempts else 0.0

    cards = [
        ("JSON reports", str(len(reports)), "Loaded benchmark result sets"),
        ("Total attempts", str(total_attempts), "Across all current report files"),
        ("Total successes", str(total_successes), f"Overall success rate {success_rate:.1f}%"),
        ("Total failures", str(total_failures), "Failures are broken out below"),
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
    <h1>GenAI Benchmark Dashboard</h1>
    <p class="lede">Generated from current JSON results in <code>runs/</code>. This view is static and self-contained, so you can open it directly in a browser.</p>
    <div class="cards">{card_html}</div>
    {render_filter_controls(filter_options)}
    <div class="chart-grid">
      {render_bars(family_rows, 'success_rate', 'Success Rate by Report / Family', '%')}
      {render_bars(family_rows, 'avg_latency', 'Average Latency by Report / Family', 's')}
    </div>
    <div class="section">
      <h2>Case-Level Charts</h2>
      <p class="muted">Each bar below represents one report/family/case combination so you can see where failures or slowdowns concentrate.</p>
    </div>
    <div class="chart-grid">
      {render_case_bars(case_rows, 'success_rate', 'Success Rate by Case', '%')}
      {render_case_bars(case_rows, 'avg_latency', 'Average Latency by Case', 's')}
    </div>
    <div class="chart-grid">
      {render_case_bars(case_rows, 'avg_tokens', 'Average Tokens by Case', '')}
      {render_case_bars(case_rows, 'avg_tokens_per_second', 'Average E2E Output Tokens/sec by Case', '')}
    </div>
    <div class="section">
      <h2>Efficiency View</h2>
      <p class="muted">Throughput uses end-to-end latency, so it is useful for client-observed efficiency but does not separate TTFT from generation speed.</p>
    </div>
    <div class="chart-grid">
      {render_scatter(case_rows, 'Latency vs Tokens Scatter')}
      <div class='panel'>
        <h2>How to Read</h2>
        <p class="muted">Points farther right produced more tokens. Points higher took longer. A model appearing much higher than peers at a similar token count is a likely latency outlier.</p>
      </div>
    </div>
    {render_reports_table(reports)}
    {render_case_table(case_rows)}
    {render_failure_summary_table(failure_summary)}
    {render_failures_table(failures)}
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
    output.write_text(render_html(reports), encoding="utf-8")
    print(f"Wrote dashboard to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
