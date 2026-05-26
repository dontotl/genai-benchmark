from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from .dashboard import load_reports, load_suite_summaries, render_html


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
DOCS_DIR = PROJECT_ROOT / "docs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish GitHub-friendly docs artifacts from benchmark runs.")
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    parser.add_argument("--docs-dir", default=str(DOCS_DIR))
    return parser.parse_args()


def choose_focus_report(reports: list[dict]) -> dict:
    reports_with_timestamps = [report for report in reports if report.get("generated_at")]
    if reports_with_timestamps:
        return max(reports_with_timestamps, key=lambda report: report["generated_at"])
    priority = [
        "cross-region-baseline-r3",
        "cross-region-smoke-r1",
        "baseline-osaka-r3",
        "smoke-osaka-escalated",
    ]
    by_name = {report["name"]: report for report in reports}
    for name in priority:
        if name in by_name:
            return by_name[name]
    return reports[-1]


def summarize_focus(report: dict) -> dict:
    by_region_family: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"attempts": 0, "successes": 0, "failures": 0, "latencies": []}
    )
    total_attempts = 0
    total_successes = 0
    total_failures = 0
    for item in report["summary"]:
        key = (item["region"], item["family"])
        bucket = by_region_family[key]
        bucket["attempts"] += item["attempts"]
        bucket["successes"] += item["successes"]
        bucket["failures"] += item["failures"]
        total_attempts += item["attempts"]
        total_successes += item["successes"]
        total_failures += item["failures"]
        if item["avg_latency_seconds"] is not None:
            bucket["latencies"].append(item["avg_latency_seconds"])

    rows = []
    for (region, family), bucket in sorted(by_region_family.items()):
        avg_latency = (
            sum(bucket["latencies"]) / len(bucket["latencies"]) if bucket["latencies"] else None
        )
        rows.append(
            {
                "region": region,
                "family": family,
                "attempts": bucket["attempts"],
                "successes": bucket["successes"],
                "failures": bucket["failures"],
                "success_rate": (bucket["successes"] / bucket["attempts"] * 100)
                if bucket["attempts"]
                else 0.0,
                "avg_latency": avg_latency,
            }
        )
    return {
        "rows": rows,
        "total_attempts": total_attempts,
        "total_successes": total_successes,
        "total_failures": total_failures,
    }


def render_preview_svg(report: dict, focus: dict) -> str:
    width = 1200
    height = 720
    success_rate = (focus["total_successes"] / focus["total_attempts"] * 100) if focus["total_attempts"] else 0.0
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    cards = [
        ("Focus Report", report["name"]),
        ("Attempts", str(focus["total_attempts"])),
        ("Successes", str(focus["total_successes"])),
        ("Failures", str(focus["total_failures"])),
        ("Success Rate", f"{success_rate:.1f}%"),
    ]

    card_width = 208
    card_gap = 16
    card_y = 120
    card_height = 108
    card_xs = [56 + i * (card_width + card_gap) for i in range(len(cards))]
    card_svg = []
    for (title, value), x in zip(cards, card_xs):
        card_svg.append(
            f"""
            <g>
              <rect x="{x}" y="{card_y}" width="{card_width}" height="{card_height}" rx="18" fill="#fffdf9" stroke="#d9cfbf"/>
              <text x="{x + 18}" y="{card_y + 30}" fill="#6b635b" font-size="18" font-family="Arial, sans-serif">{escape(title)}</text>
              <text x="{x + 18}" y="{card_y + 72}" fill="#1d1b18" font-size="34" font-weight="700" font-family="Arial, sans-serif">{escape(value)}</text>
            </g>
            """
        )

    row_y_start = 310
    row_height = 60
    rows_svg = [
        '<rect x="56" y="280" width="1088" height="46" rx="12" fill="#f1eadf" />',
        '<text x="76" y="309" fill="#4d463f" font-size="18" font-weight="700" font-family="Arial, sans-serif">Region</text>',
        '<text x="300" y="309" fill="#4d463f" font-size="18" font-weight="700" font-family="Arial, sans-serif">Family</text>',
        '<text x="460" y="309" fill="#4d463f" font-size="18" font-weight="700" font-family="Arial, sans-serif">Success</text>',
        '<text x="650" y="309" fill="#4d463f" font-size="18" font-weight="700" font-family="Arial, sans-serif">Rate</text>',
        '<text x="820" y="309" fill="#4d463f" font-size="18" font-weight="700" font-family="Arial, sans-serif">Avg Latency</text>',
    ]
    family_fill = {"openai": "#ccfbf1", "gemini": "#fef3c7", "grok": "#dbeafe", "meta": "#ede9fe"}
    family_text = {"openai": "#115e59", "gemini": "#92400e", "grok": "#1d4ed8", "meta": "#6d28d9"}

    for index, row in enumerate(focus["rows"]):
        y = row_y_start + index * row_height
        latency = f"{row['avg_latency']:.3f}s" if row["avg_latency"] is not None else "-"
        rows_svg.append(
            f"""
            <g>
              <rect x="56" y="{y}" width="1088" height="46" rx="12" fill="#fffdf9" stroke="#e5ddcf"/>
              <text x="76" y="{y + 29}" fill="#1d1b18" font-size="17" font-family="Arial, sans-serif">{escape(row['region'])}</text>
              <rect x="292" y="{y + 9}" width="98" height="28" rx="14" fill="{family_fill[row['family']]}"/>
              <text x="341" y="{y + 28}" text-anchor="middle" fill="{family_text[row['family']]}" font-size="15" font-weight="700" font-family="Arial, sans-serif">{escape(row['family'])}</text>
              <text x="460" y="{y + 29}" fill="#1d1b18" font-size="17" font-family="Arial, sans-serif">{row['successes']}/{row['attempts']}</text>
              <text x="650" y="{y + 29}" fill="#1d1b18" font-size="17" font-family="Arial, sans-serif">{row['success_rate']:.1f}%</text>
              <text x="820" y="{y + 29}" fill="#1d1b18" font-size="17" font-family="Arial, sans-serif">{escape(latency)}</text>
            </g>
            """
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Benchmark preview">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f7f1e8" />
      <stop offset="100%" stop-color="#fdfbf8" />
    </linearGradient>
  </defs>
  <rect width="{width}" height="{height}" fill="url(#bg)" />
  <text x="56" y="62" fill="#1d1b18" font-size="42" font-weight="700" font-family="Arial, sans-serif">GenAI Benchmark Dashboard Preview</text>
  <text x="56" y="94" fill="#5a534b" font-size="20" font-family="Arial, sans-serif">Latest focus report: {escape(report['name'])} | Generated {escape(generated_at)}</text>
  {''.join(card_svg)}
  {''.join(rows_svg)}
  <text x="56" y="686" fill="#5a534b" font-size="18" font-family="Arial, sans-serif">OpenAI is currently stable across Osaka, Chicago, and Frankfurt. Gemini remains region- and prompt-sensitive.</text>
</svg>
"""


def write_docs(reports: list[dict], docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    focus_report = choose_focus_report(reports)
    focus = summarize_focus(focus_report)
    runs_dir = reports[0]["path"].parent if reports and reports[0].get("path") else RUNS_DIR
    suite_summaries = load_suite_summaries(runs_dir)

    dashboard_html = render_html(reports, suite_summaries)
    (docs_dir / "dashboard.html").write_text(dashboard_html, encoding="utf-8")
    (docs_dir / "index.html").write_text(dashboard_html, encoding="utf-8")
    (docs_dir / "dashboard-preview.svg").write_text(
        render_preview_svg(focus_report, focus),
        encoding="utf-8",
    )
    (docs_dir / ".nojekyll").write_text("", encoding="utf-8")


def main() -> int:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    docs_dir = Path(args.docs_dir)
    reports = load_reports(runs_dir)
    if not reports:
        raise SystemExit(f"No JSON reports found in {runs_dir}")
    write_docs(reports, docs_dir)
    print(f"Wrote docs site to {docs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
