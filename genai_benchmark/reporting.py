from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence


def render_markdown(
    payload: Dict[str, Any],
    args: Any,
    prompt_path: Path,
    regions: Sequence[str],
) -> str:
    selected_models = payload["selected_models"]
    lines = [
        "# GenAI Benchmark Report",
        "",
        f"- Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Source Label: `{args.source_label or 'unspecified'}`",
        f"- Regions Requested: {', '.join(f'`{region}`' for region in regions)}",
        f"- Profile: `{args.profile}`",
        f"- Prompt file: `{prompt_path}`",
        f"- Repeats: `{args.repeats}`",
        f"- Concurrency: `{args.concurrency}`",
        "",
        "## Selected Models",
        "",
        "| Family | Model | Default | Experimental | Catalog Regions |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in selected_models:
        lines.append(
            "| {family} | {model_id} | {default_selected} | {experimental} | {regions} |".format(
                family=item["family"],
                model_id=item["model_id"],
                default_selected="yes" if item["default_selected"] else "no",
                experimental="yes" if item["experimental"] else "no",
                regions=", ".join(item["regions"]),
            )
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Region | Family | Model | Case | Success | Avg Latency (s) | P95 Latency (s) | Avg Tokens |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in payload["summary"]:
        lines.append(
            "| {region} | {family} | {model} | {case_id} | {successes}/{attempts} | {avg_latency_seconds} | {p95_latency_seconds} | {avg_total_tokens} |".format(
                region=item["region"],
                family=item["family"],
                model=item["model"],
                case_id=item["case_id"],
                successes=item["successes"],
                attempts=item["attempts"],
                avg_latency_seconds=item["avg_latency_seconds"] or "-",
                p95_latency_seconds=item["p95_latency_seconds"] or "-",
                avg_total_tokens=item["avg_total_tokens"] or "-",
            )
        )

    if payload["skipped"]:
        lines.extend(
            [
                "",
                "## Skipped Combinations",
                "",
                "| Region | Family | Model | Reason |",
                "| --- | --- | --- | --- |",
            ]
        )
        for item in payload["skipped"]:
            lines.append(
                "| {region} | {family} | {model} | {reason} |".format(
                    region=item["region"],
                    family=item["family"],
                    model=item["model"],
                    reason=item["reason"],
                )
            )

    return "\n".join(lines) + "\n"

