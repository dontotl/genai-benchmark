from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence


def format_optional(value: Any, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value}{suffix}"


def markdown_cell(value: Any, max_chars: int | None = None) -> str:
    text = "-" if value is None or value == "" else str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    if max_chars is not None:
        text = text[:max_chars]
    return text


def render_markdown(
    payload: Dict[str, Any],
    args: Any,
    prompt_path: Path,
    regions: Sequence[str],
) -> str:
    selected_models = payload["selected_models"]
    concurrency_levels = getattr(args, "resolved_concurrency_levels", [getattr(args, "concurrency", 1)])
    lines = [
        "# GenAI Benchmark Report",
        "",
        f"- Timestamp (UTC): {payload.get('generated_at') or datetime.now(timezone.utc).isoformat()}",
        f"- Schema Version: `{payload.get('schema_version', 'legacy')}`",
        f"- Source Label: `{args.source_label or 'unspecified'}`",
        f"- Regions Requested: {', '.join(f'`{region}`' for region in regions)}",
        f"- Profile: `{args.profile}`",
        f"- Prompt file: `{prompt_path}`",
        f"- Repeats: `{args.repeats}`",
        f"- Concurrency Levels: `{', '.join(str(level) for level in concurrency_levels)}`",
        f"- Streaming: `{'yes' if getattr(args, 'streaming', False) else 'no'}`",
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
            "| Region | Family | Model | Case | Concurrency | Streaming | Success | Avg Latency (s) | P95 Latency (s) | P99 Latency (s) | Avg TTFT (s) | Avg Tokens | Avg E2E Output Tokens/sec | Avg Post-TTFT Output Tokens/sec |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in payload["summary"]:
        lines.append(
            "| {region} | {family} | {model} | {case_id} | {concurrency} | {streaming} | {successes}/{attempts} | {avg_latency_seconds} | {p95_latency_seconds} | {p99_latency_seconds} | {avg_ttft_seconds} | {avg_total_tokens} | {avg_output_tokens_per_second} | {avg_post_ttft_output_tokens_per_second} |".format(
                region=item["region"],
                family=item["family"],
                model=item["model"],
                case_id=item["case_id"],
                concurrency=item.get("concurrency", 1),
                streaming="yes" if item.get("streaming") else "no",
                successes=item["successes"],
                attempts=item["attempts"],
                avg_latency_seconds=format_optional(item.get("avg_latency_seconds")),
                p95_latency_seconds=format_optional(item.get("p95_latency_seconds")),
                p99_latency_seconds=format_optional(item.get("p99_latency_seconds")),
                avg_ttft_seconds=format_optional(item.get("avg_ttft_seconds")),
                avg_total_tokens=format_optional(item.get("avg_total_tokens")),
                avg_output_tokens_per_second=format_optional(item.get("avg_output_tokens_per_second")),
                avg_post_ttft_output_tokens_per_second=format_optional(
                    item.get("avg_post_ttft_output_tokens_per_second")
                ),
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

    failures = [item for item in payload["results"] if item.get("error")]
    if failures:
        lines.extend(
            [
                "",
                "## Failure Details",
                "",
                "| Region | Family | Model | Case | Concurrency | Iter | Type | HTTP | Request ID | Error | Body Preview |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for item in failures:
            lines.append(
                "| {region} | {family} | {model} | {case_id} | {concurrency} | {iteration} | {error_type} | {http_status} | {request_id} | {error} | {body} |".format(
                    region=item["region"],
                    family=item["family"],
                    model=item["model"],
                    case_id=item["case_id"],
                    concurrency=item.get("concurrency", 1),
                    iteration=item["iteration"],
                    error_type=markdown_cell(item.get("error_type")),
                    http_status=markdown_cell(item.get("http_status")),
                    request_id=markdown_cell(item.get("request_id")),
                    error=markdown_cell(item.get("error"), 240),
                    body=markdown_cell(item.get("response_body_preview"), 240),
                )
            )

    return "\n".join(lines) + "\n"
