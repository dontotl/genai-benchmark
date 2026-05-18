from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .catalog import DEFAULT_FAMILIES, format_model_listing, get_family_names, resolve_models
from .reporting import render_markdown
from .runner import aggregate_results, load_cases, plan_execution_matrix, run_benchmark


if sys.version_info < (3, 10):
    raise SystemExit(
        "benchmark.py requires Python 3.10 or newer. "
        "Run it with python3.11 in this environment."
    )


DEFAULT_REGION = "ap-osaka-1"
DEFAULT_PROFILE = "DEFAULT"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark OCI Generative AI chat models through the OpenAI-compatible endpoint."
    )
    parser.add_argument(
        "--prompts",
        default="prompts/sample_prompts.jsonl",
        help="JSONL file containing benchmark cases.",
    )
    parser.add_argument(
        "--family",
        action="append",
        dest="families",
        help=(
            "Model family to include. Repeat this flag to select multiple families. "
            f"Defaults to {', '.join(DEFAULT_FAMILIES)}."
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model to benchmark. Repeat this flag to benchmark multiple models.",
    )
    parser.add_argument(
        "--region",
        action="append",
        dest="regions",
        help="Target region. Repeat this flag to benchmark multiple regions.",
    )
    parser.add_argument("--profile", default=os.getenv("OCI_PROFILE", DEFAULT_PROFILE))
    parser.add_argument("--compartment-id", default=os.getenv("OCI_COMPARTMENT_ID", ""))
    parser.add_argument("--repeats", type=int, default=3, help="Number of runs per prompt and model.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent requests per model/region/case.",
    )
    parser.add_argument(
        "--concurrency-levels",
        default="",
        help=(
            "Comma-separated concurrency levels for a ramp run, e.g. 1,5,10. "
            "When set, this overrides --concurrency."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--source-label",
        default=os.getenv("OCI_APP_REGION_LABEL", ""),
        help="Optional label describing where the benchmark client is running from.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs",
        help="Directory where JSON and Markdown reports will be written.",
    )
    parser.add_argument(
        "--report-name",
        default="",
        help="Optional basename for output files. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--include-experimental",
        action="store_true",
        help="Include default experimental families when they are selected with --family.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print supported model catalog and exit.",
    )
    parser.add_argument(
        "--list-families",
        action="store_true",
        help="Print supported families and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print the planned matrix without calling the endpoint.",
    )
    return parser.parse_args()


def resolve_regions(args: argparse.Namespace) -> list[str]:
    regions = args.regions or [os.getenv("OCI_GENAI_REGION", DEFAULT_REGION)]
    ordered: list[str] = []
    seen = set()
    for region in regions:
        if region and region not in seen:
            ordered.append(region)
            seen.add(region)
    return ordered


def parse_concurrency_levels(raw_value: str, fallback: int) -> list[int]:
    if fallback < 1:
        raise SystemExit("--concurrency must be 1 or greater.")
    if not raw_value:
        return [fallback]

    levels: list[int] = []
    seen = set()
    for raw_item in raw_value.split(","):
        item = raw_item.strip()
        if not item:
            raise SystemExit("--concurrency-levels must be a comma-separated list of positive integers.")
        try:
            value = int(item)
        except ValueError as exc:
            raise SystemExit("--concurrency-levels must contain only positive integers.") from exc
        if value < 1:
            raise SystemExit("--concurrency-levels values must be 1 or greater.")
        if value not in seen:
            levels.append(value)
            seen.add(value)
    return levels


def print_dry_run(
    cases: list[Any],
    selected_models: list[Any],
    execution_targets: list[tuple[Any, str]],
    skipped: list[Any],
    regions: list[str],
    args: argparse.Namespace,
    prompt_path: Path,
) -> None:
    concurrency_levels = getattr(args, "resolved_concurrency_levels", [args.concurrency])
    print(f"Loaded {len(cases)} benchmark cases from {prompt_path}.")
    print(f"Source label: {args.source_label or 'unspecified'}")
    print(f"Requested regions: {', '.join(regions)}")
    print(f"Selected families: {', '.join(sorted({model.family for model in selected_models}))}")
    print("Selected models:")
    for model in selected_models:
        print(f"- {model.model_id} [{model.family}]")
    print(f"Concurrency levels: {', '.join(str(level) for level in concurrency_levels)}")
    print(f"Runnable region/model targets: {len(execution_targets)}")
    print(f"Planned requests: {len(cases) * len(execution_targets) * args.repeats * len(concurrency_levels)}")
    for case in cases:
        print(f"- case {case.case_id}: {len(case.messages)} message(s)")
    if skipped:
        print("Skipped region/model targets:")
        for item in skipped:
            print(f"- {item.region} / {item.model}: {item.reason}")


def main() -> int:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv(Path(__file__).with_name(".env"))
        load_dotenv(PROJECT_ROOT / ".env")

    args = parse_args()
    if args.list_families:
        print("\n".join(get_family_names()))
        return 0
    if args.list_models:
        print(format_model_listing())
        return 0
    args.resolved_concurrency_levels = parse_concurrency_levels(args.concurrency_levels, args.concurrency)

    selected_models = resolve_models(args.families, args.models, args.include_experimental)
    regions = resolve_regions(args)
    prompt_path = Path(args.prompts)
    if not prompt_path.is_absolute():
        prompt_path = PROJECT_ROOT / prompt_path
    cases = load_cases(prompt_path)
    execution_targets, skipped = plan_execution_matrix(selected_models, regions)

    if args.dry_run:
        print_dry_run(cases, selected_models, execution_targets, skipped, regions, args, prompt_path)
        return 0

    if not execution_targets:
        raise SystemExit("No runnable region/model targets remain after catalog filtering.")

    results = run_benchmark(args, cases, execution_targets)
    payload = aggregate_results(selected_models, results, skipped)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_name = args.report_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"{report_name}.json"
    md_path = output_dir / f"{report_name}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(payload, args, prompt_path, regions), encoding="utf-8")

    print(f"Wrote JSON report to {json_path}")
    print(f"Wrote Markdown report to {md_path}")
    return 0
