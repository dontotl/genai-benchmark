from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import genai_benchmark.runner as runner_module
from genai_benchmark.cli import parse_concurrency_levels
from genai_benchmark.catalog import ModelSpec
from genai_benchmark.dashboard import load_reports, load_suite_summaries, render_html
from genai_benchmark.runner import (
    RunResult,
    aggregate_results,
    extract_failure_details,
    invoke_streaming,
    make_thread_local_llm_factory,
)
from genai_benchmark.site import choose_focus_report


def make_result(
    *,
    concurrency: int,
    iteration: int,
    latency: float,
    output_tokens: int | None,
    error: str | None = None,
    streaming: bool = False,
    ttft: float | None = None,
) -> RunResult:
    return RunResult(
        region="ap-osaka-1",
        model="openai.gpt-oss-20b",
        family="openai",
        case_id="summary-ko",
        iteration=iteration,
        concurrency=concurrency,
        latency_seconds=latency,
        ttft_seconds=ttft,
        input_tokens=10 if error is None else None,
        output_tokens=output_tokens if error is None else None,
        total_tokens=(10 + output_tokens) if output_tokens is not None and error is None else None,
        output_tokens_per_second=round(output_tokens / latency, 3)
        if output_tokens is not None and error is None
        else None,
        post_ttft_output_tokens_per_second=round(output_tokens / (latency - ttft), 3)
        if output_tokens is not None and error is None and ttft is not None and latency > ttft
        else None,
        streaming=streaming,
        response_preview="ok" if error is None else "",
        error=error,
        error_type="RuntimeError" if error else None,
        http_status=500 if error else None,
        response_body_preview="server error" if error else None,
        request_id="req-1" if error else None,
    )


class ConcurrencyLevelsTest(unittest.TestCase):
    def test_uses_fallback_when_levels_are_empty(self) -> None:
        self.assertEqual(parse_concurrency_levels("", 3), [3])

    def test_parses_and_deduplicates_levels(self) -> None:
        self.assertEqual(parse_concurrency_levels("1, 5, 5, 10", 1), [1, 5, 10])

    def test_rejects_invalid_levels(self) -> None:
        with self.assertRaises(SystemExit):
            parse_concurrency_levels("1,zero", 1)
        with self.assertRaises(SystemExit):
            parse_concurrency_levels("1,0", 1)


class AggregateResultsTest(unittest.TestCase):
    def test_groups_summary_by_concurrency_and_calculates_new_metrics(self) -> None:
        spec = ModelSpec(
            model_id="openai.gpt-oss-20b",
            family="openai",
            label="OpenAI gpt-oss-20b",
            regions=("ap-osaka-1",),
        )
        results = [
            make_result(concurrency=1, iteration=1, latency=1.0, output_tokens=10),
            make_result(concurrency=1, iteration=2, latency=2.0, output_tokens=20),
            make_result(concurrency=5, iteration=1, latency=3.0, output_tokens=30),
            make_result(concurrency=5, iteration=2, latency=4.0, output_tokens=None, error="failed"),
        ]

        payload = aggregate_results([spec], results, skipped=[])
        by_concurrency = {item["concurrency"]: item for item in payload["summary"]}

        self.assertEqual(by_concurrency[1]["attempts"], 2)
        self.assertEqual(by_concurrency[1]["successes"], 2)
        self.assertEqual(by_concurrency[1]["avg_latency_seconds"], 1.5)
        self.assertEqual(by_concurrency[1]["p95_latency_seconds"], 1.95)
        self.assertEqual(by_concurrency[1]["p99_latency_seconds"], 1.99)
        self.assertEqual(by_concurrency[1]["avg_output_tokens_per_second"], 10.0)

        self.assertEqual(by_concurrency[5]["attempts"], 2)
        self.assertEqual(by_concurrency[5]["successes"], 1)
        self.assertEqual(by_concurrency[5]["failures"], 1)
        self.assertEqual(by_concurrency[5]["avg_output_tokens_per_second"], 10.0)
        self.assertEqual(payload["schema_version"], "2.0")
        self.assertIn("generated_at", payload)

    def test_aggregates_streaming_ttft_metrics(self) -> None:
        spec = ModelSpec(
            model_id="openai.gpt-oss-20b",
            family="openai",
            label="OpenAI gpt-oss-20b",
            regions=("ap-osaka-1",),
        )
        results = [
            make_result(concurrency=1, iteration=1, latency=2.0, output_tokens=20, streaming=True, ttft=0.5),
        ]

        payload = aggregate_results([spec], results, skipped=[], benchmark_config={"streaming": True})
        summary = payload["summary"][0]

        self.assertTrue(summary["streaming"])
        self.assertEqual(summary["avg_ttft_seconds"], 0.5)
        self.assertEqual(summary["avg_post_ttft_output_tokens_per_second"], 13.333)
        self.assertTrue(payload["benchmark_config"]["streaming"])


class StreamingInvocationTest(unittest.TestCase):
    def test_invoke_streaming_combines_chunks(self) -> None:
        class Chunk:
            def __init__(self, content: str) -> None:
                self.content = content

            def __add__(self, other: "Chunk") -> "Chunk":
                return Chunk(self.content + other.content)

        class FakeLlm:
            def stream(self, _messages: list[object]) -> list[Chunk]:
                return [Chunk("hello"), Chunk(" "), Chunk("world")]

        response, content, ttft = invoke_streaming(FakeLlm(), [])

        self.assertEqual(content, "hello world")
        self.assertEqual(response.content, "hello world")
        self.assertIsNotNone(ttft)


class ThreadLocalFactoryTest(unittest.TestCase):
    def test_factory_reuses_per_thread_but_not_across_threads(self) -> None:
        calls = []
        original_make_llm = runner_module.make_llm

        def fake_make_llm(_args: object, _region: str, _model: str) -> object:
            llm = object()
            calls.append(llm)
            return llm

        runner_module.make_llm = fake_make_llm
        try:
            factory = make_thread_local_llm_factory(SimpleNamespace(), "ap-osaka-1", "model")
            same_thread_first = factory()
            same_thread_second = factory()
            with ThreadPoolExecutor(max_workers=1) as executor:
                other_thread = executor.submit(factory).result()
        finally:
            runner_module.make_llm = original_make_llm

        self.assertIs(same_thread_first, same_thread_second)
        self.assertIsNot(same_thread_first, other_thread)
        self.assertEqual(len(calls), 2)


class FailureDetailsTest(unittest.TestCase):
    def test_extracts_safe_http_failure_details(self) -> None:
        class FakeResponse:
            status_code = 503
            text = "temporary failure body"
            headers = {"opc-request-id": "opc-123"}

        class FakeError(Exception):
            response = FakeResponse()

        details = extract_failure_details(FakeError("service unavailable"))

        self.assertEqual(details["error_type"], "FakeError")
        self.assertEqual(details["http_status"], 503)
        self.assertEqual(details["response_body_preview"], "temporary failure body")
        self.assertEqual(details["request_id"], "opc-123")


class DashboardCompatibilityTest(unittest.TestCase):
    def test_dashboard_reads_legacy_json_without_new_fields(self) -> None:
        legacy_payload = {
            "summary": [
                {
                    "region": "ap-osaka-1",
                    "model": "openai.gpt-oss-20b",
                    "family": "openai",
                    "case_id": "summary-ko",
                    "attempts": 1,
                    "successes": 1,
                    "failures": 0,
                    "avg_latency_seconds": 1.0,
                    "p95_latency_seconds": 1.0,
                    "avg_total_tokens": 20.0,
                }
            ],
            "results": [
                {
                    "region": "ap-osaka-1",
                    "model": "openai.gpt-oss-20b",
                    "family": "openai",
                    "case_id": "summary-ko",
                    "iteration": 1,
                    "latency_seconds": 1.0,
                    "error": None,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            Path(tmp_dir, "legacy.json").write_text(json.dumps(legacy_payload), encoding="utf-8")
            html = render_html(load_reports(Path(tmp_dir)))

        self.assertIn("Concurrency", html)
        self.assertIn("summary-ko", html)
        self.assertIn("region-filter", html)
        self.assertIn("case-sort", html)
        self.assertIn("Avg E2E Output Tokens/sec", html)
        self.assertIn("Avg TTFT", html)
        self.assertNotIn("Runner Matrix", html)

    def test_dashboard_reads_suite_summary_markdown(self) -> None:
        summary_markdown = """# global-smoke-r1 Suite Summary

- Generated At: `2026-05-26 04:00:23 UTC`
- Target Regions: `ap-osaka-1`, `us-chicago-1`
- Attempts: `4`
- Successes: `4`
- Failures: `0`

## Source Runner Summary

| Source | Status | Attempts | Successes | Failures | Avg Latency | JSON | Markdown |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `ap-osaka-runner` | `succeeded` | 2 | 2 | 0 | 2.100s | [json](ap-osaka-runner-global-smoke-r1.json) | [md](ap-osaka-runner-global-smoke-r1.md) |
| `us-chicago-runner` | `succeeded` | 2 | 2 | 0 | 1.900s | [json](us-chicago-runner-global-smoke-r1.json) | [md](us-chicago-runner-global-smoke-r1.md) |

## Target Region Average Latency

| Source | Target Region | Avg Latency |
| --- | --- | ---: |
| `ap-osaka-runner` | `ap-osaka-1` | 2.000s |
| `ap-osaka-runner` | `us-chicago-1` | 2.200s |
| `us-chicago-runner` | `ap-osaka-1` | 2.300s |
| `us-chicago-runner` | `us-chicago-1` | 1.500s |
"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            Path(tmp_dir, "global-smoke-r1-summary.md").write_text(summary_markdown, encoding="utf-8")
            summaries = load_suite_summaries(Path(tmp_dir))

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["name"], "global-smoke-r1")
        self.assertEqual(summaries[0]["target_regions"], ["ap-osaka-1", "us-chicago-1"])
        self.assertEqual(summaries[0]["sources"][0]["avg_latency"], 2.1)
        self.assertEqual(summaries[0]["target_latency"][3]["avg_latency"], 1.5)

    def test_dashboard_renders_runner_matrix_from_suite_summary(self) -> None:
        report_payload = {
            "summary": [],
            "results": [],
        }
        suite_summary = {
            "path": Path("runs/global-smoke-r1-summary.md"),
            "name": "global-smoke-r1",
            "generated_at": "2026-05-26 04:00:23 UTC",
            "target_regions": ["ap-osaka-1", "us-chicago-1"],
            "attempts": 4,
            "successes": 4,
            "failures": 0,
            "sources": [
                {
                    "source": "ap-osaka-runner",
                    "status": "succeeded",
                    "attempts": 2,
                    "successes": 2,
                    "failures": 0,
                    "avg_latency": 2.1,
                    "json": "ap-osaka-runner-global-smoke-r1.json",
                    "markdown": "ap-osaka-runner-global-smoke-r1.md",
                },
                {
                    "source": "us-chicago-runner",
                    "status": "succeeded",
                    "attempts": 2,
                    "successes": 2,
                    "failures": 0,
                    "avg_latency": 1.9,
                    "json": "us-chicago-runner-global-smoke-r1.json",
                    "markdown": "us-chicago-runner-global-smoke-r1.md",
                },
            ],
            "target_latency": [
                {"source": "ap-osaka-runner", "target_region": "ap-osaka-1", "avg_latency": 2.0},
                {"source": "ap-osaka-runner", "target_region": "us-chicago-1", "avg_latency": 2.2},
                {"source": "us-chicago-runner", "target_region": "ap-osaka-1", "avg_latency": 2.3},
                {"source": "us-chicago-runner", "target_region": "us-chicago-1", "avg_latency": 1.5},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            Path(tmp_dir, "report.json").write_text(json.dumps(report_payload), encoding="utf-8")
            html = render_html(load_reports(Path(tmp_dir)), [suite_summary])

        self.assertIn("Runner Matrix: global-smoke-r1", html)
        self.assertIn("Runner Ranking", html)
        self.assertIn("source best", html)
        self.assertIn("target best", html)

    def test_dashboard_renders_failure_summary(self) -> None:
        payload = {
            "summary": [],
            "results": [
                {
                    "region": "ap-osaka-1",
                    "model": "google.gemini-2.5-flash",
                    "family": "gemini",
                    "case_id": "ops-checklist",
                    "concurrency": 5,
                    "iteration": 1,
                    "latency_seconds": 0.5,
                    "error": "server failed",
                    "error_type": "HTTPStatusError",
                    "http_status": 500,
                    "request_id": "opc-123",
                    "response_body_preview": "internal error",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            Path(tmp_dir, "failure.json").write_text(json.dumps(payload), encoding="utf-8")
            html = render_html(load_reports(Path(tmp_dir)))

        self.assertIn("Failure Summary", html)
        self.assertIn("HTTPStatusError", html)
        self.assertIn("opc-123", html)

    def test_dashboard_skips_non_report_json_files(self) -> None:
        report_payload = {"summary": [], "results": []}
        state_payload = {"source_label": "runner", "resources": {"instance_id": "ocid1.instance"}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            Path(tmp_dir, "report.json").write_text(json.dumps(report_payload), encoding="utf-8")
            Path(tmp_dir, "runner-state.json").write_text(json.dumps(state_payload), encoding="utf-8")
            reports = load_reports(Path(tmp_dir))

        self.assertEqual([report["name"] for report in reports], ["report"])

    def test_site_focus_prefers_latest_generated_report(self) -> None:
        focus = choose_focus_report(
            [
                {"name": "cross-region-baseline-r3", "generated_at": "2026-05-20T00:00:00+00:00"},
                {"name": "ap-osaka-runner-global-smoke-r1", "generated_at": "2026-05-21T00:00:00+00:00"},
            ]
        )

        self.assertEqual(focus["name"], "ap-osaka-runner-global-smoke-r1")


if __name__ == "__main__":
    unittest.main()
