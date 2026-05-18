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
from genai_benchmark.dashboard import load_reports, render_html
from genai_benchmark.runner import (
    RunResult,
    aggregate_results,
    extract_failure_details,
    invoke_streaming,
    make_thread_local_llm_factory,
)


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


if __name__ == "__main__":
    unittest.main()
