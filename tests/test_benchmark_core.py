from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import genai_benchmark.runner as runner_module
from genai_benchmark.cli import parse_concurrency_levels, planned_request_count
from genai_benchmark.catalog import ModelSpec, get_family_names, resolve_models
from genai_benchmark.dashboard import load_reports, load_suite_summaries, render_html
from genai_benchmark.runner import (
    BenchmarkCase,
    RunResult,
    aggregate_results,
    extract_failure_details,
    invoke_streaming,
    load_cases,
    make_thread_local_llm_factory,
    run_benchmark,
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

    def test_planned_request_count_uses_load_wave_size(self) -> None:
        self.assertEqual(planned_request_count(6, 10, 1, [1, 5, 10], load_test=True), 960)
        self.assertEqual(planned_request_count(6, 10, 1, [1, 5, 10], load_test=False), 180)


class CatalogSelectionTest(unittest.TestCase):
    def test_catalog_lists_new_multimodel_families(self) -> None:
        families = get_family_names()

        self.assertIn("cohere", families)
        self.assertIn("grok", families)
        self.assertIn("meta", families)

    def test_include_experimental_selects_representative_models(self) -> None:
        models = resolve_models(
            ["openai", "gemini", "grok", "meta", "cohere"],
            None,
            include_experimental=True,
        )

        self.assertEqual(
            [model.model_id for model in models],
            [
                "openai.gpt-oss-20b",
                "google.gemini-2.5-flash",
                "xai.grok-4.3",
                "meta.llama-4-scout-17b-16e-instruct",
                "cohere.command-a-03-2025",
            ],
        )

    def test_experimental_defaults_are_excluded_without_flag(self) -> None:
        with self.assertRaises(SystemExit):
            resolve_models(["grok", "meta", "cohere"], None, include_experimental=False)


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


class RunBenchmarkLoadTest(unittest.TestCase):
    def test_load_test_runs_concurrency_sized_waves(self) -> None:
        original_factory = runner_module.make_thread_local_llm_factory
        original_run_single = runner_module.run_single_invocation
        original_to_messages = runner_module.to_langchain_messages
        spec = ModelSpec("openai.gpt-oss-20b", "openai", "OpenAI gpt-oss-20b", ("ap-osaka-1",))
        case = BenchmarkCase("summary-ko", [{"role": "user", "content": "테스트"}])

        def fake_factory(_args: object, _region: str, _model: str) -> object:
            return object

        def fake_run_single(
            region: str,
            model_spec: ModelSpec,
            benchmark_case: BenchmarkCase,
            iteration: int,
            concurrency: int,
            _llm_factory: object,
            _prompt_messages: object,
            streaming: bool,
        ) -> RunResult:
            return RunResult(
                region=region,
                model=model_spec.model_id,
                family=model_spec.family,
                case_id=benchmark_case.case_id,
                iteration=iteration,
                concurrency=concurrency,
                latency_seconds=0.1,
                ttft_seconds=None,
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                output_tokens_per_second=10.0,
                post_ttft_output_tokens_per_second=None,
                streaming=streaming,
                response_preview="ok",
                error=None,
                error_type=None,
                http_status=None,
                response_body_preview=None,
                request_id=None,
            )

        runner_module.make_thread_local_llm_factory = fake_factory
        runner_module.run_single_invocation = fake_run_single
        runner_module.to_langchain_messages = lambda messages: messages
        try:
            results = run_benchmark(
                SimpleNamespace(
                    repeats=1,
                    concurrency=1,
                    resolved_concurrency_levels=[1, 10],
                    streaming=False,
                    load_test=True,
                ),
                [case],
                [(spec, "ap-osaka-1")],
            )
        finally:
            runner_module.make_thread_local_llm_factory = original_factory
            runner_module.run_single_invocation = original_run_single
            runner_module.to_langchain_messages = original_to_messages

        self.assertEqual(len(results), 11)
        self.assertEqual(sum(1 for result in results if result.concurrency == 1), 1)
        self.assertEqual(sum(1 for result in results if result.concurrency == 10), 10)
        self.assertEqual(sorted(result.iteration for result in results if result.concurrency == 10), list(range(1, 11)))

    def test_default_mode_preserves_repeats_per_level(self) -> None:
        original_factory = runner_module.make_thread_local_llm_factory
        original_run_single = runner_module.run_single_invocation
        original_to_messages = runner_module.to_langchain_messages
        spec = ModelSpec("openai.gpt-oss-20b", "openai", "OpenAI gpt-oss-20b", ("ap-osaka-1",))
        case = BenchmarkCase("summary-ko", [{"role": "user", "content": "테스트"}])

        def fake_factory(_args: object, _region: str, _model: str) -> object:
            return object

        def fake_run_single(
            region: str,
            model_spec: ModelSpec,
            benchmark_case: BenchmarkCase,
            iteration: int,
            concurrency: int,
            _llm_factory: object,
            _prompt_messages: object,
            streaming: bool,
        ) -> RunResult:
            return make_result(concurrency=concurrency, iteration=iteration, latency=0.1, output_tokens=1, streaming=streaming)

        runner_module.make_thread_local_llm_factory = fake_factory
        runner_module.run_single_invocation = fake_run_single
        runner_module.to_langchain_messages = lambda messages: messages
        try:
            results = run_benchmark(
                SimpleNamespace(
                    repeats=1,
                    concurrency=1,
                    resolved_concurrency_levels=[1, 10],
                    streaming=False,
                    load_test=False,
                ),
                [case],
                [(spec, "ap-osaka-1")],
            )
        finally:
            runner_module.make_thread_local_llm_factory = original_factory
            runner_module.run_single_invocation = original_run_single
            runner_module.to_langchain_messages = original_to_messages

        self.assertEqual(len(results), 2)
        self.assertEqual([result.concurrency for result in results], [1, 10])


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


class WorkloadPromptTest(unittest.TestCase):
    def test_chat_nl2sql_workload_file_loads_expected_cases(self) -> None:
        cases = load_cases(Path("prompts/chat_nl2sql_workloads.jsonl"))
        by_id = {case.case_id: case for case in cases}

        self.assertEqual(
            set(by_id),
            {
                "chat-helpdesk",
                "summary-ko",
                "code-debug",
                "reasoning-choice",
                "agentic-plan",
                "nl2sql-sales-analytics",
            },
        )
        nl2sql_messages = " ".join(message["content"] for message in by_id["nl2sql-sales-analytics"].messages)
        self.assertIn("customers", nl2sql_messages)
        self.assertIn("SELECT", nl2sql_messages)
        self.assertIn("Question:", nl2sql_messages)


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
            "benchmark_config": {"source_label": "ap-osaka-runner"},
            "summary": [
                {
                    "region": "ap-osaka-1",
                    "model": "openai.gpt-oss-20b",
                    "family": "openai",
                    "case_id": "summary-ko",
                    "attempts": 1,
                    "successes": 1,
                    "failures": 0,
                    "avg_latency_seconds": 2.0,
                    "p95_latency_seconds": 2.0,
                    "p99_latency_seconds": 2.0,
                    "avg_ttft_seconds": None,
                    "avg_total_tokens": 100.0,
                    "avg_output_tokens_per_second": 50.0,
                },
            ],
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
            Path(tmp_dir, "ap-osaka-runner-global-smoke-r1.json").write_text(json.dumps(report_payload), encoding="utf-8")
            html = render_html(load_reports(Path(tmp_dir)), [suite_summary])

        self.assertIn("Region-to-Region Performance: global-smoke-r1", html)
        self.assertIn("Runner Summary", html)
        self.assertIn("각 runner JSON report의 P95", html)
        self.assertIn("info-tooltip", html)
        self.assertIn("Tok/sec 50.0", html)
        self.assertIn("TTFT -", html)
        self.assertIn("source best", html)
        self.assertIn("target best", html)

    def test_dashboard_renders_runner_context_and_workload_descriptions(self) -> None:
        report_payload = {
            "benchmark_config": {
                "repeats": 1,
                "concurrency_levels": [1],
                "streaming": False,
                "temperature": 0.0,
                "max_tokens": 512,
            },
            "selected_models": [
                {"model_id": "openai.gpt-oss-20b", "label": "OpenAI gpt-oss-20b"},
            ],
            "summary": [
                {
                    "region": "ap-osaka-1",
                    "model": "openai.gpt-oss-20b",
                    "family": "openai",
                    "case_id": "nl2sql-sales-analytics",
                    "attempts": 1,
                    "successes": 1,
                    "failures": 0,
                    "avg_latency_seconds": 1.0,
                    "p95_latency_seconds": 1.0,
                    "avg_total_tokens": 20.0,
                },
            ],
            "results": [],
        }
        suite_summary = {
            "path": Path("runs/global-chat-nl2sql-r1-summary.md"),
            "name": "global-chat-nl2sql-r1",
            "generated_at": "2026-05-26 04:00:23 UTC",
            "target_regions": ["ap-osaka-1"],
            "attempts": 1,
            "successes": 1,
            "failures": 0,
            "sources": [
                {
                    "source": "ap-osaka-runner",
                    "status": "succeeded",
                    "attempts": 1,
                    "successes": 1,
                    "failures": 0,
                    "avg_latency": 1.0,
                    "json": "report.json",
                    "markdown": "report.md",
                },
            ],
            "target_latency": [
                {"source": "ap-osaka-runner", "target_region": "ap-osaka-1", "avg_latency": 1.0},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            Path(tmp_dir, "report.json").write_text(json.dumps(report_payload), encoding="utf-8")
            html = render_html(load_reports(Path(tmp_dir)), [suite_summary])

        self.assertIn("Test Context", html)
        self.assertIn("Workload Details", html)
        self.assertIn("OpenAI gpt-oss-20b", html)
        self.assertIn("nl2sql-sales-analytics", html)
        self.assertIn("SQL SELECT query", html)

    def test_dashboard_uses_sample_prompt_fallback_for_legacy_workloads(self) -> None:
        report_payload = {
            "summary": [
                {
                    "region": "ap-osaka-1",
                    "model": "openai.gpt-oss-20b",
                    "family": "openai",
                    "case_id": "table-en",
                    "attempts": 1,
                    "successes": 1,
                    "failures": 0,
                    "avg_latency_seconds": 1.0,
                    "p95_latency_seconds": 1.0,
                    "p99_latency_seconds": 1.0,
                    "avg_total_tokens": 20.0,
                    "avg_output_tokens_per_second": 10.0,
                },
                {
                    "region": "ap-osaka-1",
                    "model": "openai.gpt-oss-20b",
                    "family": "openai",
                    "case_id": "ops-checklist",
                    "attempts": 1,
                    "successes": 1,
                    "failures": 0,
                    "avg_latency_seconds": 1.0,
                    "p95_latency_seconds": 1.0,
                    "p99_latency_seconds": 1.0,
                    "avg_total_tokens": 20.0,
                    "avg_output_tokens_per_second": 10.0,
                },
            ],
            "results": [],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            Path(tmp_dir, "legacy.json").write_text(json.dumps(report_payload), encoding="utf-8")
            html = render_html(load_reports(Path(tmp_dir)))

        self.assertIn("Create a compact markdown table", html)
        self.assertIn("운영 환경에서 GenAI 모델 benchmark를 수행할 때 확인해야 할 체크리스트 6개", html)

    def test_dashboard_moves_raw_tables_into_debug_details(self) -> None:
        html = render_html([])

        self.assertIn("Raw Data / Debug Details", html)
        self.assertIn("<summary>Raw Data / Debug Details</summary>", html)
        self.assertIn("Latency vs Token Volume", html)
        self.assertIn("마우스를 올리면", html)
        self.assertNotIn("Efficiency View", html)

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

    def test_dashboard_renders_multimodel_load_controls_and_skips(self) -> None:
        summary_rows = []
        for family, model in (
            ("grok", "xai.grok-4.3"),
            ("meta", "meta.llama-4-scout-17b-16e-instruct"),
            ("cohere", "cohere.command-a-03-2025"),
        ):
            for concurrency, latency, ttft in ((1, 1.0, 0.2), (5, 1.5, 0.3), (10, 2.0, 0.4)):
                summary_rows.append(
                    {
                        "region": "us-chicago-1",
                        "model": model,
                        "family": family,
                        "case_id": "chat-helpdesk",
                        "concurrency": concurrency,
                        "streaming": True,
                        "attempts": 1,
                        "successes": 1,
                        "failures": 0,
                        "avg_latency_seconds": latency,
                        "p95_latency_seconds": latency,
                        "p99_latency_seconds": latency,
                        "avg_ttft_seconds": ttft,
                        "avg_total_tokens": 100.0,
                        "avg_output_tokens_per_second": 25.0,
                        "avg_post_ttft_output_tokens_per_second": 30.0,
                    }
                )
        payload = {
            "benchmark_config": {"streaming": True, "concurrency_levels": [1, 5, 10]},
            "summary": summary_rows,
            "results": [],
            "skipped": [
                {
                    "region": "eu-frankfurt-1",
                    "family": "cohere",
                    "model": "cohere.command-a-03-2025",
                    "reason": "Model is not cataloged for on-demand access in region eu-frankfurt-1.",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            Path(tmp_dir, "load.json").write_text(json.dumps(payload), encoding="utf-8")
            html = render_html(load_reports(Path(tmp_dir)))

        self.assertIn("Load Summary", html)
        self.assertIn("C10/C1 latency multiplier", html)
        self.assertIn("C10 Ranking", html)
        self.assertIn("Load Sensitivity", html)
        self.assertIn("Avg TTFT", html)
        self.assertIn("family-filter", html)
        self.assertIn("legend-dot cohere", html)
        self.assertIn("pill grok", html)
        self.assertIn("pill meta", html)
        self.assertIn("pill cohere", html)
        self.assertIn("지원 리전 아님", html)

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
