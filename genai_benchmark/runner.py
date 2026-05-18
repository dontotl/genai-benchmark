from __future__ import annotations

import concurrent.futures
import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

from .catalog import ModelSpec


DEFAULT_SYSTEM_PROMPT = "You are a precise assistant. Answer directly."
SCHEMA_VERSION = "2.0"


@dataclass
class BenchmarkCase:
    case_id: str
    messages: List[Dict[str, str]]


@dataclass
class RunResult:
    region: str
    model: str
    family: str
    case_id: str
    iteration: int
    concurrency: int
    latency_seconds: float
    ttft_seconds: float | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    output_tokens_per_second: float | None
    post_ttft_output_tokens_per_second: float | None
    streaming: bool
    response_preview: str
    error: str | None
    error_type: str | None
    http_status: int | None
    response_body_preview: str | None
    request_id: str | None


@dataclass
class SkippedCombination:
    region: str
    model: str
    family: str
    reason: str


def build_base_url(region: str) -> str:
    return f"https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/v1"


def load_cases(path: Path) -> List[BenchmarkCase]:
    cases: List[BenchmarkCase] = []
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        case_id = payload.get("id") or f"case-{lineno}"
        if "messages" in payload:
            messages = payload["messages"]
        elif "prompt" in payload:
            messages = [{"role": "user", "content": payload["prompt"]}]
        else:
            raise ValueError(f"{path}:{lineno} must contain either 'messages' or 'prompt'.")
        validate_messages(messages, path, lineno)
        cases.append(BenchmarkCase(case_id=case_id, messages=messages))
    if not cases:
        raise ValueError(f"No benchmark cases found in {path}.")
    return cases


def validate_messages(messages: Sequence[Dict[str, Any]], path: Path, lineno: int) -> None:
    allowed = {"system", "user", "assistant"}
    if not isinstance(messages, Sequence):
        raise ValueError(f"{path}:{lineno} messages must be a list.")
    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise ValueError(f"{path}:{lineno} message {index} must be an object.")
        role = message.get("role")
        content = message.get("content")
        if role not in allowed or not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"{path}:{lineno} message {index} must include a valid role ({sorted(allowed)}) and content."
            )


def to_langchain_messages(messages: Iterable[Dict[str, str]]) -> List[Any]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    converted: List[Any] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    if not converted or converted[0].type != "system":
        converted.insert(0, SystemMessage(content=DEFAULT_SYSTEM_PROMPT))
    return converted


def import_runtime_dependencies() -> tuple[Any, Any, Any]:
    try:
        import httpx
        from langchain_openai import ChatOpenAI
        from oci_openai import OciUserPrincipalAuth
    except ImportError as exc:
        missing_name = getattr(exc, "name", "runtime dependency")
        raise SystemExit(
            f"Missing dependency: {missing_name}. "
            "Create a Python 3.11 virtualenv and run `pip install -r requirements.txt`."
        ) from exc
    return httpx, ChatOpenAI, OciUserPrincipalAuth


def make_llm(args: Any, region: str, model: str) -> Any:
    httpx, ChatOpenAI, OciUserPrincipalAuth = import_runtime_dependencies()
    if not args.compartment_id:
        raise ValueError("OCI_COMPARTMENT_ID or --compartment-id is required.")

    return ChatOpenAI(
        model=model,
        api_key="OCI",
        base_url=build_base_url(region),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        http_client=httpx.Client(
            auth=OciUserPrincipalAuth(profile_name=args.profile),
            headers={"CompartmentId": args.compartment_id},
            timeout=120.0,
        ),
    )


def make_thread_local_llm_factory(args: Any, region: str, model: str) -> Callable[[], Any]:
    local_state = threading.local()

    def get_llm() -> Any:
        llm = getattr(local_state, "llm", None)
        if llm is None:
            llm = make_llm(args, region, model)
            local_state.llm = llm
        return llm

    return get_llm


def extract_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    if getattr(response, "usage_metadata", None):
        usage = response.usage_metadata
        return usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens")

    metadata = getattr(response, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
    input_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
    output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
    total_tokens = token_usage.get("total_tokens")
    return input_tokens, output_tokens, total_tokens


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "".join(parts)
    return "" if content is None else str(content)


def calculate_post_ttft_tokens_per_second(
    output_tokens: int | None,
    latency_seconds: float,
    ttft_seconds: float | None,
) -> float | None:
    if output_tokens is None or ttft_seconds is None:
        return None
    generation_seconds = latency_seconds - ttft_seconds
    if generation_seconds <= 0:
        return None
    return round(output_tokens / generation_seconds, 3)


def invoke_streaming(llm: Any, prompt_messages: Sequence[Any]) -> tuple[Any, str, float | None]:
    response = None
    content_parts: list[str] = []
    ttft_seconds: float | None = None
    start = time.perf_counter()
    for chunk in llm.stream(prompt_messages):
        chunk_text = content_to_text(getattr(chunk, "content", ""))
        if chunk_text and ttft_seconds is None:
            ttft_seconds = time.perf_counter() - start
        content_parts.append(chunk_text)
        if response is None:
            response = chunk
        else:
            try:
                response += chunk
            except Exception:
                response = chunk
    if response is None:
        raise RuntimeError("Streaming response produced no chunks.")
    return response, "".join(content_parts), ttft_seconds


def extract_request_id(headers: Any) -> str | None:
    if not headers:
        return None
    candidates = (
        "opc-request-id",
        "x-request-id",
        "request-id",
        "x-correlation-id",
        "opc-work-request-id",
    )
    for key in candidates:
        value = headers.get(key) if hasattr(headers, "get") else None
        if value is None and isinstance(headers, dict):
            value = headers.get(key.title()) or headers.get(key.upper())
        if value:
            return str(value)
    return None


def normalize_status_code(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def extract_response_body_preview(response: Any, max_chars: int = 1000) -> str | None:
    if response is None:
        return None
    text = getattr(response, "text", None)
    if text is None:
        try:
            json_payload = response.json()
        except Exception:
            json_payload = None
        if json_payload is not None:
            text = json.dumps(json_payload, ensure_ascii=False)
    if text is None:
        content = getattr(response, "content", None)
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        elif content is not None:
            text = str(content)
    if not text:
        return None
    return str(text)[:max_chars]


def extract_failure_details(exc: Exception) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    status_code = getattr(exc, "status_code", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None) or getattr(response, "status", None)
    headers = getattr(response, "headers", None)
    return {
        "error": str(exc),
        "error_type": exc.__class__.__name__,
        "http_status": normalize_status_code(status_code),
        "response_body_preview": extract_response_body_preview(response),
        "request_id": extract_request_id(headers),
    }


def plan_execution_matrix(
    selected_models: Sequence[ModelSpec],
    regions: Sequence[str],
) -> tuple[list[tuple[ModelSpec, str]], list[SkippedCombination]]:
    runnable: list[tuple[ModelSpec, str]] = []
    skipped: list[SkippedCombination] = []
    for spec in selected_models:
        for region in regions:
            if region in spec.regions:
                runnable.append((spec, region))
                continue
            skipped.append(
                SkippedCombination(
                    region=region,
                    model=spec.model_id,
                    family=spec.family,
                    reason=f"Model is not cataloged for on-demand access in region {region}.",
                )
            )
    return runnable, skipped


def run_single_invocation(
    region: str,
    spec: ModelSpec,
    case: BenchmarkCase,
    iteration: int,
    concurrency: int,
    llm_factory: Callable[[], Any],
    prompt_messages: Sequence[Any],
    streaming: bool,
) -> RunResult:
    try:
        llm = llm_factory()
        start = time.perf_counter()
        if streaming:
            response, content, ttft_seconds = invoke_streaming(llm, prompt_messages)
        else:
            response = llm.invoke(prompt_messages)
            ttft_seconds = None
            content = content_to_text(getattr(response, "content", ""))
        latency = time.perf_counter() - start
        input_tokens, output_tokens, total_tokens = extract_usage(response)
        output_tokens_per_second = (
            round(output_tokens / latency, 3) if output_tokens is not None and latency > 0 else None
        )
        post_ttft_output_tokens_per_second = calculate_post_ttft_tokens_per_second(
            output_tokens,
            latency,
            ttft_seconds,
        )
        return RunResult(
            region=region,
            model=spec.model_id,
            family=spec.family,
            case_id=case.case_id,
            iteration=iteration,
            concurrency=concurrency,
            latency_seconds=latency,
            ttft_seconds=round(ttft_seconds, 3) if ttft_seconds is not None else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            output_tokens_per_second=output_tokens_per_second,
            post_ttft_output_tokens_per_second=post_ttft_output_tokens_per_second,
            streaming=streaming,
            response_preview=content[:200],
            error=None,
            error_type=None,
            http_status=None,
            response_body_preview=None,
            request_id=None,
        )
    except Exception as exc:  # pragma: no cover - network/runtime error path
        latency = time.perf_counter() - start if "start" in locals() else 0.0
        failure = extract_failure_details(exc)
        return RunResult(
            region=region,
            model=spec.model_id,
            family=spec.family,
            case_id=case.case_id,
            iteration=iteration,
            concurrency=concurrency,
            latency_seconds=latency,
            ttft_seconds=None,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            output_tokens_per_second=None,
            post_ttft_output_tokens_per_second=None,
            streaming=streaming,
            response_preview="",
            error=failure["error"],
            error_type=failure["error_type"],
            http_status=failure["http_status"],
            response_body_preview=failure["response_body_preview"],
            request_id=failure["request_id"],
        )


def run_benchmark(
    args: Any,
    cases: Sequence[BenchmarkCase],
    execution_targets: Sequence[tuple[ModelSpec, str]],
) -> List[RunResult]:
    results: List[RunResult] = []
    concurrency_levels = getattr(args, "resolved_concurrency_levels", [args.concurrency])
    streaming = bool(getattr(args, "streaming", False))

    for spec, region in execution_targets:
        for case in cases:
            llm_factory = make_thread_local_llm_factory(args, region, spec.model_id)
            prompt_messages = to_langchain_messages(case.messages)
            for concurrency in concurrency_levels:
                iterations = range(1, args.repeats + 1)
                if concurrency <= 1:
                    for iteration in iterations:
                        results.append(
                            run_single_invocation(
                                region,
                                spec,
                                case,
                                iteration,
                                concurrency,
                                llm_factory,
                                prompt_messages,
                                streaming,
                            )
                        )
                    continue

                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    future_map = {
                        executor.submit(
                            run_single_invocation,
                            region,
                            spec,
                            case,
                            iteration,
                            concurrency,
                            llm_factory,
                            prompt_messages,
                            streaming,
                        ): iteration
                        for iteration in iterations
                    }
                    for future in concurrent.futures.as_completed(future_map):
                        results.append(future.result())
    return results


def percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def aggregate_results(
    selected_models: Sequence[ModelSpec],
    results: Sequence[RunResult],
    skipped: Sequence[SkippedCombination],
    benchmark_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    grouped: Dict[tuple[str, str, str, int], List[RunResult]] = {}
    for result in results:
        grouped.setdefault((result.region, result.model, result.case_id, result.concurrency), []).append(result)

    summaries: List[Dict[str, Any]] = []
    for (region, model, case_id, concurrency), items in sorted(grouped.items()):
        successful = [item for item in items if item.error is None]
        latencies = [item.latency_seconds for item in successful]
        total_tokens = [item.total_tokens for item in successful if item.total_tokens is not None]
        ttfts = [item.ttft_seconds for item in successful if item.ttft_seconds is not None]
        tokens_per_second = [
            item.output_tokens_per_second for item in successful if item.output_tokens_per_second is not None
        ]
        post_ttft_tokens_per_second = [
            item.post_ttft_output_tokens_per_second
            for item in successful
            if item.post_ttft_output_tokens_per_second is not None
        ]
        family = items[0].family if items else ""
        streaming = items[0].streaming if items else False
        summaries.append(
            {
                "region": region,
                "model": model,
                "family": family,
                "case_id": case_id,
                "concurrency": concurrency,
                "streaming": streaming,
                "attempts": len(items),
                "successes": len(successful),
                "failures": len(items) - len(successful),
                "avg_latency_seconds": round(statistics.fmean(latencies), 3) if latencies else None,
                "p95_latency_seconds": round(percentile(latencies, 95), 3) if latencies else None,
                "p99_latency_seconds": round(percentile(latencies, 99), 3) if latencies else None,
                "avg_ttft_seconds": round(statistics.fmean(ttfts), 3) if ttfts else None,
                "avg_total_tokens": round(statistics.fmean(total_tokens), 1) if total_tokens else None,
                "avg_output_tokens_per_second": round(statistics.fmean(tokens_per_second), 3)
                if tokens_per_second
                else None,
                "avg_post_ttft_output_tokens_per_second": round(statistics.fmean(post_ttft_tokens_per_second), 3)
                if post_ttft_tokens_per_second
                else None,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_config": benchmark_config or {},
        "selected_models": [asdict(model) for model in selected_models],
        "summary": summaries,
        "results": [asdict(result) for result in results],
        "skipped": [asdict(item) for item in skipped],
    }
