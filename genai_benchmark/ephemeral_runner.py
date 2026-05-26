from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from .dashboard import load_reports
from .site import write_docs


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = PROJECT_ROOT / "scripts" / "cloud-init" / "ephemeral-runner.sh.tmpl"


class ConfigError(ValueError):
    """Raised when the ephemeral runner config is incomplete or invalid."""


@dataclass(frozen=True)
class Runner:
    region: str
    source_label: str
    availability_domain: str
    subnet_id: str
    image_id: str
    ssh_public_key: str | None = None
    shape: str | None = None
    shape_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunnerConfig:
    compartment_id: str
    benchmark_compartment_id: str
    bucket_name: str
    object_storage_region: str
    repo_url: str
    repo_ref: str
    control_profile: str
    runner_profile: str
    runner_auth_method: str
    runner_user: str
    work_dir: str
    shape: str
    shape_config: dict[str, Any] | None
    network: dict[str, Any]
    target_regions: list[str]
    benchmark: dict[str, Any]
    runners: list[Runner]


@dataclass(frozen=True)
class ManagedResources:
    vcn_id: str
    internet_gateway_id: str
    route_table_id: str
    security_list_id: str
    subnet_id: str
    instance_id: str | None = None
    dynamic_group_id: str | None = None
    policy_id: str | None = None


def require_string(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context}.{key} is required.")
    return value


def load_config(path: Path) -> RunnerConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a JSON object.")

    target_regions = raw.get("target_regions")
    if not isinstance(target_regions, list) or not all(isinstance(item, str) and item for item in target_regions):
        raise ConfigError("target_regions must be a non-empty list of strings.")

    benchmark = raw.get("benchmark", {})
    if not isinstance(benchmark, dict):
        raise ConfigError("benchmark must be an object when provided.")
    network = raw.get("network") or {}
    if not isinstance(network, dict):
        raise ConfigError("network must be an object when provided.")
    runner_auth_method = raw.get("runner_auth_method") or "user_principal"
    if runner_auth_method not in ("user_principal", "instance_principal"):
        raise ConfigError("runner_auth_method must be user_principal or instance_principal.")

    runners_raw = raw.get("runners")
    if not isinstance(runners_raw, list) or not runners_raw:
        raise ConfigError("runners must be a non-empty list.")

    runners = []
    seen_labels = set()
    for index, item in enumerate(runners_raw, start=1):
        if not isinstance(item, dict):
            raise ConfigError(f"runners[{index}] must be an object.")
        runner = Runner(
            region=require_string(item, "region", f"runners[{index}]"),
            source_label=require_string(item, "source_label", f"runners[{index}]"),
            availability_domain=require_string(item, "availability_domain", f"runners[{index}]"),
            subnet_id=item.get("subnet_id") or "",
            image_id=require_string(item, "image_id", f"runners[{index}]"),
            ssh_public_key=item.get("ssh_public_key"),
            shape=item.get("shape"),
            shape_config=item.get("shape_config"),
        )
        if runner.source_label in seen_labels:
            raise ConfigError(f"Duplicate source_label: {runner.source_label}")
        seen_labels.add(runner.source_label)
        runners.append(runner)

    object_storage_region = raw.get("object_storage_region") or target_regions[0]
    return RunnerConfig(
        compartment_id=require_string(raw, "compartment_id", "config"),
        benchmark_compartment_id=raw.get("benchmark_compartment_id") or require_string(raw, "compartment_id", "config"),
        bucket_name=require_string(raw, "bucket_name", "config"),
        object_storage_region=object_storage_region,
        repo_url=require_string(raw, "repo_url", "config"),
        repo_ref=raw.get("repo_ref") or "main",
        control_profile=raw.get("control_profile") or "DEFAULT",
        runner_profile=raw.get("runner_profile") or "DEFAULT",
        runner_auth_method=runner_auth_method,
        runner_user=raw.get("runner_user") or "opc",
        work_dir=raw.get("work_dir") or "/home/opc/genai-benchmark",
        shape=require_string(raw, "shape", "config"),
        shape_config=raw.get("shape_config"),
        network=network,
        target_regions=target_regions,
        benchmark=benchmark,
        runners=runners,
    )


def selected_runners(config: RunnerConfig, labels: Sequence[str]) -> list[Runner]:
    if not labels:
        return config.runners
    by_label = {runner.source_label: runner for runner in config.runners}
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise ConfigError(f"Unknown runner source_label: {', '.join(missing)}")
    return [by_label[label] for label in labels]


def report_name(config: RunnerConfig, runner: Runner) -> str:
    suffix = config.benchmark.get("report_suffix") or "global-r3"
    return f"{runner.source_label}-{suffix}"


def benchmark_args(config: RunnerConfig, runner: Runner) -> list[str]:
    benchmark = config.benchmark
    args = [
        "--source-label",
        runner.source_label,
        "--profile",
        config.runner_profile,
        "--auth-method",
        config.runner_auth_method,
        "--compartment-id",
        config.benchmark_compartment_id,
    ]
    for region in config.target_regions:
        args.extend(["--region", region])
    args.extend(
        [
            "--repeats",
            str(benchmark.get("repeats", 3)),
            "--concurrency-levels",
            str(benchmark.get("concurrency_levels", "1,5,10")),
            "--prompts",
            str(benchmark.get("prompts", "prompts/sample_prompts.jsonl")),
            "--report-name",
            report_name(config, runner),
        ]
    )
    if benchmark.get("streaming"):
        args.append("--streaming")
    if benchmark.get("load_test"):
        args.append("--load-test")
    if benchmark.get("include_experimental"):
        args.append("--include-experimental")
    for key in ("temperature", "max_tokens"):
        if key in benchmark:
            args.extend([f"--{key.replace('_', '-')}", str(benchmark[key])])
    for family in benchmark.get("families", []):
        args.extend(["--family", str(family)])
    for model in benchmark.get("models", []):
        args.extend(["--model", str(model)])
    for item in benchmark.get("extra_args", []):
        args.append(str(item))
    return args


def quote_env(value: str) -> str:
    return shlex.quote(value)


def runner_oci_auth_args(config: RunnerConfig) -> list[str]:
    if config.runner_auth_method == "instance_principal":
        return ["--auth", "instance_principal"]
    return ["--profile", config.runner_profile]


def render_cloud_init(config: RunnerConfig, runner: Runner, template_path: Path = DEFAULT_TEMPLATE) -> str:
    template = template_path.read_text(encoding="utf-8")
    replacements = {
        "__BENCHMARK_ARGS__": " ".join(shlex.quote(item) for item in benchmark_args(config, runner)),
        "__OCI_AUTH_ARGS__": " ".join(shlex.quote(item) for item in runner_oci_auth_args(config)),
        "__RUNNER_USER__": config.runner_user,
        "__SOURCE_LABEL__": quote_env(runner.source_label),
        "__REPORT_NAME__": quote_env(report_name(config, runner)),
        "__COMPLETION_MARKER__": quote_env(completion_marker_name(config, runner)),
        "__BUCKET_NAME__": quote_env(config.bucket_name),
        "__OBJECT_STORAGE_REGION__": quote_env(config.object_storage_region),
        "__REPO_URL__": quote_env(config.repo_url),
        "__REPO_REF__": quote_env(config.repo_ref),
        "__WORK_DIR__": quote_env(config.work_dir),
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    return rendered


def oci_base(config: RunnerConfig) -> list[str]:
    return ["oci", "--profile", config.control_profile]


def managed_prefix(config: RunnerConfig, runner: Runner) -> str:
    prefix = config.network.get("name_prefix") or "genai-benchmark"
    return f"{prefix}-{runner.source_label}"


def managed_cidr(config: RunnerConfig, key: str, fallback: str) -> str:
    value = config.network.get(key) or fallback
    if not isinstance(value, str) or not value:
        raise ConfigError(f"network.{key} must be a non-empty string.")
    return value


def vcn_create_command(config: RunnerConfig, runner: Runner) -> list[str]:
    return [
        *oci_base(config),
        "network",
        "vcn",
        "create",
        "--region",
        runner.region,
        "--compartment-id",
        config.compartment_id,
        "--display-name",
        f"{managed_prefix(config, runner)}-vcn",
        "--dns-label",
        str(config.network.get("dns_label") or "genaibench")[:15],
        "--cidr-block",
        managed_cidr(config, "vcn_cidr", "10.91.0.0/16"),
        "--wait-for-state",
        "AVAILABLE",
        "--output",
        "json",
    ]


def internet_gateway_create_command(config: RunnerConfig, runner: Runner, vcn_id: str) -> list[str]:
    return [
        *oci_base(config),
        "network",
        "internet-gateway",
        "create",
        "--region",
        runner.region,
        "--compartment-id",
        config.compartment_id,
        "--vcn-id",
        vcn_id,
        "--is-enabled",
        "true",
        "--display-name",
        f"{managed_prefix(config, runner)}-igw",
        "--wait-for-state",
        "AVAILABLE",
        "--output",
        "json",
    ]


def route_table_create_command(config: RunnerConfig, runner: Runner, vcn_id: str, internet_gateway_id: str) -> list[str]:
    route_rules = [{"cidrBlock": "0.0.0.0/0", "networkEntityId": internet_gateway_id}]
    return [
        *oci_base(config),
        "network",
        "route-table",
        "create",
        "--region",
        runner.region,
        "--compartment-id",
        config.compartment_id,
        "--vcn-id",
        vcn_id,
        "--display-name",
        f"{managed_prefix(config, runner)}-rt",
        "--route-rules",
        json.dumps(route_rules, separators=(",", ":")),
        "--wait-for-state",
        "AVAILABLE",
        "--output",
        "json",
    ]


def security_list_create_command(config: RunnerConfig, runner: Runner, vcn_id: str) -> list[str]:
    ingress_rules = []
    if config.network.get("enable_ssh", True):
        ingress_rules.append(
            {
                "protocol": "6",
                "source": str(config.network.get("ssh_source_cidr") or "0.0.0.0/0"),
                "tcpOptions": {"destinationPortRange": {"min": 22, "max": 22}},
            }
        )
    egress_rules = [{"protocol": "all", "destination": "0.0.0.0/0"}]
    return [
        *oci_base(config),
        "network",
        "security-list",
        "create",
        "--region",
        runner.region,
        "--compartment-id",
        config.compartment_id,
        "--vcn-id",
        vcn_id,
        "--display-name",
        f"{managed_prefix(config, runner)}-sl",
        "--ingress-security-rules",
        json.dumps(ingress_rules, separators=(",", ":")),
        "--egress-security-rules",
        json.dumps(egress_rules, separators=(",", ":")),
        "--wait-for-state",
        "AVAILABLE",
        "--output",
        "json",
    ]


def subnet_create_command(
    config: RunnerConfig,
    runner: Runner,
    vcn_id: str,
    route_table_id: str,
    security_list_id: str,
) -> list[str]:
    return [
        *oci_base(config),
        "network",
        "subnet",
        "create",
        "--region",
        runner.region,
        "--compartment-id",
        config.compartment_id,
        "--vcn-id",
        vcn_id,
        "--display-name",
        f"{managed_prefix(config, runner)}-public-subnet",
        "--dns-label",
        str(config.network.get("subnet_dns_label") or "runner")[:15],
        "--cidr-block",
        managed_cidr(config, "subnet_cidr", "10.91.1.0/24"),
        "--route-table-id",
        route_table_id,
        "--security-list-ids",
        json.dumps([security_list_id]),
        "--prohibit-public-ip-on-vnic",
        "false",
        "--wait-for-state",
        "AVAILABLE",
        "--output",
        "json",
    ]


def launch_command(config: RunnerConfig, runner: Runner, cloud_init: str) -> list[str]:
    metadata = {"user_data": base64.b64encode(cloud_init.encode("utf-8")).decode("ascii")}
    if runner.ssh_public_key:
        metadata["ssh_authorized_keys"] = runner.ssh_public_key
    shape = runner.shape or config.shape
    shape_config = runner.shape_config or config.shape_config
    command = [
        *oci_base(config),
        "compute",
        "instance",
        "launch",
        "--region",
        runner.region,
        "--compartment-id",
        config.compartment_id,
        "--availability-domain",
        runner.availability_domain,
        "--display-name",
        f"genai-benchmark-{runner.source_label}",
        "--shape",
        shape,
        "--image-id",
        runner.image_id,
        "--subnet-id",
        runner.subnet_id,
        "--assign-public-ip",
        "true",
        "--metadata",
        json.dumps(metadata, separators=(",", ":")),
        "--freeform-tags",
        json.dumps(
            {"genai-benchmark": "ephemeral-runner", "source_label": runner.source_label},
            separators=(",", ":"),
        ),
        "--output",
        "json",
    ]
    if shape_config:
        command.extend(["--shape-config", json.dumps(shape_config, separators=(",", ":"))])
    return command


def terminate_command(config: RunnerConfig, runner: Runner, instance_id: str) -> list[str]:
    return [
        *oci_base(config),
        "compute",
        "instance",
        "terminate",
        "--region",
        runner.region,
        "--instance-id",
        instance_id,
        "--force",
        "--wait-for-state",
        "TERMINATED",
    ]


def dynamic_group_name(config: RunnerConfig, runner: Runner) -> str:
    return f"{managed_prefix(config, runner).replace('-', '_')}_dg"[:100]


def policy_name(config: RunnerConfig, runner: Runner) -> str:
    return f"{managed_prefix(config, runner).replace('-', '_')}_policy"[:100]


def iam_compartment_id(config: RunnerConfig) -> str:
    return str(config.network.get("iam_compartment_id") or config.compartment_id)


def iam_base(config: RunnerConfig) -> list[str]:
    command = [*oci_base(config)]
    iam_region = config.network.get("iam_region")
    if iam_region:
        command.extend(["--region", str(iam_region)])
    return command


def dynamic_group_create_command(config: RunnerConfig, runner: Runner, instance_id: str) -> list[str]:
    return [
        *iam_base(config),
        "iam",
        "dynamic-group",
        "create",
        "--compartment-id",
        iam_compartment_id(config),
        "--name",
        dynamic_group_name(config, runner),
        "--description",
        f"Ephemeral GenAI benchmark runner for {runner.source_label}",
        "--matching-rule",
        f"instance.id = '{instance_id}'",
        "--wait-for-state",
        "ACTIVE",
        "--output",
        "json",
    ]


def combine_dynamic_group_matching_rule(base_rule: str, instance_id: str) -> str:
    stripped = base_rule.strip()
    if stripped.startswith("Any {") and stripped.endswith("}"):
        inner = stripped[len("Any {") : -1].strip()
        return f"Any {{{inner}, instance.id = '{instance_id}'}}"
    return f"Any {{{stripped}, instance.id = '{instance_id}'}}"


def existing_dynamic_group_update_enabled(config: RunnerConfig) -> bool:
    if not config.network.get("existing_dynamic_group_id"):
        return False
    return bool(config.network.get("existing_dynamic_group_update", True))


def dynamic_group_update_command(config: RunnerConfig, instance_id: str) -> list[str]:
    dynamic_group_id = require_string(config.network, "existing_dynamic_group_id", "network")
    base_rule = require_string(config.network, "existing_dynamic_group_base_matching_rule", "network")
    matching_rule = combine_dynamic_group_matching_rule(base_rule, instance_id)
    return [
        *iam_base(config),
        "iam",
        "dynamic-group",
        "update",
        "--dynamic-group-id",
        dynamic_group_id,
        "--matching-rule",
        matching_rule,
        "--force",
        "--wait-for-state",
        "ACTIVE",
        "--output",
        "json",
    ]


def dynamic_group_restore_command(config: RunnerConfig) -> list[str]:
    dynamic_group_id = require_string(config.network, "existing_dynamic_group_id", "network")
    base_rule = require_string(config.network, "existing_dynamic_group_base_matching_rule", "network")
    return [
        *iam_base(config),
        "iam",
        "dynamic-group",
        "update",
        "--dynamic-group-id",
        dynamic_group_id,
        "--matching-rule",
        base_rule,
        "--force",
        "--wait-for-state",
        "ACTIVE",
        "--output",
        "json",
    ]


def policy_dynamic_group_name(config: RunnerConfig, runner: Runner) -> str:
    return str(config.network.get("existing_dynamic_group_name") or dynamic_group_name(config, runner))


def policy_create_command(config: RunnerConfig, runner: Runner) -> list[str]:
    group_name = policy_dynamic_group_name(config, runner)
    statements = [
        f"Allow dynamic-group {group_name} to use generative-ai-family in tenancy",
        f"Allow dynamic-group {group_name} to manage objects in tenancy where target.bucket.name = '{config.bucket_name}'",
    ]
    return [
        *iam_base(config),
        "iam",
        "policy",
        "create",
        "--compartment-id",
        iam_compartment_id(config),
        "--name",
        policy_name(config, runner),
        "--description",
        f"Ephemeral GenAI benchmark permissions for {runner.source_label}",
        "--statements",
        json.dumps(statements),
        "--wait-for-state",
        "ACTIVE",
        "--output",
        "json",
    ]


def policy_delete_command(config: RunnerConfig, policy_id: str) -> list[str]:
    return [*iam_base(config), "iam", "policy", "delete", "--policy-id", policy_id, "--force"]


def dynamic_group_delete_command(config: RunnerConfig, dynamic_group_id: str) -> list[str]:
    return [*iam_base(config), "iam", "dynamic-group", "delete", "--dynamic-group-id", dynamic_group_id, "--force"]


def subnet_delete_command(config: RunnerConfig, runner: Runner, subnet_id: str) -> list[str]:
    return [*oci_base(config), "network", "subnet", "delete", "--region", runner.region, "--subnet-id", subnet_id, "--force"]


def security_list_delete_command(config: RunnerConfig, runner: Runner, security_list_id: str) -> list[str]:
    return [
        *oci_base(config),
        "network",
        "security-list",
        "delete",
        "--region",
        runner.region,
        "--security-list-id",
        security_list_id,
        "--force",
    ]


def route_table_delete_command(config: RunnerConfig, runner: Runner, route_table_id: str) -> list[str]:
    return [
        *oci_base(config),
        "network",
        "route-table",
        "delete",
        "--region",
        runner.region,
        "--rt-id",
        route_table_id,
        "--force",
    ]


def internet_gateway_delete_command(config: RunnerConfig, runner: Runner, internet_gateway_id: str) -> list[str]:
    return [
        *oci_base(config),
        "network",
        "internet-gateway",
        "delete",
        "--region",
        runner.region,
        "--ig-id",
        internet_gateway_id,
        "--force",
    ]


def vcn_delete_command(config: RunnerConfig, runner: Runner, vcn_id: str) -> list[str]:
    return [*oci_base(config), "network", "vcn", "delete", "--region", runner.region, "--vcn-id", vcn_id, "--force"]


def object_name(runner: Runner, name: str) -> str:
    return f"runs/{runner.source_label}/{name}"


def completion_marker_name(config: RunnerConfig, runner: Runner) -> str:
    return object_name(runner, f"_{report_name(config, runner)}.complete.txt")


def object_head_command(config: RunnerConfig, runner: Runner) -> list[str]:
    return [
        *oci_base(config),
        "os",
        "object",
        "head",
        "--region",
        config.object_storage_region,
        "--bucket-name",
        config.bucket_name,
        "--name",
        completion_marker_name(config, runner),
    ]


def object_get_command(config: RunnerConfig, runner: Runner, extension: str, output_dir: Path) -> list[str]:
    name = f"{report_name(config, runner)}.{extension}"
    return [
        *oci_base(config),
        "os",
        "object",
        "get",
        "--region",
        config.object_storage_region,
        "--bucket-name",
        config.bucket_name,
        "--name",
        object_name(runner, name),
        "--file",
        str(output_dir / name),
    ]


def shell_join(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=check, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        details = []
        if exc.stdout:
            details.append(f"stdout:\n{exc.stdout.strip()}")
        if exc.stderr:
            details.append(f"stderr:\n{exc.stderr.strip()}")
        suffix = "\n" + "\n".join(details) if details else ""
        raise RuntimeError(f"Command failed with exit code {exc.returncode}: {shell_join(command)}{suffix}") from exc


def parse_instance_id(output: str) -> str:
    return parse_resource_id(output)


def parse_resource_id(output: str) -> str:
    data = json.loads(output)
    resource_id = data.get("data", {}).get("id")
    if not resource_id:
        raise RuntimeError("Unable to find resource id in OCI CLI output.")
    return resource_id


def execute_or_print(command: Sequence[str], dry_run: bool) -> str | None:
    if dry_run:
        print(shell_join(command))
        return None
    return parse_resource_id(run_command(command).stdout)


def provision_managed_network(config: RunnerConfig, runner: Runner, dry_run: bool) -> ManagedResources:
    placeholder = "<created-by-previous-command>"
    vcn_id = execute_or_print(vcn_create_command(config, runner), dry_run) or placeholder
    internet_gateway_id = (
        execute_or_print(internet_gateway_create_command(config, runner, vcn_id), dry_run) or placeholder
    )
    route_table_id = (
        execute_or_print(route_table_create_command(config, runner, vcn_id, internet_gateway_id), dry_run)
        or placeholder
    )
    security_list_id = (
        execute_or_print(security_list_create_command(config, runner, vcn_id), dry_run) or placeholder
    )
    subnet_id = (
        execute_or_print(subnet_create_command(config, runner, vcn_id, route_table_id, security_list_id), dry_run)
        or placeholder
    )
    return ManagedResources(
        vcn_id=vcn_id,
        internet_gateway_id=internet_gateway_id,
        route_table_id=route_table_id,
        security_list_id=security_list_id,
        subnet_id=subnet_id,
    )


def create_instance_policy(
    config: RunnerConfig,
    runner: Runner,
    resources: ManagedResources,
    dry_run: bool,
) -> ManagedResources:
    if resources.instance_id is None:
        raise RuntimeError("Instance id is required before creating IAM resources.")
    if existing_dynamic_group_update_enabled(config):
        execute_or_print(dynamic_group_update_command(config, resources.instance_id), dry_run)
        dynamic_group_id = None
    elif config.network.get("existing_dynamic_group_id"):
        dynamic_group_id = None
    else:
        dynamic_group_id = (
            execute_or_print(dynamic_group_create_command(config, runner, resources.instance_id), dry_run)
            or "<created-by-previous-command>"
        )
    if config.network.get("existing_policy_id"):
        policy_id = None
    else:
        policy_id = execute_or_print(policy_create_command(config, runner), dry_run) or "<created-by-previous-command>"
    return replace(resources, dynamic_group_id=dynamic_group_id, policy_id=policy_id)


def cleanup_managed_resources(
    config: RunnerConfig,
    runner: Runner,
    resources: ManagedResources,
    dry_run: bool,
) -> None:
    commands: list[list[str]] = []
    if resources.instance_id:
        commands.append(terminate_command(config, runner, resources.instance_id))
    if resources.policy_id:
        commands.append(policy_delete_command(config, resources.policy_id))
    if existing_dynamic_group_update_enabled(config):
        commands.append(dynamic_group_restore_command(config))
    if resources.dynamic_group_id:
        commands.append(dynamic_group_delete_command(config, resources.dynamic_group_id))
    commands.extend(
        [
            subnet_delete_command(config, runner, resources.subnet_id),
            route_table_delete_command(config, runner, resources.route_table_id),
            security_list_delete_command(config, runner, resources.security_list_id),
            internet_gateway_delete_command(config, runner, resources.internet_gateway_id),
            vcn_delete_command(config, runner, resources.vcn_id),
        ]
    )
    for command in commands:
        if dry_run:
            print(shell_join(command))
        else:
            run_command(command)


def save_resource_state(path: Path, runner: Runner, resources: ManagedResources) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source_label": runner.source_label, "resources": asdict(resources)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_resource_state(path: Path) -> tuple[str, ManagedResources]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    resources = payload.get("resources", {})
    return payload["source_label"], ManagedResources(**resources)


def run_managed_runner(
    config: RunnerConfig,
    runner: Runner,
    output_dir: Path,
    state_path: Path,
    poll_interval: int,
    timeout_seconds: int,
    keep_on_failure: bool,
    dry_run: bool,
) -> None:
    resources = provision_managed_network(config, runner, dry_run)
    managed_runner = replace(runner, subnet_id=resources.subnet_id)
    instance_id = launch_runner(config, managed_runner, dry_run) or "<created-by-previous-command>"
    resources = replace(resources, instance_id=instance_id)
    resources = create_instance_policy(config, managed_runner, resources, dry_run)
    if not dry_run:
        save_resource_state(state_path, managed_runner, resources)
    try:
        wait_for_completion(config, managed_runner, poll_interval, timeout_seconds, dry_run)
        collect_reports(config, managed_runner, output_dir, dry_run)
    except Exception:
        if keep_on_failure:
            if not dry_run:
                save_resource_state(state_path, managed_runner, resources)
            print(f"Keeping managed resources for inspection. State: {state_path}", file=sys.stderr)
            raise
        raise
    finally:
        if not keep_on_failure:
            cleanup_managed_resources(config, managed_runner, resources, dry_run)


def wait_for_completion(
    config: RunnerConfig,
    runner: Runner,
    poll_interval: int,
    timeout_seconds: int,
    dry_run: bool,
) -> None:
    command = object_head_command(config, runner)
    if dry_run:
        print(shell_join(command))
        return
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = run_command(command, check=False)
        if result.returncode == 0:
            return
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for Object Storage marker: {completion_marker_name(config, runner)}")


def collect_reports(config: RunnerConfig, runner: Runner, output_dir: Path, dry_run: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("json", "md"):
        command = object_get_command(config, runner, extension, output_dir)
        if dry_run:
            print(shell_join(command))
        else:
            run_command(command)


def suite_name(config: RunnerConfig, explicit_name: str | None = None) -> str:
    return explicit_name or str(
        config.benchmark.get("suite_name") or config.benchmark.get("report_suffix") or "managed-suite"
    )


def suite_state_path(base_state_path: Path, runner: Runner) -> Path:
    return base_state_path.parent / f"{runner.source_label}-state.json"


def validate_managed_suite_config(config: RunnerConfig, runners: Sequence[Runner], parallelism: int) -> None:
    if parallelism < 1:
        raise ConfigError("--parallelism must be at least 1.")
    if len(runners) > 1 and existing_dynamic_group_update_enabled(config):
        raise ConfigError(
            "run-managed-suite with multiple runners requires network.existing_dynamic_group_update=false "
            "when reusing an existing Dynamic Group. Add a compartment-level runner rule to that group first."
        )


def summarize_report_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    config = data.get("benchmark_config", {})
    results = data.get("results", [])
    successes = [item for item in results if not item.get("error")]
    failures = [item for item in results if item.get("error")]
    by_region: dict[str, list[float]] = defaultdict(list)
    for item in successes:
        latency = item.get("latency_seconds")
        if isinstance(latency, (int, float)):
            by_region[str(item.get("region") or "unknown")].append(float(latency))
    latencies = [
        float(item["latency_seconds"])
        for item in successes
        if isinstance(item.get("latency_seconds"), (int, float))
    ]
    return {
        "path": path,
        "source_label": config.get("source_label") or path.stem,
        "target_regions": config.get("regions") or [],
        "attempts": len(results),
        "successes": len(successes),
        "failures": len(failures),
        "avg_latency": (sum(latencies) / len(latencies)) if latencies else None,
        "target_avg_latency": {
            region: sum(values) / len(values)
            for region, values in sorted(by_region.items())
            if values
        },
    }


def format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}s"


def write_suite_summary(
    config: RunnerConfig,
    runners: Sequence[Runner],
    output_dir: Path,
    name: str,
    run_statuses: dict[str, str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{name}-summary.md"
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    runners_by_label = {runner.source_label: runner for runner in runners}
    rows = []
    for runner in runners:
        json_path = output_dir / f"{report_name(config, runner)}.json"
        if json_path.exists():
            rows.append(summarize_report_file(json_path))
        else:
            rows.append(
                {
                    "path": json_path,
                    "source_label": runner.source_label,
                    "target_regions": config.target_regions,
                    "attempts": 0,
                    "successes": 0,
                    "failures": 0,
                    "avg_latency": None,
                    "target_avg_latency": {},
                }
            )
    total_attempts = sum(row["attempts"] for row in rows)
    total_successes = sum(row["successes"] for row in rows)
    total_failures = sum(row["failures"] for row in rows)
    lines = [
        f"# {name} Suite Summary",
        "",
        f"- Generated At: `{generated_at}`",
        f"- Target Regions: {', '.join(f'`{region}`' for region in config.target_regions)}",
        f"- Attempts: `{total_attempts}`",
        f"- Successes: `{total_successes}`",
        f"- Failures: `{total_failures}`",
        "",
        "## Source Runner Summary",
        "",
        "| Source | Status | Attempts | Successes | Failures | Avg Latency | JSON | Markdown |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        source_label = row["source_label"]
        runner = runners_by_label.get(source_label)
        json_name = f"{report_name(config, runner)}.json" if runner else row["path"].name
        md_name = json_name.removesuffix(".json") + ".md"
        lines.append(
            "| "
            f"`{source_label}` | "
            f"`{run_statuses.get(source_label, 'missing')}` | "
            f"{row['attempts']} | {row['successes']} | {row['failures']} | "
            f"{format_seconds(row['avg_latency'])} | "
            f"[json]({json_name}) | [md]({md_name}) |"
        )
    lines.extend(["", "## Target Region Average Latency", "", "| Source | Target Region | Avg Latency |", "| --- | --- | ---: |"])
    for row in rows:
        for region in config.target_regions:
            lines.append(
                f"| `{row['source_label']}` | `{region}` | "
                f"{format_seconds(row['target_avg_latency'].get(region))} |"
            )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def publish_docs(output_dir: Path, docs_dir: Path) -> None:
    reports = load_reports(output_dir)
    if reports:
        write_docs(reports, docs_dir)


def run_managed_suite(
    config: RunnerConfig,
    runners: Sequence[Runner],
    output_dir: Path,
    state_path: Path,
    poll_interval: int,
    timeout_seconds: int,
    keep_on_failure: bool,
    dry_run: bool,
    parallelism: int,
    name: str,
    docs_dir: Path,
    publish_site: bool,
) -> None:
    validate_managed_suite_config(config, runners, parallelism)
    if dry_run:
        for runner in runners:
            print(f"# Runner: {runner.source_label} ({runner.region})")
            run_managed_runner(
                config,
                runner,
                output_dir,
                suite_state_path(state_path, runner),
                poll_interval,
                timeout_seconds,
                keep_on_failure,
                dry_run=True,
            )
            print()
        print(f"# Would write suite summary: {output_dir / f'{name}-summary.md'}")
        if publish_site:
            print(f"# Would publish docs: {docs_dir}")
        return

    statuses: dict[str, str] = {}
    errors: dict[str, BaseException] = {}
    workers = min(parallelism, len(runners))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_managed_runner,
                config,
                runner,
                output_dir,
                suite_state_path(state_path, runner),
                poll_interval,
                timeout_seconds,
                keep_on_failure,
                False,
            ): runner
            for runner in runners
        }
        for future in concurrent.futures.as_completed(futures):
            runner = futures[future]
            try:
                future.result()
            except BaseException as exc:
                statuses[runner.source_label] = "failed"
                errors[runner.source_label] = exc
                print(f"Runner failed: {runner.source_label}: {exc}", file=sys.stderr)
            else:
                statuses[runner.source_label] = "succeeded"

    summary_path = write_suite_summary(config, runners, output_dir, name, statuses)
    print(f"Wrote suite summary: {summary_path}")
    if publish_site:
        publish_docs(output_dir, docs_dir)
        print(f"Wrote docs site to {docs_dir}")
    if errors:
        failed = ", ".join(sorted(errors))
        raise RuntimeError(f"Managed suite failed for runner(s): {failed}")


def print_plan(config: RunnerConfig, runners: Iterable[Runner]) -> None:
    for runner in runners:
        cloud_init = render_cloud_init(config, runner)
        print(f"# Runner: {runner.source_label} ({runner.region})")
        print(shell_join(launch_command(config, runner, cloud_init)))
        print(shell_join(object_head_command(config, runner)))
        print(shell_join(object_get_command(config, runner, "json", Path("runs"))))
        print(shell_join(object_get_command(config, runner, "md", Path("runs"))))
        print()


def launch_runner(config: RunnerConfig, runner: Runner, dry_run: bool) -> str | None:
    command = launch_command(config, runner, render_cloud_init(config, runner))
    if dry_run:
        print(shell_join(command))
        return None
    result = run_command(command)
    instance_id = parse_instance_id(result.stdout)
    print(f"Launched {runner.source_label}: {instance_id}")
    return instance_id


def terminate_runner(config: RunnerConfig, runner: Runner, instance_id: str, dry_run: bool) -> None:
    command = terminate_command(config, runner, instance_id)
    if dry_run:
        print(shell_join(command))
    else:
        run_command(command)
        print(f"Terminated {runner.source_label}: {instance_id}")


def run_runner(
    config: RunnerConfig,
    runner: Runner,
    output_dir: Path,
    poll_interval: int,
    timeout_seconds: int,
    keep_on_failure: bool,
    dry_run: bool,
) -> None:
    instance_id = launch_runner(config, runner, dry_run)
    try:
        wait_for_completion(config, runner, poll_interval, timeout_seconds, dry_run)
        collect_reports(config, runner, output_dir, dry_run)
    except Exception:
        if instance_id and keep_on_failure:
            print(f"Keeping failed runner for inspection: {instance_id}", file=sys.stderr)
            raise
        raise
    finally:
        if instance_id and not keep_on_failure:
            terminate_runner(config, runner, instance_id, dry_run=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage ephemeral OCI benchmark runner VMs.")
    parser.add_argument("--config", required=True, help="Path to ephemeral runner JSON config.")
    parser.add_argument("--runner", action="append", default=[], help="source_label to operate on. Repeatable.")
    parser.add_argument(
        "--action",
        choices=(
            "plan",
            "launch",
            "collect",
            "terminate",
            "run",
            "provision-network",
            "run-managed",
            "run-managed-suite",
            "cleanup-managed",
        ),
        default="plan",
        help="Operation to perform. plan only prints commands.",
    )
    parser.add_argument("--instance-id", help="Required for --action terminate.")
    parser.add_argument("--output-dir", default="runs", help="Where collected reports are written.")
    parser.add_argument(
        "--resource-state",
        default="runs/ephemeral-runner-state.json",
        help="Path for managed resource state used by run-managed and cleanup-managed.",
    )
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--parallelism", type=int, default=3, help="Max concurrent runners for run-managed-suite.")
    parser.add_argument("--suite-name", help="Name for the suite summary report. Defaults to benchmark.suite_name or report_suffix.")
    parser.add_argument("--docs-dir", default="docs", help="Where docs artifacts are written after run-managed-suite.")
    parser.add_argument("--no-publish-site", action="store_true", help="Skip docs dashboard generation after run-managed-suite.")
    parser.add_argument("--keep-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print OCI commands without executing them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    runners = selected_runners(config, args.runner)
    output_dir = Path(args.output_dir)

    if args.action == "plan":
        print_plan(config, runners)
        return 0

    if args.action == "terminate":
        if not args.instance_id:
            raise SystemExit("--instance-id is required for --action terminate.")
        if len(runners) != 1:
            raise SystemExit("--action terminate requires exactly one --runner.")
        terminate_runner(config, runners[0], args.instance_id, args.dry_run)
        return 0

    if args.action == "cleanup-managed":
        if len(runners) != 1:
            raise SystemExit("--action cleanup-managed requires exactly one --runner.")
        source_label, resources = load_resource_state(Path(args.resource_state))
        if source_label != runners[0].source_label:
            raise SystemExit(
                f"Resource state belongs to {source_label}, not {runners[0].source_label}."
            )
        cleanup_managed_resources(config, runners[0], resources, args.dry_run)
        return 0

    if args.action == "run-managed-suite":
        run_managed_suite(
            config,
            runners,
            output_dir,
            Path(args.resource_state),
            args.poll_interval,
            args.timeout_seconds,
            args.keep_on_failure,
            args.dry_run,
            args.parallelism,
            suite_name(config, args.suite_name),
            Path(args.docs_dir),
            not args.no_publish_site,
        )
        return 0

    for runner in runners:
        if args.action == "launch":
            launch_runner(config, runner, args.dry_run)
        elif args.action == "collect":
            wait_for_completion(config, runner, args.poll_interval, args.timeout_seconds, args.dry_run)
            collect_reports(config, runner, output_dir, args.dry_run)
        elif args.action == "run":
            run_runner(
                config,
                runner,
                output_dir,
                args.poll_interval,
                args.timeout_seconds,
                args.keep_on_failure,
                args.dry_run,
            )
        elif args.action == "provision-network":
            provision_managed_network(config, runner, args.dry_run)
        elif args.action == "run-managed":
            run_managed_runner(
                config,
                runner,
                output_dir,
                Path(args.resource_state),
                args.poll_interval,
                args.timeout_seconds,
                args.keep_on_failure,
                args.dry_run,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
