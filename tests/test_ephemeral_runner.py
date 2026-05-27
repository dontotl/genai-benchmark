from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from genai_benchmark.ephemeral_runner import (
    ConfigError,
    benchmark_args,
    combine_dynamic_group_matching_rule,
    completion_marker_name,
    dynamic_group_restore_command,
    dynamic_group_update_command,
    launch_command,
    load_config,
    object_get_command,
    object_head_command,
    parse_object_names,
    policy_create_command,
    progress_object_delete_command,
    progress_object_list_command,
    progress_prefix,
    progress_status_name,
    run_managed_suite,
    security_list_create_command,
    suite_state_path,
    subnet_create_command,
    vcn_create_command,
    render_cloud_init,
    report_name,
    selected_runners,
    validate_managed_suite_config,
    write_suite_summary,
)


def write_config(payload: dict) -> Path:
    fd, raw_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    Path(raw_path).unlink()
    path = Path(raw_path)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def sample_config() -> dict:
    return {
        "compartment_id": "ocid1.compartment.oc1..example",
        "benchmark_compartment_id": "ocid1.compartment.oc1..benchmark",
        "bucket_name": "bench-results",
        "object_storage_region": "ap-osaka-1",
        "repo_url": "https://github.com/example/genai-benchmark.git",
        "repo_ref": "main",
        "control_profile": "ADMIN",
        "runner_profile": "DEFAULT",
        "runner_auth_method": "user_principal",
        "runner_user": "opc",
        "work_dir": "/home/opc/genai-benchmark",
        "shape": "VM.Standard.E4.Flex",
        "shape_config": {"ocpus": 1, "memoryInGBs": 8},
        "target_regions": ["ap-osaka-1", "us-chicago-1", "eu-frankfurt-1"],
        "benchmark": {
            "repeats": 3,
            "concurrency_levels": "1,5,10",
            "prompts": "prompts/sample_prompts.jsonl",
            "report_suffix": "global-r3",
        },
        "runners": [
            {
                "region": "ap-osaka-1",
                "source_label": "ap-osaka-runner",
                "availability_domain": "AD-1",
                "subnet_id": "subnet-osaka",
                "image_id": "image-osaka",
            }
        ],
    }


class EphemeralRunnerConfigTest(unittest.TestCase):
    def test_loads_config_and_selects_runner(self) -> None:
        config = load_config(write_config(sample_config()))

        self.assertEqual(config.bucket_name, "bench-results")
        self.assertEqual(config.control_profile, "ADMIN")
        self.assertEqual(config.runners[0].source_label, "ap-osaka-runner")
        self.assertEqual(selected_runners(config, ["ap-osaka-runner"])[0].region, "ap-osaka-1")

    def test_rejects_unknown_runner(self) -> None:
        config = load_config(write_config(sample_config()))

        with self.assertRaises(ConfigError):
            selected_runners(config, ["missing"])

    def test_loads_four_source_managed_example(self) -> None:
        config = load_config(Path("configs/ephemeral-4source-managed.example.json"))

        self.assertEqual(config.target_regions, ["ap-osaka-1", "us-chicago-1", "eu-frankfurt-1"])
        self.assertEqual(
            [runner.source_label for runner in config.runners],
            [
                "ap-osaka-runner",
                "ap-seoul-runner",
                "us-chicago-runner",
                "eu-frankfurt-runner",
            ],
        )
        self.assertFalse(config.network["existing_dynamic_group_update"])
        self.assertEqual(config.shape_config, {"ocpus": 4, "memoryInGBs": 16})
        self.assertEqual(config.benchmark["prompts"], "prompts/chat_nl2sql_workloads.jsonl")
        self.assertEqual(config.benchmark["concurrency_levels"], "1,5,10")
        self.assertTrue(config.benchmark["streaming"])
        self.assertTrue(config.benchmark["load_test"])
        self.assertTrue(config.benchmark["include_experimental"])
        self.assertEqual(config.benchmark["families"], ["openai", "gemini", "grok", "meta"])


class EphemeralRunnerCommandTest(unittest.TestCase):
    def test_builds_benchmark_args(self) -> None:
        config = load_config(write_config(sample_config()))
        runner = config.runners[0]

        args = benchmark_args(config, runner)

        self.assertIn("--source-label", args)
        self.assertIn("ap-osaka-runner", args)
        self.assertEqual(args.count("--region"), 3)
        self.assertIn("--concurrency-levels", args)
        self.assertIn("1,5,10", args)
        self.assertIn("--compartment-id", args)
        self.assertIn("ocid1.compartment.oc1..benchmark", args)
        self.assertEqual(report_name(config, runner), "ap-osaka-runner-global-r3")

    def test_builds_load_test_benchmark_args(self) -> None:
        payload = sample_config()
        payload["benchmark"]["load_test"] = True
        payload["benchmark"]["request_timeout"] = 10
        config = load_config(write_config(payload))

        args = benchmark_args(config, config.runners[0])

        self.assertIn("--load-test", args)
        self.assertIn("--request-timeout", args)
        self.assertIn("10", args)

    def test_renders_cloud_init_with_user_principal_profile(self) -> None:
        config = load_config(write_config(sample_config()))
        runner = config.runners[0]

        rendered = render_cloud_init(config, runner)

        self.assertIn("dnf install -y git python3.11", rendered)
        self.assertIn("--profile DEFAULT", rendered)
        self.assertIn("--source-label ap-osaka-runner", rendered)
        self.assertIn('BUCKET_NAME=bench-results', rendered)
        self.assertIn("oci os ns get --profile DEFAULT", rendered)

    def test_builds_launch_and_collect_commands(self) -> None:
        config = load_config(write_config(sample_config()))
        runner = config.runners[0]

        launch = launch_command(config, runner, "#cloud-init")
        get_json = object_get_command(config, runner, "json", Path("runs"))

        self.assertEqual(launch[:3], ["oci", "--profile", "ADMIN"])
        self.assertIn("--shape-config", launch)
        self.assertIn("genai-benchmark-ap-osaka-runner", launch)
        self.assertEqual(get_json[:3], ["oci", "--profile", "ADMIN"])
        self.assertIn("runs/ap-osaka-runner/ap-osaka-runner-global-r3.json", get_json)
        self.assertIn("runs/ap-osaka-runner-global-r3.json", get_json)

    def test_renders_cloud_init_with_instance_principal(self) -> None:
        payload = sample_config()
        payload["runner_auth_method"] = "instance_principal"
        config = load_config(write_config(payload))
        runner = config.runners[0]

        rendered = render_cloud_init(config, runner)

        self.assertIn("--auth-method instance_principal", rendered)
        self.assertIn("oci os ns get --auth instance_principal", rendered)
        self.assertIn("--auth instance_principal", rendered)
        self.assertIn("COMPLETION_MARKER=runs/ap-osaka-runner/_ap-osaka-runner-global-r3.complete.txt", rendered)
        self.assertIn("PROGRESS_PREFIX=runs/ap-osaka-runner/progress/ap-osaka-runner-global-r3/", rendered)
        self.assertIn("--progress-file", rendered)

    def test_progress_object_names_and_cleanup_commands(self) -> None:
        config = load_config(write_config(sample_config()))
        runner = config.runners[0]

        self.assertEqual(
            progress_prefix(config, runner),
            "runs/ap-osaka-runner/progress/ap-osaka-runner-global-r3/",
        )
        self.assertEqual(
            progress_status_name(config, runner),
            "runs/ap-osaka-runner/progress/ap-osaka-runner-global-r3/status.json",
        )
        self.assertIn(progress_prefix(config, runner), progress_object_list_command(config, runner))
        self.assertIn("runs/ap-osaka-runner/progress/ap-osaka-runner-global-r3/status.json", progress_object_delete_command(config, progress_status_name(config, runner)))

    def test_parses_object_list_shapes(self) -> None:
        self.assertEqual(
            parse_object_names(
                json.dumps(
                    {
                        "data": [
                            {"name": "runs/a/progress/r/status.json"},
                            {"name": "runs/a/progress/r/benchmark.log"},
                        ]
                    }
                )
            ),
            ["runs/a/progress/r/status.json", "runs/a/progress/r/benchmark.log"],
        )
        self.assertEqual(
            parse_object_names(json.dumps({"data": {"objects": [{"name": "nested"}]}})),
            ["nested"],
        )

    def test_completion_marker_is_report_specific(self) -> None:
        config = load_config(write_config(sample_config()))
        runner = config.runners[0]

        marker = completion_marker_name(config, runner)
        head = object_head_command(config, runner)

        self.assertEqual(marker, "runs/ap-osaka-runner/_ap-osaka-runner-global-r3.complete.txt")
        self.assertIn(marker, head)

    def test_builds_managed_network_and_policy_commands(self) -> None:
        payload = sample_config()
        payload["network"] = {"ssh_source_cidr": "203.0.113.10/32"}
        payload["runner_auth_method"] = "instance_principal"
        config = load_config(write_config(payload))
        runner = config.runners[0]

        vcn = vcn_create_command(config, runner)
        security_list = security_list_create_command(config, runner, "vcn-1")
        subnet = subnet_create_command(config, runner, "vcn-1", "rt-1", "sl-1")
        policy = policy_create_command(config, runner)

        self.assertIn("--cidr-block", vcn)
        self.assertIn("10.91.0.0/16", vcn)
        self.assertIn("203.0.113.10/32", " ".join(security_list))
        self.assertIn("--prohibit-public-ip-on-vnic", subnet)
        self.assertIn("use generative-ai-family", " ".join(policy))

    def test_can_disable_ssh_ingress_for_managed_runner(self) -> None:
        payload = sample_config()
        payload["network"] = {"enable_ssh": False}
        config = load_config(write_config(payload))
        runner = config.runners[0]

        security_list = security_list_create_command(config, runner, "vcn-1")

        self.assertIn("--ingress-security-rules", security_list)
        self.assertIn("[]", security_list)
        self.assertNotIn("destinationPortRange", " ".join(security_list))

    def test_iam_commands_can_use_root_compartment_override(self) -> None:
        payload = sample_config()
        payload["network"] = {
            "iam_compartment_id": "ocid1.tenancy.oc1..root",
            "iam_region": "us-ashburn-1",
        }
        config = load_config(write_config(payload))
        runner = config.runners[0]

        dynamic_group = " ".join(policy_create_command(config, runner))

        self.assertIn("ocid1.tenancy.oc1..root", dynamic_group)
        self.assertIn("us-ashburn-1", dynamic_group)

    def test_can_update_existing_dynamic_group_for_instance(self) -> None:
        payload = sample_config()
        payload["network"] = {
            "iam_region": "us-ashburn-1",
            "existing_dynamic_group_id": "ocid1.dynamicgroup.oc1..existing",
            "existing_dynamic_group_base_matching_rule": "instance.compartment.id = 'ocid1.compartment.oc1..base'",
        }
        config = load_config(write_config(payload))

        command = " ".join(dynamic_group_update_command(config, "ocid1.instance.oc1.ap-seoul-1.example"))

        self.assertIn("dynamic-group update", command)
        self.assertIn("ocid1.dynamicgroup.oc1..existing", command)
        self.assertIn("ocid1.instance.oc1.ap-seoul-1.example", command)

    def test_can_restore_existing_dynamic_group_base_rule(self) -> None:
        payload = sample_config()
        payload["network"] = {
            "iam_region": "us-ashburn-1",
            "existing_dynamic_group_id": "ocid1.dynamicgroup.oc1..existing",
            "existing_dynamic_group_base_matching_rule": "Any {instance.compartment.id = 'ocid1.compartment.oc1..base'}",
        }
        config = load_config(write_config(payload))

        command = " ".join(dynamic_group_restore_command(config))

        self.assertIn("dynamic-group update", command)
        self.assertIn("Any {instance.compartment.id = 'ocid1.compartment.oc1..base'}", command)
        self.assertNotIn("ocid1.instance", command)

    def test_wraps_existing_any_matching_rule_without_nesting(self) -> None:
        matching_rule = combine_dynamic_group_matching_rule(
            "Any {instance.compartment.id = 'ocid1.compartment.oc1..base'}",
            "ocid1.instance.oc1.ap-seoul-1.example",
        )

        self.assertEqual(
            matching_rule,
            "Any {instance.compartment.id = 'ocid1.compartment.oc1..base', "
            "instance.id = 'ocid1.instance.oc1.ap-seoul-1.example'}",
        )

    def test_policy_can_target_existing_dynamic_group_name(self) -> None:
        payload = sample_config()
        payload["network"] = {"existing_dynamic_group_name": "ociexplain-dyn-group"}
        config = load_config(write_config(payload))
        runner = config.runners[0]

        policy = " ".join(policy_create_command(config, runner))

        self.assertIn("Allow dynamic-group ociexplain-dyn-group", policy)

    def test_rejects_parallel_suite_with_existing_dynamic_group_updates(self) -> None:
        payload = sample_config()
        payload["network"] = {
            "existing_dynamic_group_id": "ocid1.dynamicgroup.oc1..existing",
            "existing_dynamic_group_base_matching_rule": "instance.compartment.id = 'ocid1.compartment.oc1..base'",
        }
        payload["runners"].append(
            {
                "region": "us-chicago-1",
                "source_label": "us-chicago-runner",
                "availability_domain": "AD-1",
                "subnet_id": "subnet-chicago",
                "image_id": "image-chicago",
            }
        )
        config = load_config(write_config(payload))

        with self.assertRaises(ConfigError):
            validate_managed_suite_config(config, config.runners, parallelism=2)

    def test_suite_state_path_uses_runner_label(self) -> None:
        config = load_config(write_config(sample_config()))
        runner = config.runners[0]

        self.assertEqual(
            suite_state_path(Path("runs/ephemeral-runner-state.json"), runner),
            Path("runs/ap-osaka-runner-state.json"),
        )

    def test_run_managed_suite_dry_run_allows_compartment_rule_mode(self) -> None:
        payload = sample_config()
        payload["network"] = {
            "existing_dynamic_group_id": "ocid1.dynamicgroup.oc1..existing",
            "existing_dynamic_group_update": False,
            "existing_policy_id": "ocid1.policy.oc1..existing",
        }
        config = load_config(write_config(payload))

        with contextlib.redirect_stdout(io.StringIO()):
            run_managed_suite(
                config,
                config.runners,
                Path("runs"),
                Path("runs/ephemeral-runner-state.json"),
                poll_interval=30,
                timeout_seconds=3600,
                keep_on_failure=False,
                keep_progress_logs=False,
                dry_run=True,
                parallelism=3,
                name="global-r3",
                docs_dir=Path("docs"),
                publish_site=True,
            )

    def test_writes_suite_summary_from_collected_reports(self) -> None:
        payload = sample_config()
        config = load_config(write_config(payload))
        runner = config.runners[0]
        with tempfile.TemporaryDirectory() as raw_dir:
            output_dir = Path(raw_dir)
            report_path = output_dir / f"{report_name(config, runner)}.json"
            report_path.write_text(
                json.dumps(
                    {
                        "benchmark_config": {
                            "source_label": runner.source_label,
                            "regions": ["ap-osaka-1", "us-chicago-1"],
                        },
                        "results": [
                            {"region": "ap-osaka-1", "latency_seconds": 1.0, "error": None},
                            {"region": "us-chicago-1", "latency_seconds": 3.0, "error": None},
                            {"region": "us-chicago-1", "latency_seconds": 2.0, "error": "failed"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = write_suite_summary(
                config,
                [runner],
                output_dir,
                "global-r3",
                {runner.source_label: "succeeded"},
            )

            rendered = summary.read_text(encoding="utf-8")

        self.assertIn("# global-r3 Suite Summary", rendered)
        self.assertIn("| `ap-osaka-runner` | `succeeded` | 3 | 2 | 1 | 2.000s |", rendered)
        self.assertIn("| `ap-osaka-runner` | `us-chicago-1` | 3.000s |", rendered)


if __name__ == "__main__":
    unittest.main()
