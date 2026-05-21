from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from genai_benchmark.ephemeral_runner import (
    ConfigError,
    benchmark_args,
    combine_dynamic_group_matching_rule,
    dynamic_group_restore_command,
    dynamic_group_update_command,
    launch_command,
    load_config,
    object_get_command,
    policy_create_command,
    security_list_create_command,
    subnet_create_command,
    vcn_create_command,
    render_cloud_init,
    report_name,
    selected_runners,
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


if __name__ == "__main__":
    unittest.main()
