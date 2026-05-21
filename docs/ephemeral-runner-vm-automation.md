# Ephemeral Runner VM Automation Design

## Purpose

This design extends the benchmark from a single client location into a global
latency surface test.

The target analysis axis is:

```text
source_label x target_region x model x workload x concurrency
```

Each runner VM is created in a specific OCI region, calls multiple OCI
Generative AI target regions, uploads its reports to Object Storage, and is
then terminated.

This keeps the benchmark environment clean, reduces idle cost, and captures
the latency that an application server would see from each regional point of
presence.

## Recommended Architecture

```text
Control node
  |
  |-- creates ephemeral runner VM in ap-osaka-1
  |     |-- runs benchmark against ap-osaka-1, us-chicago-1, eu-frankfurt-1
  |     |-- uploads reports to Object Storage
  |     `-- terminates
  |
  |-- creates ephemeral runner VM in us-chicago-1
  |     |-- runs benchmark against ap-osaka-1, us-chicago-1, eu-frankfurt-1
  |     |-- uploads reports to Object Storage
  |     `-- terminates
  |
  `-- creates ephemeral runner VM in eu-frankfurt-1
        |-- runs benchmark against ap-osaka-1, us-chicago-1, eu-frankfurt-1
        |-- uploads reports to Object Storage
        `-- terminates
```

Default runner regions:

- `ap-osaka-1`
- `us-chicago-1`
- `eu-frankfurt-1`

Default GenAI target regions:

- `ap-osaka-1`
- `us-chicago-1`
- `eu-frankfurt-1`

## OCI Resources

Required resources:

- One compartment for benchmark resources.
- One Object Storage bucket for benchmark reports.
- One VCN/subnet per runner region, or existing regional subnets.
- One small compute shape per runner VM.
- Oracle Linux 8 images are preferred for the managed runner smoke path. The
  cloud-init template installs Python 3.11 with `dnf`; avoid changing to newer
  OS images until the bootstrap path has been re-tested.
- For existing-subnet runners, OCI config/profile can be prepared for the runner
  OS user.
- For managed public runners, Dynamic Group and Policy are created for Instance
  Principal authentication.

Suggested variables:

```bash
COMPARTMENT_ID=<compartment-ocid>
BUCKET_NAME=<benchmark-result-bucket>
NAMESPACE=<object-storage-namespace>
REPO_URL=https://github.com/<owner>/<repo>.git
REPO_REF=main
CONTROL_PROFILE=DEFAULT
RUNNER_PROFILE=DEFAULT
RUNNER_AUTH_METHOD=instance_principal
RUNNER_USER=opc
SHAPE=<small-compute-shape>
IMAGE_ID_<REGION>=<image-ocid>
SUBNET_ID_<REGION>=<subnet-ocid>
```

Use small shapes for the first pass. The benchmark is primarily measuring
network path and hosted model behavior, not local CPU throughput.

## Authentication Model

The first implementation uses existing OCI config/profile based authentication.
The managed public runner flow uses Instance Principal so newly created stock
VMs do not need OCI key material in user-data.

The control node uses `control_profile`. Runner VMs use one of these modes:

- `runner_auth_method = "user_principal"`: use `runner_profile` and existing
  `~opc/.oci/config` on the runner VM.
- `runner_auth_method = "instance_principal"`: use instance principal for both
  GenAI calls and Object Storage uploads.

Expected runner VM state for user principal mode:

```text
~opc/.oci/config
~opc/.oci/<private-key>
```

Required permissions for that OCI identity:

- Call OCI Generative AI in the target compartment.
- Write objects to the benchmark Object Storage bucket.
- Read objects from that bucket if the same profile is used for collection.

Do not put private key material in the repo or in the JSON config. Use an image
or bootstrap process where the profile is already present on the VM. Prefer
Instance Principal for newly created ephemeral runners.

Operational notes from the first Seoul smoke:

- IAM Dynamic Group and Policy create/update/delete calls must be sent to the
  tenancy home region. In this tenancy that is `us-ashburn-1`, so managed
  runner configs should set `network.iam_region`.
- OCI has a small Dynamic Group quota. If the quota is already full, reuse an
  existing group such as `ociexplain-dyn-group` and set
  `network.existing_dynamic_group_id`.
- If the existing Dynamic Group already matches the runner compartment, set
  `network.existing_dynamic_group_update` to `false`. Otherwise provide
  `network.existing_dynamic_group_base_matching_rule`; cleanup restores that
  base rule after the VM run.
- Parallel managed suites require `network.existing_dynamic_group_update=false`
  when reusing an existing Dynamic Group. The group should already include a
  rule that matches the runner compartment.
- If creating a temporary policy for an existing Dynamic Group, set
  `network.existing_dynamic_group_name` so the policy statement names the real
  group.
- `repo_ref` must point at a branch or tag that already contains the runner
  code. For `repo_ref=main`, push the automation before launching cloud-init
  smoke tests.

## Execution Flow

1. The control node creates a runner VM in the selected source region.
2. The VM starts with cloud-init user data.
3. cloud-init installs dependencies and prepares the benchmark repo.
4. The VM runs benchmark commands with a region-specific `--source-label`.
5. The VM uploads JSON and Markdown reports to Object Storage.
6. The VM uploads a report-specific completion marker.
7. The control node confirms reports exist in Object Storage.
8. The control node terminates the runner VM.
9. Reports are downloaded into `runs/`.
10. The dashboard/docs site is regenerated.

## Implemented Files

The initial automation implementation is in the repo:

- `genai_benchmark/ephemeral_runner.py`: Python control CLI implementation.
- `scripts/ephemeral_runner.py`: executable wrapper for the control CLI.
- `scripts/cloud-init/ephemeral-runner.sh.tmpl`: Oracle Linux cloud-init template.
- `configs/ephemeral-runners.example.json`: existing-subnet runner example.
- `configs/ephemeral-3region-managed.example.json`: managed public 3-region
  suite example.

The control CLI uses OCI CLI through `subprocess`. It supports `plan`,
`launch`, `collect`, `terminate`, `run`, `provision-network`, `run-managed`,
`run-managed-suite`, and `cleanup-managed` actions.

## Cloud-Init Bootstrap Outline

The runner VM user data should do the following:

```bash
#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

dnf install -y git python3.11 python3.11-pip python3.11-devel

git clone --branch "${REPO_REF}" "${REPO_URL}" /opt/genai-benchmark
cd /opt/genai-benchmark

python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

python benchmark.py \
  --source-label "${SOURCE_LABEL}" \
  --profile "${OCI_PROFILE}" \
  --region ap-osaka-1 \
  --region us-chicago-1 \
  --region eu-frankfurt-1 \
  --repeats 3 \
  --concurrency-levels 1,5,10 \
  --report-name "${SOURCE_LABEL}-global-r3"

oci os object put \
  --region "${OBJECT_STORAGE_REGION}" \
  --profile "${OCI_PROFILE}" \
  --bucket-name "${BUCKET_NAME}" \
  --name "runs/${SOURCE_LABEL}/${SOURCE_LABEL}-global-r3.json" \
  --file "runs/${SOURCE_LABEL}-global-r3.json" \
  --force

oci os object put \
  --region "${OBJECT_STORAGE_REGION}" \
  --profile "${OCI_PROFILE}" \
  --bucket-name "${BUCKET_NAME}" \
  --name "runs/${SOURCE_LABEL}/${SOURCE_LABEL}-global-r3.md" \
  --file "runs/${SOURCE_LABEL}-global-r3.md" \
  --force

printf "completed\n" > /tmp/benchmark-complete.txt
oci os object put \
  --region "${OBJECT_STORAGE_REGION}" \
  --profile "${OCI_PROFILE}" \
  --bucket-name "${BUCKET_NAME}" \
  --name "runs/${SOURCE_LABEL}/_${SOURCE_LABEL}-global-r3.complete.txt" \
  --file /tmp/benchmark-complete.txt \
  --force
```

The actual cloud-init template should inject `SOURCE_LABEL`, `BUCKET_NAME`,
`REPO_URL`, and `REPO_REF` from the control script.

## Control CLI Usage

The first implementation uses OCI CLI from a control node instead of Terraform.
This keeps the iteration loop short while the benchmark shape is still
changing.

Print the planned OCI commands without launching anything:

```bash
source .venv/bin/activate
python scripts/ephemeral_runner.py \
  --config configs/ephemeral-runners.example.json \
  --action plan
```

Launch a runner only:

```bash
python scripts/ephemeral_runner.py \
  --config <runner-config.json> \
  --runner ap-osaka-runner \
  --action launch
```

Run the full lifecycle:

```bash
python scripts/ephemeral_runner.py \
  --config <runner-config.json> \
  --runner ap-osaka-runner \
  --action run \
  --keep-on-failure
```

Collect already-uploaded reports:

```bash
python scripts/ephemeral_runner.py \
  --config <runner-config.json> \
  --runner ap-osaka-runner \
  --action collect \
  --output-dir runs
```

Terminate a known instance:

```bash
python scripts/ephemeral_runner.py \
  --config <runner-config.json> \
  --runner ap-osaka-runner \
  --action terminate \
  --instance-id <instance-ocid>
```

Create a managed public VCN/subnet/VM, use Instance Principal on the runner,
run the benchmark, collect reports, and clean up on success:

```bash
cp configs/ephemeral-seoul-managed.example.json configs/ephemeral-seoul-managed.local.json

python scripts/ephemeral_runner.py \
  --config configs/ephemeral-seoul-managed.local.json \
  --runner ap-seoul-runner \
  --action run-managed \
  --resource-state runs/ap-seoul-runner-state.json \
  --keep-on-failure
```

For the first real smoke, keep `--keep-on-failure` so VM/network/IAM resources
remain available if cloud-init or IAM propagation fails. The state file can be
used later for cleanup:

```bash
python scripts/ephemeral_runner.py \
  --config configs/ephemeral-seoul-managed.local.json \
  --runner ap-seoul-runner \
  --action cleanup-managed \
  --resource-state runs/ap-seoul-runner-state.json
```

Run a managed 3-region suite in parallel and publish reports:

```bash
cp configs/ephemeral-3region-managed.example.json configs/ephemeral-3region-managed.local.json

python scripts/ephemeral_runner.py \
  --config configs/ephemeral-3region-managed.local.json \
  --action run-managed-suite \
  --parallelism 3 \
  --resource-state runs/ephemeral-runner-state.json \
  --suite-name global-smoke-r1
```

The suite action writes runner state files as `runs/<source-label>-state.json`,
downloads each runner JSON/Markdown report, writes
`runs/<suite-name>-summary.md`, and regenerates `docs/dashboard.html`,
`docs/index.html`, and `docs/dashboard-preview.svg`. Use `--no-publish-site`
to skip docs generation.

The managed Seoul smoke example uses:

- VCN CIDR `10.91.0.0/16`
- Public subnet CIDR `10.91.1.0/24`
- SSH ingress disabled by default. If SSH inspection is needed, enable it only
  with a narrow `/32` source CIDR.
- Runner auth method `instance_principal`
- Benchmark smoke settings `repeats=1`, `concurrency_levels=1`

The generated VM launch command has this shape:

```bash
oci --profile "${CONTROL_PROFILE}" compute instance launch \
  --region <runner-region> \
  --compartment-id "${COMPARTMENT_ID}" \
  --availability-domain <availability-domain> \
  --display-name "genai-benchmark-${SOURCE_LABEL}" \
  --shape "${SHAPE}" \
  --image-id "${IMAGE_ID}" \
  --subnet-id "${SUBNET_ID}" \
  --metadata '{"user_data":"<base64-cloud-init>"}' \
  --freeform-tags '{"genai-benchmark":"ephemeral-runner","source_label":"<source-label>"}'
```

Check uploaded reports:

```bash
oci --profile "${CONTROL_PROFILE}" os object list \
  --region "${OBJECT_STORAGE_REGION}" \
  --bucket-name "${BUCKET_NAME}" \
  --prefix "runs/${SOURCE_LABEL}/"
```

Download reports to the local repo:

```bash
oci --profile "${CONTROL_PROFILE}" os object get \
  --region "${OBJECT_STORAGE_REGION}" \
  --bucket-name "${BUCKET_NAME}" \
  --name "runs/${SOURCE_LABEL}/${SOURCE_LABEL}-global-r3.json" \
  --file "runs/${SOURCE_LABEL}-global-r3.json"

oci --profile "${CONTROL_PROFILE}" os object get \
  --region "${OBJECT_STORAGE_REGION}" \
  --bucket-name "${BUCKET_NAME}" \
  --name "runs/${SOURCE_LABEL}/${SOURCE_LABEL}-global-r3.md" \
  --file "runs/${SOURCE_LABEL}-global-r3.md"
```

Terminate the runner VM:

```bash
oci --profile "${CONTROL_PROFILE}" compute instance terminate \
  --region <runner-region> \
  --instance-id <instance-ocid> \
  --force
```

## Report Naming

Use stable names so dashboards can compare source regions cleanly.

Recommended labels:

| Runner Region | Source Label | Report Name |
| --- | --- | --- |
| `ap-osaka-1` | `ap-osaka-runner` | `ap-osaka-runner-global-r3` |
| `us-chicago-1` | `us-chicago-runner` | `us-chicago-runner-global-r3` |
| `eu-frankfurt-1` | `eu-frankfurt-runner` | `eu-frankfurt-runner-global-r3` |

The benchmark command should always include the same target region list for
each runner so cross-region penalty is comparable.

## Operational Phases

### Phase 1: Single-Region Smoke

Run one ephemeral runner, preferably `ap-osaka-1`, against the three baseline
target regions.

Success criteria:

- VM launches successfully.
- Runner OCI profile can call GenAI.
- Runner OCI profile can upload Object Storage reports.
- JSON and Markdown reports appear under the expected prefix.
- The VM is terminated after the run.

### Phase 2: Three-Region Parallel Run

Run ephemeral runners in APAC, US, and EU.

Success criteria:

- All three source labels produce reports.
- Reports are downloaded into `runs/`.
- Dashboard can render all reports together.
- The same prompt/model/target-region matrix is present for each source label.

### Phase 3: Dashboard Upgrade

Promote `source_label` from report metadata into first-class dashboard analysis.

Recommended dashboard additions:

- Source label filter.
- Source label column in summary tables.
- Grouped view by `source_label` and target region.
- Same-region and cross-region latency penalty calculation.
- Scatter plot with failure markers.

## Failure Handling

The automation should treat these as separate failure classes:

- VM provisioning failure.
- cloud-init/bootstrap failure.
- dependency installation failure.
- OCI GenAI benchmark failure.
- Object Storage upload failure.
- cleanup failure.

Even if benchmark requests fail, the runner should still upload the report files
because failed model calls are valid benchmark observations.

If report upload fails, keep the VM alive long enough for manual inspection
during early smoke tests. After the automation is stable, terminate failed
runners by default and rely on console logs or instance logs.

## Follow-Up Implementation Items

- Add dashboard support for `source_label`.
- Add same-region versus cross-region penalty metrics.
- Add a scatter plot for latency, throughput, and failure markers.
- Add optional Object Storage bucket creation when no bucket name is supplied.
