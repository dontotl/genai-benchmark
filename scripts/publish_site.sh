#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
python -m genai_benchmark.dashboard --output runs/dashboard.html
python -m genai_benchmark.site --docs-dir docs --runs-dir runs
echo "Generated docs/dashboard.html, docs/index.html, docs/dashboard-preview.svg"
