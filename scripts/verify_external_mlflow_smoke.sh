#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

scripts/verify_local_compose.sh

: "${REGIME_EXTERNAL_MLFLOW_URI:=http://127.0.0.1:5000}"
export REGIME_EXTERNAL_MLFLOW_URI
export REGIME_RUN_EXTERNAL_MLFLOW=1

.venv/bin/pytest -m external tests/external/test_local_mlflow_smoke.py
