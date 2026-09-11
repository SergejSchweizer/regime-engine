#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${REGIME_EXTERNAL_MLFLOW_URI:=http://10.10.1.3:5000}"
export REGIME_EXTERNAL_MLFLOW_URI
export REGIME_RUN_EXTERNAL_MLFLOW=1

.venv/bin/python -m pytest -m external tests/external/test_local_mlflow_smoke.py
