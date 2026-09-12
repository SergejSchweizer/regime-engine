#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${REGIME_ENGINE_CONFIG_FILE:-$ROOT/config.yaml}"
LOCK_FILE="${REGIME_ENGINE_V4_EVALUATION_LOCK:-${XDG_RUNTIME_DIR:-/tmp}/regime-engine-xetra-v4-evaluation.lock}"

fail() {
  printf 'regime-engine xetra v4 evaluation: %s\n' "$*" >&2
  exit 2
}

[[ -x "$ROOT/.venv/bin/python" ]] || fail "missing .venv/bin/python"
[[ -f "$ROOT/configs/profiles/xetra_v4.yaml" ]] || fail "missing v4 profile"
[[ -f "$CONFIG_FILE" ]] || fail "feature PostgreSQL config not found: $CONFIG_FILE"
command -v flock >/dev/null 2>&1 || fail "flock is required"
command -v curl >/dev/null 2>&1 || fail "curl is required"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export REGIME_ENGINE_ROOT="$ROOT"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://10.10.1.3:5000}"
# Reserve two logical CPUs for the operating system and service overhead;
# operators may lower this for a shared host.
export REGIME_CPU_WORKERS="${REGIME_CPU_WORKERS:-86}"
# Native BLAS/OpenMP threads are deliberately one per explicit process/task;
# otherwise each HMM worker creates a second machine-sized thread pool.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
eval "$($ROOT/.venv/bin/python - "$CONFIG_FILE" <<'PY'
from __future__ import annotations

import shlex
import sys
from pathlib import Path

import yaml

raw = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
feature_postgres = raw["feature_postgres"]
for key, variable in {
    "host": "REGIME_FEATURE_PGHOST",
    "port": "REGIME_FEATURE_PGPORT",
    "database": "REGIME_FEATURE_PGDATABASE",
    "user": "REGIME_FEATURE_PGUSER",
    "sslmode": "REGIME_FEATURE_PGSSLMODE",
    "password_file": "REGIME_FEATURE_PGPASSWORD_FILE",
}.items():
    print(f"export {variable}={shlex.quote(str(feature_postgres[key]))}")
PY
)"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf 'regime-engine xetra v4 evaluation: another run is active; no-op\n' >&2
  exit 0
fi

curl --fail --silent --show-error "$MLFLOW_TRACKING_URI/health" >/dev/null \
  || fail "external MLflow health check failed: $MLFLOW_TRACKING_URI"

exec "$ROOT/.venv/bin/python" "$ROOT/scripts/run_xetra_v4_full_evaluation.py"
