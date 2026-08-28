#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="/tmp/regime-engine-xetra-v3-evaluation.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf '%s evaluation_skipped reason=already_running\n' "$(date --iso-8601=seconds)"
  exit 0
fi

cd "$ROOT"
eval "$(.venv/bin/python - config.yaml <<'PY'
from __future__ import annotations

import shlex
import sys
from pathlib import Path

import yaml

raw = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
settings = raw["feature_postgres"]
for source, target in {
    "host": "REGIME_FEATURE_PGHOST",
    "port": "REGIME_FEATURE_PGPORT",
    "database": "REGIME_FEATURE_PGDATABASE",
    "user": "REGIME_FEATURE_PGUSER",
    "sslmode": "REGIME_FEATURE_PGSSLMODE",
    "password_file": "REGIME_FEATURE_PGPASSWORD_FILE",
}.items():
    print(f"export {target}={shlex.quote(str(settings[source]))}")
PY
)"

curl --fail --silent http://127.0.0.1:5000/regime-engine/v1/health >/dev/null
export REGIME_ENGINE_ROOT="$ROOT"
export REGIME_ENGINE_GIT_SHA="$(git rev-parse HEAD)"
export REGIME_EVALUATION_CACHE_DIR=/volume2/docker/mlflow/artifacts/evaluation-cache
export REGIME_LOG_DIR=/volume2/docker/mlflow/.logs
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_ARTIFACT_ROOT=/volume2/docker/mlflow/artifacts
export MPLCONFIGDIR=/tmp/matplotlib-regime-engine
exec .venv/bin/python scripts/run_xetra_v3_evaluations.py
