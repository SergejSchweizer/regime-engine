#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dev_engine/regime-engine
exec 9>/tmp/regime-engine-xetra-v3-evaluation.lock
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

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))["feature_postgres"]
for key, variable in {
    "host": "REGIME_FEATURE_PGHOST", "port": "REGIME_FEATURE_PGPORT",
    "database": "REGIME_FEATURE_PGDATABASE", "user": "REGIME_FEATURE_PGUSER",
    "sslmode": "REGIME_FEATURE_PGSSLMODE", "password_file": "REGIME_FEATURE_PGPASSWORD_FILE",
}.items():
    print(f"export {variable}={shlex.quote(str(config[key]))}")
PY
)"

curl --fail --silent http://127.0.0.1:5000/regime-engine/v1/health >/dev/null
export REGIME_ENGINE_ROOT="$ROOT"
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_ARTIFACT_ROOT=/volume2/docker/mlflow/artifacts
export MPLCONFIGDIR=/tmp/matplotlib-regime-engine
if .venv/bin/python scripts/check_xetra_v3_evaluation_dedup.py; then
  printf '%s evaluation_skipped reason=unchanged\n' "$(date --iso-8601=seconds)"
  exit 0
else
  status=$?
  if [[ $status -ne 3 ]]; then
    exit "$status"
  fi
fi
exec .venv/bin/python scripts/run_xetra_v3_evaluations.py
