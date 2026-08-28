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
docker inspect --format '{{.State.Running}}' mlflow | grep -qx true
docker cp scripts/run_xetra_v3_evaluations.py mlflow:/tmp/run_xetra_v3_evaluations.py
exec docker exec --user 10001:10001 \
  -e REGIME_ENGINE_ROOT=/opt/regime-engine \
  -e MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
  -e MLFLOW_ARTIFACT_ROOT=/mlflow/artifacts \
  -e MPLCONFIGDIR=/tmp/matplotlib-regime-engine \
  mlflow python /tmp/run_xetra_v3_evaluations.py
