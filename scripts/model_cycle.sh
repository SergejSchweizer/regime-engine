#!/usr/bin/env bash
set -euo pipefail

PROFILE="${REGIME_ENGINE_PROFILE:-xetra}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${REGIME_ENGINE_CONFIG_FILE:-$ROOT/config.yaml}"
LOCK_ROOT="${XDG_RUNTIME_DIR:-/tmp}"
LOCK_FILE="${REGIME_ENGINE_MODEL_CYCLE_LOCK:-${LOCK_ROOT}/regime-engine-model-cycle-${PROFILE}.lock}"

fail() {
  printf 'regime-engine model cycle: %s\n' "$*" >&2
  exit 2
}

if [[ "$PROFILE" != "xetra" ]]; then
  fail "only profile xetra is supported"
fi

command -v flock >/dev/null 2>&1 || fail "flock is required for single-run locking"
[[ -x "$ROOT/.venv/bin/regime-engine" ]] || fail "missing .venv/bin/regime-engine"
[[ -f "$CONFIG_FILE" ]] || fail "feature PostgreSQL config not found: $CONFIG_FILE"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export REGIME_ENGINE_ROOT="$ROOT"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://10.10.1.3:5000}"
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

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
feature_postgres = config["feature_postgres"]
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
  printf 'regime-engine model cycle: profile %s is already running; deterministic no-op\n' "$PROFILE" >&2
  exit 0
fi

run_cli() {
  "$ROOT/.venv/bin/regime-engine" "$@"
}

json_field() {
  local document="$1"
  local field="$2"
  printf '%s' "$document" | "$ROOT/.venv/bin/python" -c '
import json
import sys

field = sys.argv[1]
payload = json.load(sys.stdin)
value = payload.get("fields", {}).get(field)
if value is None:
    raise SystemExit(f"missing lifecycle field: {field}")
print(value)
' "$field"
}

json_optional_field() {
  local document="$1"
  local field="$2"
  printf '%s' "$document" | "$ROOT/.venv/bin/python" -c '
import json
import sys

field = sys.argv[1]
payload = json.load(sys.stdin)
value = payload.get("fields", {}).get(field)
if value is not None:
    print(value)
' "$field"
}

STATUS_JSON="$(run_cli status --profile "$PROFILE")"
CURRENT_SOURCE_BUILD="$(json_field "$STATUS_JSON" current_source_build_id)"
COMPLETED_SOURCE_BUILD="$(json_optional_field "$STATUS_JSON" completed_source_build_id)"

if [[ -n "$COMPLETED_SOURCE_BUILD" && "$CURRENT_SOURCE_BUILD" == "$COMPLETED_SOURCE_BUILD" ]]; then
  printf 'regime-engine model cycle: source build %s already completed; no-op\n' \
    "$CURRENT_SOURCE_BUILD" >&2
  printf '%s\n' "$STATUS_JSON"
  exit 0
fi

EVALUATION_JSON="$(run_cli evaluate --profile "$PROFILE")"
EVALUATION_ID="$(json_field "$EVALUATION_JSON" evaluation_id)"
EVALUATION_SOURCE_BUILD="$(json_field "$EVALUATION_JSON" source_build_id)"
STATISTICAL_CHAMPION="$(json_field "$EVALUATION_JSON" statistical_champion_candidate_id)"

if [[ "$EVALUATION_SOURCE_BUILD" != "$CURRENT_SOURCE_BUILD" ]]; then
  fail "source build changed between status and evaluate"
fi

REFIT_JSON="$(run_cli final-refit --profile "$PROFILE" --evaluation-id "$EVALUATION_ID")"
PRODUCTION_PACKAGE="$(json_field "$REFIT_JSON" production_package)"

OOS_JSON="$(run_cli publish-oos --profile "$PROFILE" --evaluation-id "$EVALUATION_ID")"
OOS_BUILD_ID="$(json_field "$OOS_JSON" oos_build_id)"

REGISTER_JSON="$(run_cli register \
  --profile "$PROFILE" \
  --production-package "$PRODUCTION_PACKAGE" \
  --oos-build-id "$OOS_BUILD_ID")"
CHALLENGER_VERSION="$(json_field "$REGISTER_JSON" exact_version)"

printf 'regime-engine model cycle: source=%s evaluation=%s statistical_champion=%s oos=%s challenger=%s\n' \
  "$CURRENT_SOURCE_BUILD" \
  "$EVALUATION_ID" \
  "$STATISTICAL_CHAMPION" \
  "$OOS_BUILD_ID" \
  "$CHALLENGER_VERSION" >&2
printf '%s\n' "$REGISTER_JSON"
