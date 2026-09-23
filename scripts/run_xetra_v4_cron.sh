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

export REGIME_ENGINE_ROOT="$ROOT"
eval "$($ROOT/.venv/bin/python "$ROOT/scripts/export_config_env.py" "$CONFIG_FILE")"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf 'regime-engine xetra v4 evaluation: another run is active; no-op\n' >&2
  exit 0
fi

curl --fail --silent --show-error "$MLFLOW_TRACKING_URI/health" >/dev/null \
  || fail "external MLflow health check failed: $MLFLOW_TRACKING_URI"

exec "$ROOT/.venv/bin/python" -m market_regime_engine.cli evaluate --profile xetra
