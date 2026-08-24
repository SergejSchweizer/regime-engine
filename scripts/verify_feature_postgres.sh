#!/usr/bin/env bash
set -euo pipefail
set +x

CONFIG_FILE="${REGIME_FEATURE_PG_CONFIG_FILE:-config.yaml}"
if [[ -f "$CONFIG_FILE" ]]; then
  eval "$(.venv/bin/python - "$CONFIG_FILE" <<'PY'
from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
raw = yaml.safe_load(path.read_text(encoding="utf-8"))
if not isinstance(raw, dict) or not isinstance(raw.get("feature_postgres"), dict):
    raise SystemExit("config.yaml requires a feature_postgres mapping")
settings = raw["feature_postgres"]
mapping = {
    "host": "REGIME_FEATURE_PGHOST",
    "port": "REGIME_FEATURE_PGPORT",
    "database": "REGIME_FEATURE_PGDATABASE",
    "user": "REGIME_FEATURE_PGUSER",
    "sslmode": "REGIME_FEATURE_PGSSLMODE",
    "password_file": "REGIME_FEATURE_PGPASSWORD_FILE",
}
for source_key, environment_key in mapping.items():
    value = settings.get(source_key)
    if value is not None and environment_key not in os.environ:
        print(f"export {environment_key}={shlex.quote(str(value))}")
PY
)"
fi

: "${REGIME_FEATURE_PGDATABASE:?REGIME_FEATURE_PGDATABASE is required}"

if [[ -n "${REGIME_FEATURE_PGPASSWORD:-}" && -n "${REGIME_FEATURE_PGPASSWORD_FILE:-}" ]]; then
  echo "configure exactly one feature PostgreSQL password source" >&2
  exit 2
fi
if [[ -z "${REGIME_FEATURE_PGPASSWORD:-}" && -z "${REGIME_FEATURE_PGPASSWORD_FILE:-}" ]]; then
  echo "REGIME_FEATURE_PGPASSWORD or REGIME_FEATURE_PGPASSWORD_FILE is required" >&2
  exit 2
fi

export REGIME_FEATURE_PGHOST="${REGIME_FEATURE_PGHOST:-10.10.1.3}"
export REGIME_FEATURE_PGPORT="${REGIME_FEATURE_PGPORT:-54321}"
export REGIME_FEATURE_PGUSER="${REGIME_FEATURE_PGUSER:-regime-engine}"
export REGIME_FEATURE_PGSSLMODE="${REGIME_FEATURE_PGSSLMODE:-require}"

[[ "$REGIME_FEATURE_PGHOST" == "10.10.1.3" ]] || {
  echo "external feature PostgreSQL host must be exactly 10.10.1.3" >&2
  exit 2
}
[[ "$REGIME_FEATURE_PGPORT" == "54321" ]] || {
  echo "external feature PostgreSQL port must be exactly 54321" >&2
  exit 2
}
[[ "$REGIME_FEATURE_PGUSER" == "regime-engine" ]] || {
  echo "external feature PostgreSQL user must be exactly regime-engine" >&2
  exit 2
}
[[ "$REGIME_FEATURE_PGSSLMODE" == "require" ]] || {
  echo "external feature PostgreSQL sslmode must be exactly require" >&2
  exit 2
}

export REGIME_RUN_EXTERNAL_FEATURE_PG=1
exec .venv/bin/pytest -m external tests/external/test_feature_postgres_smoke.py
