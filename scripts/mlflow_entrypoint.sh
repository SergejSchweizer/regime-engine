#!/bin/sh
set -eu

require_value() {
  name="$1"
  eval "value=\${$name:-}"
  if [ -z "$value" ]; then
    printf '%s\n' "required environment variable is missing: $name" >&2
    exit 64
  fi
}

read_secret() {
  file="$1"
  name="$2"
  if [ -z "$file" ] || [ ! -r "$file" ]; then
    printf '%s\n' "required secret file is unavailable: $name" >&2
    exit 64
  fi
  value=$(cat "$file")
  if [ -z "$value" ]; then
    printf '%s\n' "required secret file is empty: $name" >&2
    exit 64
  fi
  printf '%s' "$value"
}

uri_escape() {
  URI_VALUE="$1" python -c 'import os, urllib.parse; print(urllib.parse.quote(os.environ["URI_VALUE"], safe=""), end="")'
}

require_value MLFLOW_BACKEND_DB_NAME
require_value MLFLOW_BACKEND_DB_USER
require_value MLFLOW_BACKEND_DB_PASSWORD_FILE

MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-/mlflow/artifacts}"
MLFLOW_RUNTIME_UID="${MLFLOW_RUNTIME_UID:-10001}"
MLFLOW_RUNTIME_GID="${MLFLOW_RUNTIME_GID:-10001}"
MLFLOW_WORKERS="${MLFLOW_WORKERS:-4}"
MLFLOW_THREADS_PER_WORKER="${MLFLOW_THREADS_PER_WORKER:-4}"
MLFLOW_HTTP_TIMEOUT_SECONDS="${MLFLOW_HTTP_TIMEOUT_SECONDS:-120}"
MLFLOW_GRACEFUL_TIMEOUT_SECONDS="${MLFLOW_GRACEFUL_TIMEOUT_SECONDS:-30}"

mkdir -p "$MLFLOW_ARTIFACT_ROOT"
chown -R "$MLFLOW_RUNTIME_UID:$MLFLOW_RUNTIME_GID" "$MLFLOW_ARTIFACT_ROOT"

backend_password=$(read_secret "$MLFLOW_BACKEND_DB_PASSWORD_FILE" MLFLOW_BACKEND_DB_PASSWORD_FILE)
backend_user=$(uri_escape "$MLFLOW_BACKEND_DB_USER")
backend_password_escaped=$(uri_escape "$backend_password")
backend_db=$(uri_escape "$MLFLOW_BACKEND_DB_NAME")
unset backend_password
backend_uri="postgresql+psycopg://${backend_user}:${backend_password_escaped}@mlflow-postgres:5432/${backend_db}"
unset backend_password_escaped

exec setpriv --reuid "$MLFLOW_RUNTIME_UID" --regid "$MLFLOW_RUNTIME_GID" --clear-groups \
  mlflow server \
  --app-name regime-engine \
  --backend-store-uri "$backend_uri" \
  --artifacts-destination "$MLFLOW_ARTIFACT_ROOT" \
  --host 0.0.0.0 \
  --port 5000 \
  --workers "$MLFLOW_WORKERS" \
  --gunicorn-opts "--worker-class gthread --threads ${MLFLOW_THREADS_PER_WORKER} --timeout ${MLFLOW_HTTP_TIMEOUT_SECONDS} --graceful-timeout ${MLFLOW_GRACEFUL_TIMEOUT_SECONDS}"
