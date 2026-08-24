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

backend_password=$(read_secret "$MLFLOW_BACKEND_DB_PASSWORD_FILE" MLFLOW_BACKEND_DB_PASSWORD_FILE)
backend_user=$(uri_escape "$MLFLOW_BACKEND_DB_USER")
backend_password_escaped=$(uri_escape "$backend_password")
backend_db=$(uri_escape "$MLFLOW_BACKEND_DB_NAME")
unset backend_password
backend_uri="postgresql+psycopg://${backend_user}:${backend_password_escaped}@mlflow-postgres:5432/${backend_db}"
unset backend_password_escaped

exec mlflow db upgrade "$backend_uri"
