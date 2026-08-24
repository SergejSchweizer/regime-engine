#!/usr/bin/env bash
set -euo pipefail

: "${REGIME_FEATURE_PGDATABASE:?REGIME_FEATURE_PGDATABASE is required}"
: "${REGIME_FEATURE_PGPASSWORD_FILE:?REGIME_FEATURE_PGPASSWORD_FILE is required}"
: "${REGIME_FEATURE_PGADMIN_DSN_FILE:?REGIME_FEATURE_PGADMIN_DSN_FILE is required}"

for file in "${REGIME_FEATURE_PGPASSWORD_FILE}" "${REGIME_FEATURE_PGADMIN_DSN_FILE}"; do
  [[ -r "${file}" ]] || { echo "required secret file is not readable" >&2; exit 2; }
done

role_password="$(cat "${REGIME_FEATURE_PGPASSWORD_FILE}")"
admin_dsn="$(cat "${REGIME_FEATURE_PGADMIN_DSN_FILE}")"
[[ -n "${role_password}" && -n "${admin_dsn}" ]] || { echo "secret files cannot be empty" >&2; exit 2; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
psql "${admin_dsn}" \
  --no-psqlrc \
  --set=ON_ERROR_STOP=1 \
  --set=target_db="${REGIME_FEATURE_PGDATABASE}" \
  --set=role_password="${role_password}" \
  --file="${script_dir}/regime_engine_reader.sql"

exec "${script_dir}/verify_reader.sh"
