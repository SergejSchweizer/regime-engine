#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '%s\n' "$1" >&2
  exit 64
}

assert_local_docker() {
  if [[ -n "${DOCKER_HOST:-}" && "${DOCKER_HOST}" != unix://* ]]; then
    fail "remote DOCKER_HOST is forbidden"
  fi
  local context endpoint
  context="$(docker context show)"
  endpoint="$(docker context inspect "$context" --format '{{(index .Endpoints "docker").Host}}')"
  [[ "$endpoint" == unix://* ]] || fail "Docker context must use a local Unix socket"
}

field() {
  local key="$1"
  awk -F= -v key="$key" '$1 == key {sub($1 "=", ""); print; found=1} END {if (!found) exit 1}' \
    "$MANIFEST"
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
assert_local_docker
[[ $# -eq 2 && "$2" == "--confirm-destructive-restore" ]] \
  || fail "usage: scripts/local_mlflow_restore.sh <backup-directory> --confirm-destructive-restore"
BACKUP_DIR="$(cd "$1" && pwd)"
MANIFEST="$BACKUP_DIR/manifest.env"
scripts/verify_mlflow_backup.sh "$BACKUP_DIR" >/dev/null

current_image_id="$(docker image inspect regime-engine-mlflow:local --format '{{.Id}}')"
[[ "$current_image_id" == "$(field application_image_id)" ]] \
  || fail "restore requires the exact locally built application image recorded by the backup"
[[ "$(git rev-parse HEAD)" == "$(field repository_git_sha)" ]] \
  || fail "restore requires the exact repository Git SHA recorded by the backup"

docker compose stop mlflow >/dev/null
restart_mlflow() {
  docker compose up -d --no-build mlflow >/dev/null 2>&1 || true
}
trap restart_mlflow EXIT

docker compose exec -T mlflow-postgres sh -eu -c \
  'dropdb --if-exists --force --username "$POSTGRES_USER" "$POSTGRES_DB"; createdb --username "$POSTGRES_USER" "$POSTGRES_DB"'
cat "$BACKUP_DIR/$(field database_dump)" | docker compose exec -T mlflow-postgres sh -eu -c \
  'pg_restore --exit-on-error --no-owner --no-privileges --username "$POSTGRES_USER" --dbname "$POSTGRES_DB"'

cat "$BACKUP_DIR/$(field artifact_archive)" | docker compose run --rm --no-deps \
  --entrypoint sh mlflow -eu -c \
  'find /mlflow/artifacts -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +; tar -C /mlflow/artifacts -xzf -'

metadata_tables="$(docker compose exec -T mlflow-postgres sh -eu -c \
  'psql -At --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = '\''public'\''"')"
[[ "$metadata_tables" =~ ^[0-9]+$ && "$metadata_tables" -gt 0 ]] \
  || fail "restored MLflow metadata schema is empty"
docker compose run --rm --no-deps --entrypoint sh mlflow -eu -c \
  'test -d /mlflow/artifacts && test -r /mlflow/artifacts'

trap - EXIT
docker compose up -d --no-build mlflow >/dev/null
scripts/verify_local_compose.sh >/dev/null
printf 'restore_verified=true\n'
printf 'metadata_table_count=%s\n' "$metadata_tables"
