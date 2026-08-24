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
[[ $# -eq 1 ]] || fail "usage: scripts/verify_mlflow_backup.sh <backup-directory>"
BACKUP_DIR="$(cd "$1" && pwd)"
MANIFEST="$BACKUP_DIR/manifest.env"
[[ -f "$MANIFEST" ]] || fail "backup manifest is missing"

[[ "$(field manifest_version)" == "1" ]] || fail "unsupported backup manifest version"
[[ "$(field mlflow_version)" == "3.15.1" ]] || fail "backup MLflow version is not 3.15.1"
[[ "$(field application_image_id)" == sha256:* ]] || fail "application image ID is missing"
[[ "$(field repository_git_sha)" =~ ^[0-9a-f]{40}$ ]] || fail "repository Git SHA is invalid"
[[ "$(field database_dump)" == "mlflow-backend.dump" ]] \
  || fail "database dump filename is invalid"
[[ "$(field artifact_archive)" == "mlflow-artifacts.tar.gz" ]] \
  || fail "artifact archive filename is invalid"

db_file="$BACKUP_DIR/$(field database_dump)"
artifact_file="$BACKUP_DIR/$(field artifact_archive)"
[[ -s "$db_file" && -s "$artifact_file" ]] || fail "backup payload is missing or empty"
[[ "$(sha256sum "$db_file" | awk '{print $1}')" == "$(field database_dump_sha256)" ]] \
  || fail "database dump hash mismatch"
[[ "$(sha256sum "$artifact_file" | awk '{print $1}')" == "$(field artifact_archive_sha256)" ]] \
  || fail "artifact archive hash mismatch"

tar -tzf "$artifact_file" >/dev/null
cat "$db_file" | docker compose exec -T mlflow-postgres pg_restore --list >/dev/null

if grep -Eiq '(password|secret|credential|dsn)=' "$MANIFEST"; then
  fail "backup manifest contains forbidden credential material"
fi

printf 'backup_verified=true\n'
printf 'application_image_id=%s\n' "$(field application_image_id)"
printf 'repository_git_sha=%s\n' "$(field repository_git_sha)"
printf 'mlflow_version=%s\n' "$(field mlflow_version)"
printf 'postgres_version=%s\n' "$(field postgres_version)"
