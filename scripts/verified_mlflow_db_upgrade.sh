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

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
assert_local_docker
source scripts/compose_provenance_env.sh
[[ $# -eq 1 ]] || fail "usage: scripts/verified_mlflow_db_upgrade.sh <verified-backup-directory>"
BACKUP_DIR="$1"

scripts/verify_mlflow_backup.sh "$BACKUP_DIR" >/dev/null
scripts/verify_local_compose.sh >/dev/null

docker compose stop mlflow >/dev/null
restart_mlflow() {
  docker compose up -d --no-build mlflow >/dev/null 2>&1 || true
}
trap restart_mlflow EXIT

docker compose run --rm --no-deps \
  --entrypoint /usr/local/bin/regime-engine-mlflow-db-upgrade mlflow

trap - EXIT
docker compose up -d --no-build mlflow >/dev/null
scripts/verify_local_compose.sh >/dev/null
printf 'mlflow_db_upgrade_verified=true\n'
