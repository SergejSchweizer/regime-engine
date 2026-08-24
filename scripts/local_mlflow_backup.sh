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

[[ $# -eq 1 ]] || fail "usage: scripts/local_mlflow_backup.sh <new-backup-directory>"
BACKUP_DIR="$1"
[[ ! -e "$BACKUP_DIR" ]] || fail "backup destination already exists"
mkdir -m 0700 -p "$BACKUP_DIR"
BACKUP_DIR="$(cd "$BACKUP_DIR" && pwd)"

scripts/verify_local_compose.sh >/dev/null
mlflow_container="$(docker compose ps -q mlflow)"
image_id="$(docker inspect "$mlflow_container" --format '{{.Image}}')"
git_sha="$(git rev-parse HEAD)"
mlflow_version="$(docker compose exec -T mlflow python -c 'import mlflow; print(mlflow.__version__)')"
postgres_version="$(docker compose exec -T mlflow-postgres postgres --version | tr ' ' '_')"
created_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

docker compose stop mlflow >/dev/null
restart_mlflow() {
  docker compose up -d --no-build mlflow >/dev/null 2>&1 || true
}
trap restart_mlflow EXIT

docker compose exec -T mlflow-postgres sh -eu -c \
  'pg_dump --format=custom --no-owner --no-privileges --username "$POSTGRES_USER" --dbname "$POSTGRES_DB"' \
  >"$BACKUP_DIR/mlflow-backend.dump"

docker compose run --rm --no-deps --entrypoint sh mlflow -eu -c \
  'tar -C /mlflow/artifacts -czf - .' >"$BACKUP_DIR/mlflow-artifacts.tar.gz"

db_sha="$(sha256sum "$BACKUP_DIR/mlflow-backend.dump" | awk '{print $1}')"
artifact_sha="$(sha256sum "$BACKUP_DIR/mlflow-artifacts.tar.gz" | awk '{print $1}')"

cat >"$BACKUP_DIR/manifest.env" <<EOF
manifest_version=1
created_utc=$created_utc
repository_git_sha=$git_sha
application_image_id=$image_id
mlflow_version=$mlflow_version
postgres_version=$postgres_version
database_dump=mlflow-backend.dump
database_dump_sha256=$db_sha
artifact_archive=mlflow-artifacts.tar.gz
artifact_archive_sha256=$artifact_sha
EOF
chmod 0600 "$BACKUP_DIR/manifest.env" "$BACKUP_DIR/mlflow-backend.dump" \
  "$BACKUP_DIR/mlflow-artifacts.tar.gz"

scripts/verify_mlflow_backup.sh "$BACKUP_DIR"
trap - EXIT
docker compose up -d --no-build mlflow >/dev/null
scripts/verify_local_compose.sh >/dev/null
printf 'verified_backup=%s\n' "$BACKUP_DIR"
