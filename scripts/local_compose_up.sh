#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '%s\n' "$1" >&2
  exit 64
}

assert_local_docker() {
  if [[ -n "${DOCKER_HOST:-}" && "${DOCKER_HOST}" != unix://* ]]; then
    fail "remote DOCKER_HOST is forbidden; use the deployment host local Unix socket"
  fi
  local context endpoint
  context="$(docker context show)"
  endpoint="$(docker context inspect "$context" --format '{{(index .Endpoints "docker").Host}}')"
  [[ "$endpoint" == unix://* ]] || fail "Docker context must use a local Unix socket: $context"
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
assert_local_docker

docker image inspect regime-engine-mlflow:local >/dev/null 2>&1 \
  || fail "local image regime-engine-mlflow:local is missing; run scripts/local_compose_build.sh first"

source scripts/compose_provenance_env.sh

mkdir -p /volume2/docker/mlflow/artifacts /volume2/docker/mlflow/postgres

docker compose up -d --no-build
