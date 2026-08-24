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

  if docker buildx version >/dev/null 2>&1; then
    local builder_endpoint
    while IFS= read -r builder_endpoint; do
      [[ "$builder_endpoint" == "default" || "$builder_endpoint" == unix://* ]] \
        || fail "remote Buildx endpoint is forbidden: $builder_endpoint"
    done < <(docker buildx inspect | awk -F: '/^[[:space:]]*Endpoint:/ {sub(/^[[:space:]]*/, "", $2); print $2}')
  fi
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
assert_local_docker

source scripts/compose_provenance_env.sh

docker compose build --pull mlflow
