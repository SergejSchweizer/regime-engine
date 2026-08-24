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

docker compose down
