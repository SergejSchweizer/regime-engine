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
source scripts/compose_provenance_env.sh

services="$(docker compose config --services | LC_ALL=C sort)"
[[ "$services" == $'mlflow\nmlflow-postgres' ]] || fail "Compose service set is not exactly mlflow + mlflow-postgres"

mlflow_container="$(docker compose ps -q mlflow)"
backend_container="$(docker compose ps -q mlflow-postgres)"
[[ -n "$mlflow_container" && -n "$backend_container" ]] || fail "both Compose services must be running"

project="$(docker inspect "$mlflow_container" --format '{{index .Config.Labels "com.docker.compose.project"}}')"
[[ "$project" == "regime-engine" ]] || fail "running Compose project is not regime-engine"

configured_image="$(docker inspect "$mlflow_container" --format '{{.Config.Image}}')"
[[ "$configured_image" == "regime-engine-mlflow:local" ]] || fail "mlflow is not running the canonical local image tag"

image_id="$(docker inspect "$mlflow_container" --format '{{.Image}}')"
repo_digests="$(docker image inspect "$image_id" --format '{{join .RepoDigests ","}}')"
[[ -z "$repo_digests" ]] || fail "custom application image unexpectedly has registry RepoDigests"

git_sha="$(git rev-parse HEAD)"
image_git_sha="$(docker image inspect "$image_id" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
[[ "$image_git_sha" == "$git_sha" ]] || fail "running local image Git SHA does not match the deployment checkout"

mlflow_label="$(docker image inspect "$image_id" --format '{{index .Config.Labels "io.regime-engine.mlflow-version"}}')"
[[ "$mlflow_label" == "3.15.1" ]] || fail "running local image has the wrong MLflow version label"
mlflow_runtime="$(docker exec "$mlflow_container" python -c 'import mlflow; print(mlflow.__version__)')"
[[ "$mlflow_runtime" == "3.15.1" ]] || fail "running container has the wrong MLflow runtime version"

port_mapping="$(docker port "$mlflow_container" 5000/tcp)"
grep -Eq '(:|\])5000$' <<<"$port_mapping" || fail "mlflow host port 5000 is not published"
backend_port_mapping="$(docker port "$backend_container" 5432/tcp 2>/dev/null || true)"
[[ -z "$backend_port_mapping" ]] || fail "mlflow-postgres must not publish host port 5432"

build_timestamp="$(docker image inspect "$image_id" --format '{{index .Config.Labels "org.opencontainers.image.created"}}')"

printf 'compose_project=%s\n' "$project"
printf 'services=%s\n' "$(tr '\n' ',' <<<"$services" | sed 's/,$//')"
printf 'application_image_id=%s\n' "$image_id"
printf 'repository_git_sha=%s\n' "$image_git_sha"
printf 'mlflow_version=%s\n' "$mlflow_runtime"
printf 'build_timestamp=%s\n' "$build_timestamp"
printf 'port_5000=%s\n' "$(tr '\n' ',' <<<"$port_mapping" | sed 's/,$//')"
printf 'custom_image_repo_digests=none\n'
