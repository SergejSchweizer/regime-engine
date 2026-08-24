#!/usr/bin/env bash
set -euo pipefail

PROFILE="${REGIME_ENGINE_PROFILE:-xetra}"
COMPOSE_FILE="${REGIME_ENGINE_COMPOSE_FILE:-compose.yaml}"
LOCK_ROOT="${XDG_RUNTIME_DIR:-/tmp}"
LOCK_FILE="${REGIME_ENGINE_MODEL_CYCLE_LOCK:-${LOCK_ROOT}/regime-engine-model-cycle-${PROFILE}.lock}"

fail() {
  printf 'regime-engine model cycle: %s\n' "$*" >&2
  exit 2
}

if [[ "$PROFILE" != "xetra" ]]; then
  fail "only profile xetra is supported"
fi

command -v docker >/dev/null 2>&1 || fail "docker is required"
command -v flock >/dev/null 2>&1 || fail "flock is required for single-run locking"
[[ -f "$COMPOSE_FILE" ]] || fail "compose file not found: $COMPOSE_FILE"

case "${DOCKER_HOST:-}" in
  ""|unix://*) ;;
  *) fail "remote DOCKER_HOST is forbidden; use the local Unix-socket Docker daemon" ;;
esac

DOCKER_CONTEXT="$(docker context show)"
DOCKER_ENDPOINT="$(docker context inspect "$DOCKER_CONTEXT" --format '{{(index .Endpoints "docker").Host}}')"
case "$DOCKER_ENDPOINT" in
  unix://*) ;;
  *) fail "Docker context $DOCKER_CONTEXT is not local Unix-socket based: $DOCKER_ENDPOINT" ;;
esac

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf 'regime-engine model cycle: profile %s is already running; deterministic no-op\n' "$PROFILE" >&2
  exit 0
fi

compose() {
  docker compose -f "$COMPOSE_FILE" "$@"
}

run_cli() {
  compose exec -T mlflow regime-engine "$@"
}

json_field() {
  local document="$1"
  local field="$2"
  printf '%s' "$document" | compose exec -T mlflow python -c '
import json
import sys

field = sys.argv[1]
payload = json.load(sys.stdin)
value = payload.get("fields", {}).get(field)
if value is None:
    raise SystemExit(f"missing lifecycle field: {field}")
print(value)
' "$field"
}

json_optional_field() {
  local document="$1"
  local field="$2"
  printf '%s' "$document" | compose exec -T mlflow python -c '
import json
import sys

field = sys.argv[1]
payload = json.load(sys.stdin)
value = payload.get("fields", {}).get(field)
if value is not None:
    print(value)
' "$field"
}

STATUS_JSON="$(run_cli status --profile "$PROFILE")"
CURRENT_SOURCE_BUILD="$(json_field "$STATUS_JSON" current_source_build_id)"
COMPLETED_SOURCE_BUILD="$(json_optional_field "$STATUS_JSON" completed_source_build_id)"

if [[ -n "$COMPLETED_SOURCE_BUILD" && "$CURRENT_SOURCE_BUILD" == "$COMPLETED_SOURCE_BUILD" ]]; then
  printf 'regime-engine model cycle: source build %s already completed; no-op\n' \
    "$CURRENT_SOURCE_BUILD" >&2
  printf '%s\n' "$STATUS_JSON"
  exit 0
fi

EVALUATION_JSON="$(run_cli evaluate --profile "$PROFILE")"
EVALUATION_ID="$(json_field "$EVALUATION_JSON" evaluation_id)"
EVALUATION_SOURCE_BUILD="$(json_field "$EVALUATION_JSON" source_build_id)"
STATISTICAL_CHAMPION="$(json_field "$EVALUATION_JSON" statistical_champion_candidate_id)"

if [[ "$EVALUATION_SOURCE_BUILD" != "$CURRENT_SOURCE_BUILD" ]]; then
  fail "source build changed between status and evaluate"
fi

REFIT_JSON="$(run_cli final-refit --profile "$PROFILE" --evaluation-id "$EVALUATION_ID")"
PRODUCTION_PACKAGE="$(json_field "$REFIT_JSON" production_package)"

OOS_JSON="$(run_cli publish-oos --profile "$PROFILE" --evaluation-id "$EVALUATION_ID")"
OOS_BUILD_ID="$(json_field "$OOS_JSON" oos_build_id)"

REGISTER_JSON="$({
  run_cli register \
    --profile "$PROFILE" \
    --production-package "$PRODUCTION_PACKAGE" \
    --oos-build-id "$OOS_BUILD_ID"
})"
CHALLENGER_VERSION="$(json_field "$REGISTER_JSON" exact_version)"

printf 'regime-engine model cycle: source=%s evaluation=%s statistical_champion=%s oos=%s challenger=%s\n' \
  "$CURRENT_SOURCE_BUILD" \
  "$EVALUATION_ID" \
  "$STATISTICAL_CHAMPION" \
  "$OOS_BUILD_ID" \
  "$CHALLENGER_VERSION" >&2
printf '%s\n' "$REGISTER_JSON"
