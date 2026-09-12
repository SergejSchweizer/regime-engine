#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${PR231_PROOF_OUTPUT_DIR:-$ROOT/.artifacts/pr231-hermetic-proof}"
PHASE="${1:-all}"
mkdir -p "$OUTPUT_DIR"

[[ -x "$ROOT/.venv/bin/python" ]] || {
  printf 'PR-231 proof: missing %s/.venv/bin/python\n' "$ROOT" >&2
  exit 2
}

# The host exposes 88 logical CPUs; reserve two for the OS and use at most 86
# evaluation processes. Callers may lower this explicitly for shared hosts.
export REGIME_CPU_WORKERS="${REGIME_CPU_WORKERS:-86}"

case "$PHASE" in
  all)
    TEST_SELECTOR="$ROOT/tests/e2e/test_global_regime_v4_full_compute.py"
    COMPUTATION_PROOF="$OUTPUT_DIR/pr231-computation-proof.json"
    ;;
  pipeline-math)
    TEST_SELECTOR="$ROOT/tests/e2e/test_global_regime_v4_subproofs.py::test_global_v4_subproof_pipeline_math"
    COMPUTATION_PROOF="$OUTPUT_DIR/pr231-pipeline-math-proof.json"
    ;;
  tracking-and-plots)
    TEST_SELECTOR="$ROOT/tests/e2e/test_global_regime_v4_subproofs.py::test_global_v4_subproof_tracking_and_plots"
    COMPUTATION_PROOF="$OUTPUT_DIR/pr231-tracking-and-plots-proof.json"
    ;;
  independent-process-and-labels)
    TEST_SELECTOR="$ROOT/tests/e2e/test_global_regime_v4_subproofs.py::test_global_v4_subproof_independent_process_and_labels"
    COMPUTATION_PROOF="$OUTPUT_DIR/pr231-independent-process-and-labels-proof.json"
    ;;
  future-mutation-isolation)
    TEST_SELECTOR="$ROOT/tests/e2e/test_global_regime_v4_subproofs.py::test_global_v4_subproof_future_mutation_isolation"
    COMPUTATION_PROOF="$OUTPUT_DIR/pr231-future-mutation-isolation-proof.json"
    ;;
  *)
    printf 'PR-231 proof: unknown phase %s\n' "$PHASE" >&2
    printf 'valid phases: all, pipeline-math, tracking-and-plots, independent-process-and-labels, future-mutation-isolation\n' >&2
    exit 2
    ;;
esac

RUN_METADATA="$OUTPUT_DIR/pr231-${PHASE}.json"
JUNIT_XML="$OUTPUT_DIR/pr231-junit.xml"
START_EPOCH="$(date +%s)"
STARTED_UTC="$(date -u --iso-8601=seconds)"
set +e
if [[ "$PHASE" == "all" ]]; then
  PR231_PROOF_OUTPUT="$COMPUTATION_PROOF" "$ROOT/.venv/bin/pytest" -q -n 1 \
    "$TEST_SELECTOR" -m "integration and slow" --junitxml="$JUNIT_XML"
else
  PR231_SUBPROOF_OUTPUT="$COMPUTATION_PROOF" "$ROOT/.venv/bin/pytest" -q -n 1 \
    "$TEST_SELECTOR" -m "integration and slow" --junitxml="$JUNIT_XML"
fi
EXIT_CODE=$?
set -e
END_EPOCH="$(date +%s)"
FINISHED_UTC="$(date -u --iso-8601=seconds)"

PR231_START_EPOCH="$START_EPOCH" \
PR231_STARTED_UTC="$STARTED_UTC" \
PR231_END_EPOCH="$END_EPOCH" \
PR231_FINISHED_UTC="$FINISHED_UTC" \
PR231_EXIT_CODE="$EXIT_CODE" \
PR231_ROOT="$ROOT" \
PR231_RUN_METADATA="$RUN_METADATA" \
PR231_COMPUTATION_PROOF="$COMPUTATION_PROOF" \
PR231_PHASE="$PHASE" \
PR231_COMMAND=".venv/bin/pytest -q -n 1 ${TEST_SELECTOR#$ROOT/} -m 'integration and slow' --junitxml=pr231-junit.xml" \
  "$ROOT/.venv/bin/python" - <<'PY'
from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path

root = Path(os.environ["PR231_ROOT"])
source = root / "tests/e2e/test_global_regime_v4_full_compute.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
golden_hash = next(
    node.value.value
    for node in tree.body
    if isinstance(node, ast.Assign)
    and any(
        isinstance(target, ast.Name) and target.id == "PR231_GOLDEN_SNAPSHOT_HASH"
        for target in node.targets
    )
    and isinstance(node.value, ast.Constant)
)
repository_sha = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=root,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
payload = {
    "workflow": "local",
    "phase": os.environ["PR231_PHASE"],
    "repository_sha": repository_sha,
    "command": os.environ["PR231_COMMAND"],
    "started_utc": os.environ["PR231_STARTED_UTC"],
    "finished_utc": os.environ["PR231_FINISHED_UTC"],
    "duration_seconds": int(os.environ["PR231_END_EPOCH"])
    - int(os.environ["PR231_START_EPOCH"]),
    "exit_code": int(os.environ["PR231_EXIT_CODE"]),
    "golden_snapshot_hash": golden_hash,
    "computation_proof_path": str(Path(os.environ["PR231_COMPUTATION_PROOF"]).resolve()),
}
Path(os.environ["PR231_RUN_METADATA"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

printf 'PR-231 proof metadata: %s\n' "$RUN_METADATA"
printf 'PR-231 computation proof: %s\n' "$COMPUTATION_PROOF"
exit "$EXIT_CODE"
