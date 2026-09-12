#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${PR231_PROOF_OUTPUT_DIR:-$ROOT/.artifacts/pr231-hermetic-proof}"
mkdir -p "$OUTPUT_DIR"

[[ -x "$ROOT/.venv/bin/python" ]] || {
  printf 'PR-231 proof: missing %s/.venv/bin/python\n' "$ROOT" >&2
  exit 2
}

# The host exposes 88 logical CPUs; reserve two for the OS and use at most 86
# evaluation processes. Callers may lower this explicitly for shared hosts.
export REGIME_CPU_WORKERS="${REGIME_CPU_WORKERS:-86}"

COMPUTATION_PROOF="$OUTPUT_DIR/pr231-computation-proof.json"
RUN_METADATA="$OUTPUT_DIR/pr231-proof.json"
JUNIT_XML="$OUTPUT_DIR/pr231-junit.xml"
START_EPOCH="$(date +%s)"
STARTED_UTC="$(date -u --iso-8601=seconds)"
set +e
PR231_PROOF_OUTPUT="$COMPUTATION_PROOF" "$ROOT/.venv/bin/pytest" -q -n 1 \
  "$ROOT/tests/e2e/test_global_regime_v4_full_compute.py" \
  -m "integration and slow" --junitxml="$JUNIT_XML"
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
    "repository_sha": repository_sha,
    "command": ".venv/bin/pytest -q -n 1 tests/e2e/test_global_regime_v4_full_compute.py -m 'integration and slow' --junitxml=pr231-junit.xml",
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
