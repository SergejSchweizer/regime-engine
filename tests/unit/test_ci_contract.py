from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_quality_contract_and_gate_workflows_cannot_diverge() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["tool"]["coverage"]["report"]["fail_under"] == 90
    for name in ("merge-gate.yml", "push-gate.yml"):
        workflow = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert 'pytest tests -m "not integration and not external"' in workflow
        assert "integration:" not in workflow
        assert "needs: [lint, type, unit, integration]" not in workflow
        assert "coverage-integration" not in workflow
        assert "coverage-unit" in workflow
        assert "coverage report --data-file=.coverage.unit --fail-under=80" in workflow
        assert "fail-under=90" not in workflow
        assert "test_global_regime_v4_full_compute.py" not in workflow


def test_long_hermetic_proof_is_local_only() -> None:
    assert not (ROOT / ".github" / "workflows" / "hermetic-v4-proof.yml").exists()
    runner = (ROOT / "scripts" / "run_pr231_hermetic_proof.sh").read_text(encoding="utf-8")
    assert '"integration and slow"' in runner
    assert "PR231_PROOF_OUTPUT" in runner
    assert 'RUN_METADATA="$OUTPUT_DIR/pr231-${PHASE}.json"' in runner
    assert "pr231-computation-proof.json" in runner


def test_local_pre_commit_hook_runs_only_hermetic_integration_tests() -> None:
    hook = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "id: hermetic-integration-tests" in hook
    assert (
        'entry: .venv/bin/pytest -n 1 tests -m "integration and not slow and not external"' in hook
    )
    assert "pass_filenames: false" in hook
    assert "always_run: true" in hook
