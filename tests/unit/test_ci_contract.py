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


def test_long_hermetic_proof_has_a_dedicated_manual_and_scheduled_workflow() -> None:
    workflow = (ROOT / ".github" / "workflows" / "hermetic-v4-proof.yml").read_text(
        encoding="utf-8"
    )
    assert "workflow_dispatch:" in workflow
    assert 'cron: "17 3 * * 0"' in workflow
    assert "integration and slow" in workflow
    assert "pr231-proof.json" in workflow
