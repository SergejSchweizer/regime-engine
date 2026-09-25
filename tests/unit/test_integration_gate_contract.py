from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
WORKFLOW_NAMES = ("merge-gate.yml", "push-gate.yml")


def _workflows() -> tuple[str, ...]:
    return tuple(
        (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        for name in WORKFLOW_NAMES
    )


def _assert_integration_gate_contract(workflow: str) -> None:
    assert "\n  integration:\n" not in workflow
    assert 'pytest -n auto tests -m "integration and not slow and not external"' not in workflow
    assert "pytest -n auto tests -m integration" not in workflow
    assert "needs: [lint, type, unit]" in workflow
    assert "INTEGRATION:" not in workflow
    assert 'test "${INTEGRATION}" = success' not in workflow


def test_both_authoritative_workflows_require_hermetic_integration() -> None:
    workflows = _workflows()
    assert len(workflows) == 2
    assert all((ROOT / ".github" / "workflows" / name).is_file() for name in WORKFLOW_NAMES)
    for workflow in workflows:
        _assert_integration_gate_contract(workflow)


@pytest.mark.parametrize("workflow_index", (0, 1))
def test_removing_integration_from_either_terminal_gate_fails_qa(workflow_index: int) -> None:
    workflows = list(_workflows())
    workflows[workflow_index] = workflows[workflow_index].replace(
        "needs: [lint, type, unit]", "needs: [lint, type, unit, integration]", 1
    )
    with pytest.raises(AssertionError):
        _assert_integration_gate_contract(workflows[workflow_index])


def test_bare_integration_selector_fails_qa() -> None:
    for workflow in _workflows():
        mutated = workflow.replace(
            'pytest -n auto tests -m "not integration and not external and not slow"',
            "pytest -n auto tests -m integration",
            1,
        )
        with pytest.raises(AssertionError):
            _assert_integration_gate_contract(mutated)


def test_integration_lane_excludes_slow_and_external_tests() -> None:
    for workflow in _workflows():
        assert "\n  integration:\n" not in workflow


def test_integration_gate_has_no_external_service_configuration() -> None:
    forbidden = ("10.10.1.3", "MLFLOW_TRACKING_URI", "PGPASSWORD", "password_file")
    for workflow in _workflows():
        assert not any(value in workflow for value in forbidden)
