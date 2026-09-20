from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
WORKFLOW_NAMES = ("merge-gate.yml", "push-gate.yml")
INTEGRATION_SELECTOR = 'pytest -n auto tests -m "integration and not slow and not external"'


def _workflows() -> tuple[str, ...]:
    return tuple(
        (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        for name in WORKFLOW_NAMES
    )


def _assert_integration_gate_contract(workflow: str) -> None:
    assert "integration:" in workflow
    assert INTEGRATION_SELECTOR in workflow
    assert "needs: [lint, type, unit, integration]" in workflow
    assert "INTEGRATION: ${{ needs.integration.result }}" in workflow
    assert 'test "${INTEGRATION}" = success' in workflow


def _integration_section(workflow: str) -> str:
    match = re.search(
        r"\n  integration:\n(?P<section>.*?)(?=\n  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
        re.DOTALL,
    )
    assert match is not None
    return match.group("section")


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
        "needs: [lint, type, unit, integration]", "needs: [lint, type, unit]", 1
    )
    with pytest.raises(AssertionError):
        _assert_integration_gate_contract(workflows[workflow_index])


def test_bare_integration_selector_fails_qa() -> None:
    for workflow in _workflows():
        mutated = workflow.replace(INTEGRATION_SELECTOR, "pytest -n auto tests -m integration", 1)
        with pytest.raises(AssertionError):
            _assert_integration_gate_contract(mutated)


def test_integration_lane_excludes_slow_and_external_tests() -> None:
    for workflow in _workflows():
        integration_section = _integration_section(workflow)
        assert INTEGRATION_SELECTOR in integration_section
        assert "-m integration" not in integration_section.replace(
            '"integration and not slow and not external"', ""
        )


def test_integration_gate_has_no_external_service_configuration() -> None:
    forbidden = ("10.10.1.3", "MLFLOW_TRACKING_URI", "PGPASSWORD", "password_file")
    for workflow in _workflows():
        integration_section = _integration_section(workflow)
        assert not any(value in integration_section for value in forbidden)
