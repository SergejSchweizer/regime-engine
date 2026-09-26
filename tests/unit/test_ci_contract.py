from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
WORKFLOW_NAMES = ("merge-gate.yml", "push-gate.yml")
CANONICAL_COVERAGE = 85


def _coverage_contract_inputs() -> tuple[dict[str, object], tuple[str, ...]]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    workflows = tuple(
        (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        for name in WORKFLOW_NAMES
    )
    return project, workflows


def _assert_coverage_contract(project: dict[str, object], workflows: tuple[str, ...]) -> None:
    assert project["tool"]["coverage"]["report"]["fail_under"] == CANONICAL_COVERAGE  # type: ignore[index]
    command_pattern = re.compile(r"coverage report --data-file=\.coverage\.unit --fail-under=(\d+)")
    commands = tuple(command_pattern.search(workflow) for workflow in workflows)
    assert all(commands)
    thresholds = tuple(int(match.group(1)) for match in commands if match is not None)
    assert thresholds == (CANONICAL_COVERAGE, CANONICAL_COVERAGE)
    assert len(set(workflows)) == 2
    for workflow in workflows:
        assert 'pytest -n auto tests -m "not integration and not external and not slow"' in workflow
        assert "actions/download-artifact" not in workflow
        assert "actions/upload-artifact" not in workflow
        assert "coverage-integration" not in workflow
        assert "test_global_regime_v4_full_compute.py" not in workflow


def test_quality_contract_and_gate_workflows_cannot_diverge() -> None:
    project, workflows = _coverage_contract_inputs()
    _assert_coverage_contract(project, workflows)
    for workflow in workflows:
        assert "\n  integration:\n" not in workflow
        assert 'pytest -n auto tests -m "integration and not slow and not external"' not in workflow
        assert "needs: [lint, type, unit]" in workflow
        assert "INTEGRATION:" not in workflow
        assert 'test "${INTEGRATION}" = success' not in workflow


@pytest.mark.parametrize("mutated_threshold", (84, 80))
def test_lower_threshold_mutations_fail_the_qa_contract(mutated_threshold: int) -> None:
    project, workflows = _coverage_contract_inputs()
    mutated_project = {
        **project,
        "tool": {
            **project["tool"],  # type: ignore[dict-item]
            "coverage": {
                **project["tool"]["coverage"],  # type: ignore[index]
                "report": {
                    **project["tool"]["coverage"]["report"],  # type: ignore[index]
                    "fail_under": mutated_threshold,
                },
            },
        },
    }
    with pytest.raises(AssertionError):
        _assert_coverage_contract(mutated_project, workflows)

    mutated_workflow = re.sub(
        r"(--fail-under=)85",
        rf"\g<1>{mutated_threshold}",
        workflows[0],
        count=1,
    )
    with pytest.raises(AssertionError):
        _assert_coverage_contract(project, (mutated_workflow, workflows[1]))


def test_merge_and_push_coverage_commands_are_semantically_identical() -> None:
    _project, workflows = _coverage_contract_inputs()
    report_lines = tuple(
        line.strip()
        for workflow in workflows
        for line in workflow.splitlines()
        if "coverage report --data-file=.coverage.unit" in line
    )
    assert len(report_lines) == 2
    assert report_lines[0] == report_lines[1]


def test_qa_contract_covers_only_the_two_authoritative_gate_workflows() -> None:
    assert WORKFLOW_NAMES == ("merge-gate.yml", "push-gate.yml")
    assert all((ROOT / ".github" / "workflows" / name).is_file() for name in WORKFLOW_NAMES)


def test_external_feature_postgres_smoke_script_is_executable() -> None:
    script = ROOT / "scripts" / "verify_feature_postgres.sh"
    assert script.stat().st_mode & 0o111


def test_local_pre_commit_hook_runs_only_hermetic_integration_tests() -> None:
    hook = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "id: hermetic-integration-tests" in hook
    assert (
        "entry: .venv/bin/pytest -n auto tests -m "
        '"integration and not slow and not external"' in hook
    )
    assert "pass_filenames: false" in hook
    assert "always_run: true" in hook


def test_cpu_bootstrap_caps_all_supported_native_thread_pools() -> None:
    bootstrap = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    export_script = (ROOT / "scripts" / "export_config_env.py").read_text(encoding="utf-8")
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert f'"{variable}"' in bootstrap
        assert f'"{variable}": str(native_threads)' in export_script
