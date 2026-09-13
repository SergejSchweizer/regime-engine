from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
VERIFIER = ROOT / "scripts" / "verify_zero_legacy.py"


def _run_verifier(
    root: Path,
    tmp_path: Path,
    *,
    pythonpath: Path | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    report_path = tmp_path / "zero-legacy-report.json"
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    if pythonpath is not None:
        environment["PYTHONPATH"] = str(pythonpath)
    else:
        environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            str(VERIFIER),
            "--root",
            str(root),
            "--json-out",
            str(report_path),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, json.loads(report_path.read_text(encoding="utf-8"))


def _configured_scan_paths() -> tuple[str, ...]:
    tree = ast.parse(VERIFIER.read_text(encoding="utf-8"), filename=str(VERIFIER))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_SCAN_PATHS" for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, tuple)
            assert all(isinstance(item, str) for item in value)
            return value
    raise AssertionError("zero-legacy verifier has no literal _SCAN_PATHS contract")


def _write_fixture_file(root: Path, relative: str, content: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_active_verifier_runs_without_production_service_imports(tmp_path: Path) -> None:
    guard = tmp_path / "import-guard"
    package = guard / "market_regime_engine"
    package.mkdir(parents=True)
    (guard / "mlflow.py").write_text(
        "raise AssertionError('the static verifier imported MLflow')\n", encoding="utf-8"
    )
    (guard / "psycopg.py").write_text(
        "raise AssertionError('the static verifier imported PostgreSQL')\n", encoding="utf-8"
    )
    (package / "__init__.py").write_text(
        "raise AssertionError('the static verifier imported production code')\n", encoding="utf-8"
    )

    fixture = tmp_path / "fixture"
    _write_fixture_file(fixture, "README.md", "static verifier fixture\n")
    completed, report = _run_verifier(fixture, tmp_path, pythonpath=guard)

    assert completed.returncode == 0, completed.stderr
    assert report["status"] == "verified"
    assert report["violations"] == []


def test_active_repository_is_verified_by_the_cli(tmp_path: Path) -> None:
    completed, report = _run_verifier(ROOT, tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert report["status"] == "verified"
    assert report["violations"] == []
    assert report["scanned_file_count"] >= 1
    assert report["python_module_count"] >= 1
    assert report["import_edge_count"] == len(report["import_graph"])


def test_scan_contract_names_every_active_repository_surface() -> None:
    configured = set(_configured_scan_paths())

    required = {
        "src",
        "configs",
        "tests",
        "scripts",
        "docs",
        ".github",
        "README.md",
        "pyproject.toml",
    }
    assert required <= configured

    # The fixture tree is part of the tests surface and must remain covered by
    # the recursive tests entry.  This also makes the package path explicit.
    assert (ROOT / "tests" / "fixtures").is_dir()
    assert any(path == "tests" or path.startswith("tests/") for path in configured)
    assert (ROOT / "src" / "market_regime_engine").is_dir()


@pytest.mark.parametrize(
    "relative",
    (
        "src/current.py",
        "configs/current.yaml",
        "tests/fixtures/current.py",
        "scripts/current.py",
        "docs/current.md",
        ".github/current.yml",
        "pyproject.toml",
    ),
)
def test_verifier_reaches_each_active_surface(tmp_path: Path, relative: str) -> None:
    token = "semantic" + "-" + "medoid"
    path = _write_fixture_file(tmp_path / "fixture", relative, f"{token}\n")

    completed, report = _run_verifier(tmp_path / "fixture", tmp_path)

    assert completed.returncode == 1
    assert report["status"] == "failed"
    assert any(
        violation["path"] == path.relative_to(tmp_path / "fixture").as_posix()
        and violation["kind"] == "content"
        for violation in report["violations"]
    )


def test_verifier_rejects_forbidden_import_in_temporary_source(tmp_path: Path) -> None:
    removed_module = "feature" + "_selection"
    fixture = tmp_path / "fixture"
    _write_fixture_file(
        fixture,
        "src/market_regime_engine/current.py",
        f"import market_regime_engine.{removed_module}\n",
    )

    completed, report = _run_verifier(fixture, tmp_path)

    assert completed.returncode == 1
    assert report["status"] == "failed"
    assert any(
        violation["kind"] == "import" and removed_module in violation["detail"]
        for violation in report["violations"]
    )


def test_verifier_rejects_forbidden_path_component_in_temporary_source(tmp_path: Path) -> None:
    removed_directory = "legacy" + "_package"
    fixture = tmp_path / "fixture"
    _write_fixture_file(
        fixture,
        f"src/market_regime_engine/{removed_directory}/current.py",
        "value = 1\n",
    )

    completed, report = _run_verifier(fixture, tmp_path)

    assert completed.returncode == 1
    assert report["status"] == "failed"
    assert any(
        violation["kind"] == "path" and removed_directory in violation["detail"]
        for violation in report["violations"]
    )
