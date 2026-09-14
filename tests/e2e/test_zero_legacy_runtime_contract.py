from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
VERIFIER = ROOT / "scripts" / "verify_zero_legacy.py"


def _module() -> object:
    spec = importlib.util.spec_from_file_location("verify_zero_legacy_runtime", VERIFIER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load zero-legacy verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_runtime_audit_proves_v4_only_fail_closed_boundaries() -> None:
    module = _module()

    report = module._runtime_audit(ROOT)  # type: ignore[attr-defined]

    assert report["status"] == "verified"
    assert report["violations"] == []
    checks = report["checks"]
    assert isinstance(checks, dict)
    assert checks["cli_commands"] == [
        "evaluate",
        "final-refit",
        "publish-oos",
        "register",
        "status",
    ]
    assert checks["public_profiles"] == ["xetra"]
    assert checks["old_package_rejection"] == "verified"


class _ExternalMlflowFixture:
    def __init__(self) -> None:
        self.experiment = SimpleNamespace(experiment_id="evaluation")
        self.runs = [
            SimpleNamespace(
                info=SimpleNamespace(run_id="parent", tags={}),
                data=SimpleNamespace(
                    params={
                        "profile_id": "xetra",
                        "profile_config_version": "4",
                        "evaluation_id": "global_regime_v4",
                    },
                    tags={},
                ),
            ),
            SimpleNamespace(
                info=SimpleNamespace(
                    run_id="fold-001",
                    tags={"mlflow.parentRunId": "parent"},
                ),
                data=SimpleNamespace(
                    params={"evaluation_id": "global_regime_v4"},
                    tags={},
                ),
            ),
        ]
        self.models = [
            SimpleNamespace(
                model_id="logged-v4",
                tags={
                    "regime_engine.profile_id": "xetra",
                    "regime_engine.profile_config_version": "4",
                },
            )
        ]
        self.versions = [
            SimpleNamespace(
                version="7",
                tags={
                    "regime_engine.package_schema": "RegimeEngineProductionModel.v4",
                    "regime_engine.profile_id": "xetra",
                    "regime_engine.profile_config_version": "4",
                },
            )
        ]
        self.registered = SimpleNamespace(aliases={"champion": "7"})

    def get_experiment_by_name(self, name: str) -> object:
        assert name == "regime-engine-evaluation"
        return self.experiment

    def search_runs(self, **kwargs: object) -> list[object]:
        assert kwargs["experiment_ids"] == ["evaluation"]
        return self.runs

    def search_logged_models(self, **kwargs: object) -> list[object]:
        assert kwargs["experiment_ids"] == ["evaluation"]
        return self.models

    def search_model_versions(self, **kwargs: object) -> list[object]:
        assert kwargs["filter_string"] == "name='regime-xetra'"
        return self.versions

    def get_registered_model(self, name: str) -> object:
        assert name == "regime-xetra"
        return self.registered


def test_external_mlflow_audit_is_read_only_and_accepts_v4_fixture() -> None:
    module = _module()

    report = module.audit_external_mlflow(  # type: ignore[attr-defined]
        _ExternalMlflowFixture(), tracking_uri="http://mlflow.test"
    )

    assert report["status"] == "verified"
    assert report["violations"] == []
    assert report["evaluation"]["legacy_run_ids"] == []
    assert report["registry"]["legacy_version_ids"] == []
    assert report["registry"]["legacy_aliases"] == []


def test_external_mlflow_audit_rejects_legacy_runs_models_versions_and_aliases() -> None:
    module = _module()
    client = _ExternalMlflowFixture()
    client.runs.append(
        SimpleNamespace(
            info=SimpleNamespace(run_id="legacy-run", tags={}),
            data=SimpleNamespace(
                params={"profile_id": "xetra", "profile_config_version": "2"}, tags={}
            ),
        )
    )
    client.models.append(
        SimpleNamespace(
            model_id="logged-legacy",
            tags={"regime_engine.profile_id": "xetra", "regime_engine.profile_config_version": "2"},
        )
    )
    client.versions.append(
        SimpleNamespace(
            version="8",
            tags={"regime_engine.package_schema": "RegimeEngineProductionModel.v3"},
        )
    )
    client.registered.aliases["retired"] = "8"

    report = module.audit_external_mlflow(  # type: ignore[attr-defined]
        client, tracking_uri="http://mlflow.test"
    )

    assert report["status"] == "failed"
    assert report["evaluation"]["legacy_run_ids"] == ["legacy-run"]
    assert report["evaluation"]["legacy_logged_model_ids"] == ["logged-legacy"]
    assert report["registry"]["legacy_version_ids"] == ["8"]
    assert report["registry"]["legacy_aliases"] == ["retired->8"]
