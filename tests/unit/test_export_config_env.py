from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT = Path("scripts/export_config_env.py")


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "mlflow": {
                    "tracking_uri": "http://10.10.1.3:5000",
                    "registry_uri": "http://10.10.1.3:5000",
                },
                "feature_postgres": {
                    "host": "10.10.1.3",
                    "port": 54321,
                    "database": "postgres",
                    "user": "macro-loader",
                    "sslmode": "disable",
                    "password_file": "/run/secrets/pg_macro_loader_password",
                },
                "evaluation": {"state_root": "/var/lib/regime-engine/state"},
                "runtime": {"cpu_workers": 86, "native_threads": 1},
                "environment": {"REGIME_REPLAY_MAX_ROWS": 10000},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_exports_all_runtime_metadata_without_secret_values(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(_config(tmp_path))],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "export MLFLOW_TRACKING_URI=http://10.10.1.3:5000" in result.stdout
    assert "export REGIME_FEATURE_PGDATABASE=postgres" in result.stdout
    assert "export REGIME_ENGINE_STATE_ROOT=/var/lib/regime-engine/state" in result.stdout
    assert "export REGIME_CPU_WORKERS=86" in result.stdout
    assert "export NUMEXPR_NUM_THREADS=1" in result.stdout
    assert "REGIME_REPLAY_MAX_ROWS=10000" in result.stdout
    assert "must-not-be-written" not in result.stdout


def test_rejects_inline_password(tmp_path: Path) -> None:
    path = _config(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["feature_postgres"]["password"] = "must-not-be-written"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "must-not-be-written" not in result.stderr
    assert "unknown keys for feature_postgres" in result.stderr


def test_rejects_removed_evaluation_alias(tmp_path: Path) -> None:
    path = _config(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["evaluation"]["checkpoint_root"] = "/var/lib/regime-engine/old"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unknown keys for evaluation" in result.stderr
