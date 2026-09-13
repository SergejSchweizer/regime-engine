"""Validate deployment config.yaml and emit safe shell exports.

The generated exports are an internal compatibility bridge for the Python
runtime. Secrets are deliberately accepted only as a password-file path; a
password value in config.yaml is rejected.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required(mapping: dict[str, Any], key: str, name: str) -> Any:
    value = mapping.get(key)
    if value is None or value == "":
        raise ValueError(f"{name}.{key} is required")
    return value


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def load_config(path: Path) -> dict[str, str]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read config file {path}: {exc}") from exc
    root = _mapping(raw, "config")

    mlflow = _mapping(root.get("mlflow"), "mlflow")
    tracking_uri = str(_required(mlflow, "tracking_uri", "mlflow"))
    registry_uri = str(mlflow.get("registry_uri", tracking_uri))
    if tracking_uri != "http://10.10.1.3:5000" or registry_uri != tracking_uri:
        raise ValueError("MLflow tracking and registry must use http://10.10.1.3:5000")

    feature = _mapping(root.get("feature_postgres"), "feature_postgres")
    password_keys = {"password", "password_value", "dsn", "connection_string"}
    present_secrets = password_keys.intersection(feature)
    if present_secrets:
        raise ValueError(
            "feature_postgres must not contain a password, DSN, or connection string; "
            "use password_file"
        )
    feature_exports = {
        "REGIME_FEATURE_PGHOST": str(_required(feature, "host", "feature_postgres")),
        "REGIME_FEATURE_PGPORT": str(_required(feature, "port", "feature_postgres")),
        "REGIME_FEATURE_PGDATABASE": str(_required(feature, "database", "feature_postgres")),
        "REGIME_FEATURE_PGUSER": str(_required(feature, "user", "feature_postgres")),
        "REGIME_FEATURE_PGSSLMODE": str(_required(feature, "sslmode", "feature_postgres")),
        "REGIME_FEATURE_PGPASSWORD_FILE": str(
            _required(feature, "password_file", "feature_postgres")
        ),
    }
    if not Path(feature_exports["REGIME_FEATURE_PGPASSWORD_FILE"]).is_absolute():
        raise ValueError("feature_postgres.password_file must be absolute")

    evaluation = _mapping(root.get("evaluation"), "evaluation")
    checkpoint_root = str(_required(evaluation, "checkpoint_root", "evaluation"))
    if not Path(checkpoint_root).expanduser().is_absolute():
        raise ValueError("evaluation.checkpoint_root must be absolute")

    runtime = _mapping(root.get("runtime"), "runtime")
    cpu_workers = _positive_int(_required(runtime, "cpu_workers", "runtime"), "runtime.cpu_workers")
    native_threads = _positive_int(
        _required(runtime, "native_threads", "runtime"), "runtime.native_threads"
    )

    exports = {
        "MLFLOW_TRACKING_URI": tracking_uri,
        "MLFLOW_REGISTRY_URI": registry_uri,
        **feature_exports,
        "REGIME_EVALUATION_CHECKPOINT_ROOT": str(Path(checkpoint_root).expanduser()),
        "REGIME_CPU_WORKERS": str(cpu_workers),
        "OMP_NUM_THREADS": str(native_threads),
        "OPENBLAS_NUM_THREADS": str(native_threads),
        "MKL_NUM_THREADS": str(native_threads),
    }

    environment = root.get("environment", {})
    for key, value in _mapping(environment, "environment").items():
        if not isinstance(key, str) or not key.isidentifier() or not key.isupper():
            raise ValueError(f"environment key {key!r} must be an uppercase identifier")
        if "PASSWORD" in key or "SECRET" in key:
            raise ValueError(f"secret environment key {key!r} is not allowed in config.yaml")
        if value is None or isinstance(value, (dict, list)):
            raise ValueError(f"environment.{key} must be a scalar")
        exports[key] = str(value)
    return exports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    try:
        exports = load_config(arguments.config)
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    for key, value in exports.items():
        print(f"export {key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
