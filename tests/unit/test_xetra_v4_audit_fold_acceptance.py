from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest


def _verifier() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "verify_xetra_v4_math.py"
    spec = importlib.util.spec_from_file_location("verify_xetra_v4_math_acceptance", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the standalone Xetra v4 verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _likelihood_item(model_family: str, observations: list[list[float]]) -> dict[str, object]:
    item: dict[str, object] = {
        "model_family": model_family,
        "observations": observations,
        "start_probabilities": [1.0],
        "transition_matrix": [[1.0]],
        "means": [[0.0]],
        "covariances": [[[1.0]]],
    }
    if model_family == "gmm_hmm":
        item.update(
            {
                "mixture_weights": [[0.25, 0.75]],
                "mixture_means": [[[0.0], [0.0]]],
                "mixture_covariances": [[[[1.0]], [[1.0]]]],
            }
        )
    elif model_family == "student_t_hmm":
        item["degrees_of_freedom"] = [5.0]
    return item


def _known_likelihood(model_family: str, observations: list[list[float]]) -> float:
    values = [row[0] for row in observations]
    if model_family in {"gaussian_hmm", "gmm_hmm"}:
        return sum(-0.5 * (math.log(2.0 * math.pi) + value * value) for value in values)
    degree = 5.0
    return sum(
        math.lgamma((degree + 1.0) / 2.0)
        - math.lgamma(degree / 2.0)
        - 0.5 * math.log(degree * math.pi)
        - 0.5 * (degree + 1.0) * math.log1p(value * value / degree)
        for value in values
    )


def _bundle(tmp_path: Path) -> tuple[Path, Path, dict[str, float]]:
    snapshot_path = tmp_path / "xetra-folds.arrow"
    table = pa.table(
        {
            "timestamp_m1": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "feature_a": [0.0, 1.0, 2.0],
            "feature_b": [2.0, 1.0, 0.0],
        }
    )
    with snapshot_path.open("wb") as handle, ipc.new_file(handle, table.schema) as writer:
        writer.write_table(table)

    observations = [[0.0], [1.0]]
    families = ("gaussian_hmm", "gmm_hmm", "student_t_hmm")
    expected_likelihoods = {family: _known_likelihood(family, observations) for family in families}
    dossiers = [
        {
            "outer_fold_index": fold_index,
            "timestamp_column": "timestamp_m1",
            "feature_order": ["feature_a", "feature_b"],
            "distance": [[0.0, 0.0], [0.0, 0.0]],
            "likelihoods": [
                {
                    **_likelihood_item(family, observations),
                    "fold_index": fold_index,
                    "scope": "TRAIN",
                    "log_likelihood": expected_likelihoods[family],
                }
            ],
        }
        for fold_index, family in zip((1, 2, 3), families, strict=True)
    ]
    expectations_path = tmp_path / "xetra-folds.json"
    expectations_path.write_text(
        json.dumps(
            {
                "audit_outer_fold_indices": [1, 2, 3],
                "fold_audits": dossiers,
            }
        ),
        encoding="utf-8",
    )
    return snapshot_path, expectations_path, expected_likelihoods


def test_public_likelihood_callable_checks_all_supported_families() -> None:
    module = _verifier()
    observations = [[0.0], [1.0]]

    for family in ("gaussian_hmm", "gmm_hmm", "student_t_hmm"):
        actual = module.independent_hmm_log_likelihood(_likelihood_item(family, observations))
        assert actual == pytest.approx(_known_likelihood(family, observations), abs=1.0e-12)


def test_public_verifier_accepts_first_middle_last_fold_bundle(tmp_path: Path) -> None:
    module = _verifier()
    _snapshot_path, expectations_path, _expected = _bundle(tmp_path)
    columns = {
        "timestamp_m1": np.asarray(["2026-01-01", "2026-01-02", "2026-01-03"], dtype=object),
        "feature_a": np.asarray([0.0, 1.0, 2.0]),
        "feature_b": np.asarray([2.0, 1.0, 0.0]),
    }
    expectations = json.loads(expectations_path.read_text(encoding="utf-8"))

    report = module.verify_expectations(columns, expectations, max_workers=2)

    assert report["status"] == "verified"
    assert report["audited_outer_fold_indices"] == [1, 2, 3]
    assert report["audited_outer_fold_count"] == 3
    assert report["gaussian_likelihood_max_abs_error"] == pytest.approx(0.0, abs=1.0e-12)


def test_cli_verifier_accepts_the_same_local_arrow_json_bundle(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    snapshot_path, expectations_path, _expected = _bundle(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "verify_xetra_v4_math.py"),
            "--snapshot",
            str(snapshot_path),
            "--expectations",
            str(expectations_path),
            "--workers",
            "2",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "verified"
    assert report["audited_outer_fold_indices"] == [1, 2, 3]
    assert report["audited_outer_fold_count"] == 3


def test_verifier_fails_closed_when_snapshot_identity_changes(tmp_path: Path) -> None:
    module = _verifier()
    snapshot_path, expectations_path, _expected = _bundle(tmp_path)
    expectations = json.loads(expectations_path.read_text(encoding="utf-8"))
    snapshot_digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    expectations["source_identity"] = {
        "source_build_id": "build-1",
        "source_data_sha256": "a" * 64,
        "source_catalog_hash": "b" * 64,
        "dataset_snapshot_key": "c" * 64,
        "snapshot_sha256": snapshot_digest,
    }
    columns = module._read_snapshot(snapshot_path)
    report = module.verify_expectations(
        columns,
        expectations,
        max_workers=1,
        snapshot_sha256=snapshot_digest,
    )
    assert report["status"] == "verified"
    with pytest.raises(SystemExit, match="snapshot hash"):
        module.verify_expectations(
            columns,
            expectations,
            max_workers=1,
            snapshot_sha256="e" * 64,
        )
