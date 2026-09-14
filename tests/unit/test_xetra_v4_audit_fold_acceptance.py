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


def _strict_current_contract(
    module: Any,
    snapshot_path: Path,
    *,
    valid_indices: list[int] | None = None,
) -> dict[str, object]:
    indices = [1, 2, 3] if valid_indices is None else valid_indices
    final_ids = [
        "gaussian_hmm_k2_full",
        "gaussian_hmm_k3_full",
        "gaussian_hmm_k4_full",
        "gaussian_hmm_k5_full",
        "gmm_hmm_k2_m2_full",
        "gmm_hmm_k3_m2_full",
        "gmm_hmm_k4_m2_full",
        "gmm_hmm_k5_m2_full",
        "student_t_hmm_k2_full",
        "student_t_hmm_k3_full",
        "student_t_hmm_k4_full",
        "student_t_hmm_k5_full",
    ]
    discovery = {
        "feature_universe_mode": "all_non_timestamp_m1_double_precision",
        "excluded_source_column": "timestamp_m1",
        "feature_ordering": "postgresql_ordinal_position",
        "cluster_count_min": 2,
        "cluster_count_max": 12,
        "provisional_state_counts": [2, 3, 4, 5],
        "minimum_prefix_length": 2,
        "maximum_prefix_length": 8,
        "prefix_state_counts": [2, 3, 4, 5],
        "inner_train_source_observations": 756,
        "inner_test_source_observations": 63,
        "inner_step_source_observations": 63,
        "inner_partial_final_test": False,
        "outer_train_source_observations": 1260,
        "outer_test_source_observations": 63,
        "outer_step_source_observations": 63,
        "outer_partial_final_test": False,
        "minimum_outer_valid_fold_rate": 0.8,
        "minimum_outer_valid_folds": 3,
    }
    walk = {
        "minimum_train_source_observations": 1260,
        "test_source_observations": 63,
        "step_source_observations": 63,
        "allow_partial_final_test": False,
        "minimum_model_train_observations": 504,
        "minimum_model_test_observations": 42,
    }
    per_fold = [
        {
            "outer_fold_index": index,
            "eligible_feature_count": 3,
            "ranked_feature_count": 2,
            "cluster_count_candidates": [2],
            "prefix_length_candidates": [2],
            "provisional_candidate_ids": [f"gaussian_hmm_k{k}_full" for k in (2, 3, 4, 5)],
            "final_candidate_ids": final_ids,
        }
        for index in indices
    ]
    identity = {
        "source_build_id": "source-build-1",
        "source_data_sha256": "a" * 64,
        "source_catalog_hash": "b" * 64,
        "materialized_feature_data_sha256": "c" * 64,
        "dataset_snapshot_key": "d" * 64,
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "profile_hash": "e" * 64,
        "outer_plan_hash": "f" * 64,
        "repository_commit_sha": "1" * 40,
        "uv_lock_sha256": "2" * 64,
    }
    request = {
        "mode": "schema_discovery",
        "feature_names": [],
        "start": None,
        "end": None,
        "all_source_rows": True,
        "dynamic_catalog": True,
    }
    return {
        "schema_version": 1,
        "source_request": request,
        "audit_source_request": request,
        "source_bounds": {
            "source_dataset": "xetra_gold",
            "source_table": "regime_features_daily",
            "source_row_count": 3,
            "source_min_timestamp": "2026-01-01",
            "source_max_timestamp": "2026-01-03",
            "feature_count": 2,
            "feature_names_sha256": module._json_sha256(["feature_a", "feature_b"]),
            "materialized_row_count": 3,
            "materialized_min_timestamp": "2026-01-01",
            "materialized_max_timestamp": "2026-01-03",
            "skipped_incomplete_row_count": 0,
        },
        "search_bounds": {
            "declared_bounds": {
                "feature_discovery": discovery,
                "walk_forward": walk,
                "candidate_families": {
                    "gaussian_state_counts": [2, 3, 4, 5],
                    "gaussian_multistart_seeds": [11, 23, 37, 53, 71, 89, 107, 131],
                    "gmm_state_mixture_pairs": [[2, 2], [3, 2], [4, 2], [5, 2]],
                    "student_t_state_counts": [2, 3, 4, 5],
                    "final_candidate_ids": final_ids,
                },
            },
            "per_fold": per_fold,
        },
        "identity_hashes": identity,
        "valid_outer_fold_indices": indices,
        "audit_outer_fold_indices": [indices[0], indices[len(indices) // 2], indices[-1]],
        "resource_evidence": {
            "performance_report_path": "/tmp/current-xetra-performance.json",
            "available_logical_cpus": 4,
            "native_thread_environment": {"OMP_NUM_THREADS": "1"},
        },
    }


def test_strict_current_contract_requires_bounds_identity_folds_and_resources(
    tmp_path: Path,
) -> None:
    module = _verifier()
    snapshot_path, expectations_path, _expected = _bundle(tmp_path)
    expectations = json.loads(expectations_path.read_text(encoding="utf-8"))
    contract = _strict_current_contract(module, snapshot_path)
    expectations.update(
        {
            "schema_version": 3,
            "audit_contract": contract,
            "valid_outer_fold_indices": [1, 2, 3],
            "audit_outer_fold_indices": [1, 2, 3],
        }
    )
    columns = module._read_snapshot(snapshot_path)
    report = module.verify_expectations(
        columns,
        expectations,
        max_workers=2,
        snapshot_sha256=contract["identity_hashes"]["snapshot_sha256"],
        require_current_audit_contract=True,
    )
    assert report["status"] == "verified"
    assert report["audit_contract_verified"] is True

    broken = json.loads(json.dumps(expectations))
    del broken["audit_contract"]["source_request"]
    with pytest.raises(SystemExit, match="source_request"):
        module.verify_expectations(
            columns,
            broken,
            max_workers=1,
            snapshot_sha256=contract["identity_hashes"]["snapshot_sha256"],
            require_current_audit_contract=True,
        )


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
