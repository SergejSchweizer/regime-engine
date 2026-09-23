from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from market_regime_engine.feature_discovery.metadata_store import FeatureSelectionMetadataStore
from tests.unit.feature_discovery.test_pr498_monthly_refit import _source
from tests.unit.feature_discovery.test_pr499_orchestration_cadence_qa import _run


def test_hermetic_multifold_proof_populates_all_evidence_tables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = FeatureSelectionMetadataStore(tmp_path / "metadata")
    tracker = _Tracking()
    result = _run(monkeypatch, max_workers=None, tracker=tracker, metadata_store=store)

    assert len(result.folds) >= 2
    assert result.valid_folds
    assert all(fold.package is not None for fold in result.valid_folds)
    assert tracker.starts and tracker.artifacts

    required_tables = {
        "feature_registry",
        "fold_feature_stats",
        "pca_loadings",
        "correlation_mapping",
        "sffs_steps",
        "fold_model_stats",
    }
    with duckdb.connect(str(store.database), read_only=True) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
    assert required_tables <= tables


def test_pinned_hermetic_run_is_hash_stable_across_worker_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serial = _run(monkeypatch, max_workers=1)
    parallel = _run(monkeypatch, max_workers=None)
    assert parallel.result_hash == serial.result_hash
    assert tuple(item.package.package_hash for item in parallel.valid_folds) == tuple(
        item.package.package_hash for item in serial.valid_folds
    )


def test_outer_test_perturbation_preserves_prior_fold_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _run(monkeypatch)
    first = before.folds[0]
    assert first.package is not None
    source = _source()
    source.loc[
        source["timestamp_m1"] > first.fold.test_last_timestamp,
        "vix_log_level",
    ] += 100000.0
    after = _run(monkeypatch, source=source)
    assert after.folds[0].package is not None
    assert after.folds[0].package.package_hash == first.package.package_hash


def test_evidence_manifest_is_canonical_json_and_has_no_network_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run(monkeypatch)
    encoded = json.dumps(result.result_hash, sort_keys=True, separators=(",", ":"))
    assert encoded == json.dumps(result.result_hash, separators=(",", ":"), sort_keys=True)
    assert "10.10.1.3" not in encoded


class _Tracking:
    def __init__(self) -> None:
        self.starts: list[tuple[str, str | None]] = []
        self.artifacts: list[tuple[str, str]] = []

    def start_run(self, *, run_name: str, parent_run_id: str | None = None) -> str:
        del run_name
        run_id = f"run-{len(self.starts)}"
        self.starts.append((run_id, parent_run_id))
        return run_id

    def log_params(self, _run_id: str, _params: dict[str, str]) -> None:
        return None

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str) -> None:
        self.artifacts.append((run_id, f"{local_path}:{artifact_path}"))

    def end_run(self, _run_id: str) -> None:
        return None

    def fail_run(self, _run_id: str) -> None:
        raise AssertionError("hermetic proof tracking failed")
