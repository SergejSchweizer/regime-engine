from __future__ import annotations

import pickle
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd  # type: ignore[import-untyped]
import pytest

import market_regime_engine.evaluations.global_regime_v4 as global_v4
from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluations.run_store import DatasetSnapshotKey, EvaluationRunState
from market_regime_engine.features.ports import (
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
    FeatureRequest,
    FeatureRow,
    FeatureSnapshot,
)
from market_regime_engine.profiles.loader import load_profile

START = datetime(2020, 1, 1, tzinfo=UTC)


def _lineage() -> SourceLineage:
    return SourceLineage(
        source_dataset="regime_loader",
        source_build_id="schema-build-1",
        data_sha256="a" * 64,
        schema_version=2,
        feature_version=1,
        source_table="regime_loader",
        synced_at_utc=START,
        row_count=2,
        min_timestamp=START,
        max_timestamp=START + timedelta(days=1),
    )


def test_v4_source_entrypoint_requests_the_complete_catalog(monkeypatch) -> None:
    lineage = _lineage()
    catalog = FeatureCatalogSnapshot.from_entries(
        lineage,
        "timestamp_m1",
        (FeatureCatalogEntry("feature_a", 1), FeatureCatalogEntry("feature_b", 2)),
    )
    snapshot = FeatureSnapshot(
        lineage,
        ("feature_a", "feature_b"),
        (
            FeatureRow(START, (1.0, None)),
            FeatureRow(START + timedelta(days=1), (2.0, 3.0)),
        ),
    )
    bound_catalog = catalog.with_materialization(snapshot)

    class Source:
        def __init__(self) -> None:
            self.request: FeatureRequest | None = None

        def read_schema_wide_with_catalog(self, request: FeatureRequest):
            self.request = request
            return bound_catalog, snapshot

    source = Source()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    captured: dict[str, object] = {}

    def fake_evaluate(rows: pd.DataFrame, **kwargs):
        captured["rows"] = rows
        captured.update(kwargs)
        return "evaluated"

    monkeypatch.setattr(global_v4, "evaluate_global_regime_v4", fake_evaluate)
    result = global_v4.evaluate_global_regime_v4_from_source(source, profile=profile)

    assert result == "evaluated"
    assert source.request is not None
    assert source.request.feature_names == ()
    assert source.request.mode.value == "feature_selection"
    rows = captured["rows"]
    assert isinstance(rows, pd.DataFrame)
    assert tuple(rows.columns) == ("timestamp_m1", "feature_a", "feature_b")
    assert tuple(rows["feature_a"]) == (1.0, 2.0)
    assert pd.isna(rows["feature_b"].iloc[0])
    assert rows["feature_b"].iloc[1] == 3.0
    assert captured["catalog"] == bound_catalog


def test_v4_source_entrypoint_fails_closed_for_invalid_snapshot_contracts() -> None:
    lineage = _lineage()
    catalog = FeatureCatalogSnapshot.from_entries(
        lineage,
        "timestamp_m1",
        (FeatureCatalogEntry("feature_a", 1), FeatureCatalogEntry("feature_b", 2)),
    )
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    invalid_profile = load_profile("configs/profiles/xetra_v3.yaml")

    class Source:
        def __init__(self, result) -> None:
            self.result = result

        def read_schema_wide_with_catalog(self, request: FeatureRequest):
            del request
            return self.result

    valid_snapshot = FeatureSnapshot(
        lineage,
        catalog.feature_names,
        (FeatureRow(START, (1.0, 2.0)),),
    )
    bound_catalog = catalog.with_materialization(valid_snapshot)

    with pytest.raises(ValueError, match="canonical Xetra v4 profile"):
        global_v4.evaluate_global_regime_v4_from_source(
            Source((bound_catalog, valid_snapshot)), profile=invalid_profile
        )

    mismatched_snapshot = FeatureSnapshot(
        lineage,
        ("feature_a",),
        (FeatureRow(START, (1.0,)),),
    )
    with pytest.raises(ValueError, match="columns do not match"):
        global_v4.evaluate_global_regime_v4_from_source(
            Source((catalog, mismatched_snapshot)), profile=profile
        )

    missing_digest = type(
        "Snapshot",
        (),
        {
            "feature_names": catalog.feature_names,
            "materialized_feature_data_sha256": None,
            "rows": (object(),),
        },
    )()
    with pytest.raises(ValueError, match="missing its materialization digest"):
        global_v4.evaluate_global_regime_v4_from_source(
            Source((catalog, missing_digest)), profile=profile
        )

    mismatched_digest = type(
        "Snapshot",
        (),
        {
            "feature_names": catalog.feature_names,
            "materialized_feature_data_sha256": "b" * 64,
            "rows": (object(),),
        },
    )()
    with pytest.raises(ValueError, match="digests differ"):
        global_v4.evaluate_global_regime_v4_from_source(
            Source((catalog, mismatched_digest)), profile=profile
        )

    empty_snapshot = FeatureSnapshot(lineage, catalog.feature_names, ())
    empty_catalog = catalog.with_materialization(empty_snapshot)
    with pytest.raises(ValueError, match="contains no rows"):
        global_v4.evaluate_global_regime_v4_from_source(
            Source((empty_catalog, empty_snapshot)), profile=profile
        )


def test_v4_source_entrypoint_finalizes_snapshot_and_run_identity(monkeypatch) -> None:
    lineage = _lineage()
    catalog = FeatureCatalogSnapshot.from_entries(
        lineage,
        "timestamp_m1",
        (FeatureCatalogEntry("feature_a", 1), FeatureCatalogEntry("feature_b", 2)),
    )
    snapshot = FeatureSnapshot(
        lineage,
        catalog.feature_names,
        (
            FeatureRow(START, (1.0, 2.0)),
            FeatureRow(START + timedelta(days=1), (2.0, 3.0)),
        ),
    )
    bound_catalog = catalog.with_materialization(snapshot)
    profile = load_profile("configs/profiles/xetra_v4.yaml")

    class Source:
        def read_schema_wide_with_catalog(self, request: FeatureRequest):
            assert request.feature_names == ()
            return bound_catalog, snapshot

    class SnapshotStore:
        def __init__(self) -> None:
            self.finalized: tuple[DatasetSnapshotKey, FeatureSnapshot] | None = None

        def finalize(self, key: DatasetSnapshotKey, value: FeatureSnapshot) -> None:
            self.finalized = key, value

        def load(self, key: DatasetSnapshotKey) -> FeatureSnapshot:
            assert self.finalized is not None
            assert self.finalized[0] == key
            return self.finalized[1]

    class RunStore:
        def __init__(self) -> None:
            self.opened = None
            self.completed = None

        def open_run(self, key):
            self.opened = key
            return EvaluationRunState(key.key, key.dataset_snapshot_key, "RUNNING", None, 0)

        def load_completed_work_unit(self, key, work_unit_key, input_hash):
            del key, work_unit_key, input_hash
            return None

        def claim_work_unit(self, key, work_unit_key, input_hash):
            del key, work_unit_key, input_hash
            return True

        def complete_work_unit(self, key, work_unit_key, input_hash, payload):
            del key, work_unit_key, input_hash, payload

        def complete_run(self, key, root_identity_hash, payload):
            self.completed = key, root_identity_hash, payload

        def load_completed_run(self, key):
            del key
            return None

    snapshot_store = SnapshotStore()
    run_store = RunStore()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        global_v4,
        "plan_walk_forward",
        lambda timestamps, walk_forward: SimpleNamespace(
            plan_hash="c" * 64,
            evaluation_cutoff=timestamps[-1],
        ),
    )

    class Result:
        result_hash = "f" * 64

    def fake_evaluate(rows: pd.DataFrame, **kwargs):
        captured["rows"] = rows
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(global_v4, "evaluate_global_regime_v4", fake_evaluate)
    result = global_v4.evaluate_global_regime_v4_from_source(
        Source(),
        profile=profile,
        snapshot_store=snapshot_store,
        run_store=run_store,
        repository_commit_sha="d" * 40,
        uv_lock_sha256="e" * 64,
        python_version="3.14.7",
    )

    assert isinstance(result, Result)
    assert snapshot_store.finalized is not None
    assert run_store.opened is not None
    expected_dataset_key = DatasetSnapshotKey.from_catalog(bound_catalog)
    assert run_store.opened.dataset_snapshot_key == expected_dataset_key.key
    assert captured["run_store"] is run_store
    assert captured["run_key"] == run_store.opened


def test_v4_source_entrypoint_returns_a_completed_durable_run(monkeypatch) -> None:
    lineage = _lineage()
    catalog = FeatureCatalogSnapshot.from_entries(
        lineage,
        "timestamp_m1",
        (FeatureCatalogEntry("feature_a", 1), FeatureCatalogEntry("feature_b", 2)),
    )
    snapshot = FeatureSnapshot(
        lineage,
        catalog.feature_names,
        (FeatureRow(START, (1.0, 2.0)), FeatureRow(START + timedelta(days=1), (2.0, 3.0))),
    )
    bound_catalog = catalog.with_materialization(snapshot)
    profile = load_profile("configs/profiles/xetra_v4.yaml")

    class Source:
        def read_schema_wide_with_catalog(self, request: FeatureRequest):
            del request
            return bound_catalog, snapshot

    class SnapshotStore:
        def finalize(self, key, value):
            del key, value

        def load(self, key):
            del key
            return snapshot

    train_end = START + timedelta(days=1259)
    test_start = train_end + timedelta(days=1)
    test_end = test_start + timedelta(days=62)
    fold = global_v4.WalkForwardFold(
        1,
        "fold_001",
        START,
        train_end,
        test_start,
        test_end,
        1260,
        63,
    )
    configuration = global_v4._fallback_configuration(
        catalog, lineage.source_build_id, fold, "cached"
    )
    cached_result = global_v4.AdaptiveEvaluationResult(
        source_build_id=lineage.source_build_id,
        catalog_hash=catalog.catalog_hash,
        validation_evaluation_cutoff=START + timedelta(days=1),
        outer_folds=(global_v4._invalid_outer_fold(fold, configuration, "cached"),),
        valid_fold_count=0,
        valid_fold_rate=0.0,
        soft_nmi_mean=None,
        soft_nmi_population_std=None,
        soft_nmi_worst=None,
        latest_complete_fold_valid=False,
        production_eligible=False,
        policy_hash="f" * 64,
        failure_reason="cached invalid fold",
    )

    class RunStore:
        def open_run(self, key):
            return EvaluationRunState(
                key.key,
                key.dataset_snapshot_key,
                "COMPLETE",
                cached_result.result_hash,
                1,
            )

        def load_completed_run(self, key):
            del key
            return pickle.dumps(cached_result, protocol=pickle.HIGHEST_PROTOCOL)

    monkeypatch.setattr(
        global_v4,
        "plan_walk_forward",
        lambda timestamps, walk_forward: SimpleNamespace(
            plan_hash="c" * 64,
            evaluation_cutoff=timestamps[-1],
        ),
    )
    assert (
        global_v4.evaluate_global_regime_v4_from_source(
            Source(),
            profile=profile,
            snapshot_store=SnapshotStore(),
            run_store=RunStore(),
            repository_commit_sha="d" * 40,
            uv_lock_sha256="e" * 64,
            python_version="3.14.7",
        )
        == cached_result
    )
