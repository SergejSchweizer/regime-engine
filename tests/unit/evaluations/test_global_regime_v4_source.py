from __future__ import annotations

import pickle
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd  # type: ignore[import-untyped]
import pytest

import market_regime_engine.evaluations.global_regime_v4 as global_v4
from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluation_runs.contracts import DatasetSnapshotIdentity
from market_regime_engine.evaluation_runs.store import EvaluationRunState
from market_regime_engine.features.ports import (
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
    FeatureRequest,
    FeatureRow,
    FeatureSnapshot,
)
from market_regime_engine.profiles.loader import load_profile

START = datetime(2020, 1, 1, tzinfo=UTC)


def _raw_profile():
    profile = load_profile("configs/profiles/xetra_v4.yaml")

    # Keep source-entrypoint fixtures small while preserving the mandatory
    # raw-plus-PCA contract exercised by the production route.
    class SmallPCAProfile:
        pca = SimpleNamespace(variance_threshold=0.90, component_count=2)

        def __getattr__(self, name):
            return getattr(profile, name)

    return SmallPCAProfile()


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


def _production_clock_source(
    *, complete_row_count: int, row_count: int = 1449
) -> tuple[FeatureCatalogSnapshot, FeatureSnapshot]:
    lineage = SourceLineage(
        source_dataset="regime_loader",
        source_build_id="production-clock-build",
        data_sha256="9" * 64,
        schema_version=2,
        feature_version=1,
        source_table="regime_loader",
        synced_at_utc=START,
        row_count=row_count,
        min_timestamp=START,
        max_timestamp=START + timedelta(days=row_count - 1),
    )
    catalog = FeatureCatalogSnapshot.from_entries(
        lineage,
        "timestamp_m1",
        tuple(
            FeatureCatalogEntry(name, ordinal)
            for ordinal, name in enumerate(("feature_a", "feature_b", "feature_c"), start=1)
        ),
    )
    snapshot = FeatureSnapshot(
        lineage,
        catalog.feature_names,
        tuple(
            FeatureRow(
                START + timedelta(days=index),
                (
                    float(index + 1),
                    float((index % 17) ** 2 + index / 1000),
                    float((index % 31) + index / 100),
                )
                if index < complete_row_count
                else (float(index + 1), float(index % 17), None),
            )
            for index in range(row_count)
        ),
    )
    return catalog.with_materialization(snapshot), snapshot


def test_source_clock_preflight_accepts_a_structurally_eligible_synthetic_source() -> None:
    catalog, snapshot = _production_clock_source(complete_row_count=1449)

    potentially_valid = global_v4._require_production_eligible_source_clock(
        catalog,
        snapshot,
        _raw_profile(),
    )

    expected_plan = global_v4.plan_walk_forward(
        tuple(row.timestamp for row in snapshot.rows), _raw_profile().walk_forward
    )
    assert potentially_valid == tuple(range(1, len(expected_plan.folds) + 1))


def test_source_clock_preflight_preserves_the_eighty_percent_outer_gate() -> None:
    catalog, complete_snapshot = _production_clock_source(
        complete_row_count=1575,
        row_count=1575,
    )
    snapshot = FeatureSnapshot(
        complete_snapshot.lineage,
        complete_snapshot.feature_names,
        tuple(
            FeatureRow(
                row.timestamp,
                (row.values[0], row.values[1], None) if 1260 <= index < 1290 else row.values,
            )
            for index, row in enumerate(complete_snapshot.rows)
        ),
    )
    catalog = FeatureCatalogSnapshot.from_entries(
        snapshot.lineage,
        "timestamp_m1",
        catalog.entries,
    ).with_materialization(snapshot)

    potentially_valid = global_v4._require_production_eligible_source_clock(
        catalog,
        snapshot,
        _raw_profile(),
    )

    expected_plan = global_v4.plan_walk_forward(
        tuple(row.timestamp for row in snapshot.rows), _raw_profile().walk_forward
    )
    assert potentially_valid == tuple(range(1, len(expected_plan.folds) + 1))


def test_current_source_clock_preflight_fails_before_pca(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, snapshot = _production_clock_source(complete_row_count=400)

    class Source:
        def read_with_catalog(self, request: FeatureRequest):
            assert request == FeatureRequest.all_features()
            return catalog, snapshot

    def unexpected_pca(*args, **kwargs):
        del args, kwargs
        raise AssertionError("PCA must not run after an impossible source-clock preflight")

    monkeypatch.setattr(global_v4, "fit_and_materialize_pca_source", unexpected_pca)

    with pytest.raises(RuntimeError, match="cannot satisfy production eligibility"):
        global_v4.evaluate_global_regime_v4_from_source(
            Source(),
            profile=_raw_profile(),
            require_production_eligible_source_clock=True,
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
            FeatureRow(START + timedelta(days=2), (4.0, 6.0)),
        ),
    )
    bound_catalog = catalog.with_materialization(snapshot)

    class Source:
        def __init__(self) -> None:
            self.request: FeatureRequest | None = None

        def read_with_catalog(self, request: FeatureRequest):
            self.request = request
            return bound_catalog, snapshot

    source = Source()
    profile = _raw_profile()
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
    assert source.request.mode.value == "schema_discovery"
    rows = captured["rows"]
    assert isinstance(rows, pd.DataFrame)
    assert tuple(rows.columns) == (
        "timestamp_m1",
        "feature_a",
        "feature_b",
        "pca_pc_001",
        "pca_pc_002",
    )
    assert tuple(rows["feature_a"]) == (1.0, 2.0, 4.0)
    assert pd.isna(rows["feature_b"].iloc[0])
    assert rows["feature_b"].iloc[1] == 3.0
    generated_catalog = captured["catalog"]
    assert generated_catalog.feature_names == tuple(rows.columns[1:])
    assert generated_catalog.feature_names[:2] == bound_catalog.feature_names


def test_v4_source_entrypoint_can_persist_only_the_input_snapshot(monkeypatch) -> None:
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
    profile = _raw_profile()

    class Source:
        def read_with_catalog(self, request: FeatureRequest):
            del request
            return bound_catalog, snapshot

    class SnapshotStore:
        def __init__(self) -> None:
            self.finalized = None

        def finalize(self, key, value, *, catalog=None):
            self.finalized = key, value, catalog

        def load(self, key):
            assert self.finalized is not None
            assert self.finalized[0] == key
            return self.finalized[1]

    captured: dict[str, object] = {}

    def fake_evaluate(rows: pd.DataFrame, **kwargs):
        captured["rows"] = rows
        captured.update(kwargs)
        return "evaluated"

    monkeypatch.setattr(global_v4, "evaluate_global_regime_v4", fake_evaluate)
    snapshot_store = SnapshotStore()
    assert (
        global_v4.evaluate_global_regime_v4_from_source(
            Source(), profile=profile, snapshot_store=snapshot_store
        )
        == "evaluated"
    )
    assert snapshot_store.finalized is not None
    assert captured["run_store"] is None
    assert captured["run_identity"] is None


def test_v4_source_entrypoint_fails_closed_for_invalid_snapshot_contracts() -> None:
    lineage = _lineage()
    catalog = FeatureCatalogSnapshot.from_entries(
        lineage,
        "timestamp_m1",
        (FeatureCatalogEntry("feature_a", 1), FeatureCatalogEntry("feature_b", 2)),
    )
    profile = _raw_profile()
    invalid_profile = SimpleNamespace(profile_id="xetra", profile_config_version=99)

    class Source:
        def __init__(self, result) -> None:
            self.result = result

        def read_with_catalog(self, request: FeatureRequest):
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
    profile = _raw_profile()

    class Source:
        def read_with_catalog(self, request: FeatureRequest):
            assert request.feature_names == ()
            return bound_catalog, snapshot

    class SnapshotStore:
        def __init__(self) -> None:
            self.finalized: tuple[DatasetSnapshotIdentity, FeatureSnapshot] | None = None

        def finalize(
            self,
            key: DatasetSnapshotIdentity,
            value: FeatureSnapshot,
            *,
            catalog: FeatureCatalogSnapshot | None = None,
        ) -> None:
            del catalog
            self.finalized = key, value

        def load(self, key: DatasetSnapshotIdentity) -> FeatureSnapshot:
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

        def load_completed_work_unit(self, key, unit):
            del key, unit
            return None

        def claim_work_unit(self, key, unit):
            del key, unit
            return True

        def complete_work_unit(self, key, unit, payload):
            del key, unit, payload

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
    generated_catalog = captured["catalog"]
    assert isinstance(generated_catalog, FeatureCatalogSnapshot)
    expected_dataset_key = DatasetSnapshotIdentity.from_catalog(generated_catalog)
    assert run_store.opened.dataset_snapshot_key == expected_dataset_key.key
    assert captured["run_store"] is run_store
    assert captured["run_identity"] == run_store.opened


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
    profile = _raw_profile()

    class Source:
        def read_with_catalog(self, request: FeatureRequest):
            del request
            return bound_catalog, snapshot

    class SnapshotStore:
        def finalize(self, key, value, *, catalog=None):
            del key, value, catalog

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
