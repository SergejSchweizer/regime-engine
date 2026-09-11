from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd  # type: ignore[import-untyped]

import market_regime_engine.evaluations.global_regime_v4 as global_v4
from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardFold
from market_regime_engine.evaluation_runs.contracts import (
    DatasetSnapshotIdentity,
    EvaluationRunIdentity,
)
from market_regime_engine.evaluation_runs.store import SQLiteEvaluationRunStore
from market_regime_engine.features.ports import (
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
    FeatureRow,
    FeatureSnapshot,
)
from market_regime_engine.profiles.loader import load_profile

START = datetime(2020, 1, 1, tzinfo=UTC)


def test_global_v4_reuses_completed_outer_folds_after_restart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    timestamps = tuple(START + timedelta(days=index) for index in range(1_386))
    lineage = SourceLineage(
        source_dataset="synthetic",
        source_build_id="build-1",
        data_sha256="a" * 64,
        schema_version=2,
        feature_version=4,
        source_table="regime_loader.synthetic",
        synced_at_utc=START,
        row_count=len(timestamps),
        min_timestamp=timestamps[0],
        max_timestamp=timestamps[-1],
    )
    names = ("feature_a", "feature_b")
    snapshot = FeatureSnapshot(
        lineage,
        names,
        tuple(FeatureRow(timestamp, (1.0, 2.0)) for timestamp in timestamps),
    )
    catalog = FeatureCatalogSnapshot.from_entries(
        lineage,
        "timestamp_m1",
        tuple(FeatureCatalogEntry(name, index) for index, name in enumerate(names, 1)),
    ).with_materialization(snapshot)
    source_rows = pd.DataFrame(
        {
            "timestamp_m1": timestamps,
            "feature_a": [1.0] * len(timestamps),
            "feature_b": [2.0] * len(timestamps),
        }
    )
    folds = tuple(
        WalkForwardFold(
            fold_index=index,
            fold_id=f"fold_{index:03d}",
            train_start=timestamps[0],
            train_end=timestamps[train_count - 1],
            test_start=timestamps[train_count],
            test_end=timestamps[train_count + 62],
            train_source_observations=train_count,
            test_source_observations=63,
        )
        for index, train_count in ((1, 1_260), (2, 1_323))
    )
    plan = SimpleNamespace(plan_hash="b" * 64, evaluation_cutoff=timestamps[-1], folds=folds)
    monkeypatch.setattr(global_v4, "plan_walk_forward", lambda *_args: plan)
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    dataset_key = DatasetSnapshotIdentity.from_catalog(catalog)
    run_identity = EvaluationRunIdentity(
        evaluation_id="global_regime_v4",
        profile_id="xetra",
        profile_config_version=4,
        profile_hash=profile.profile_hash,
        evaluation_contract_version=1,
        evaluation_plan_hash=plan.plan_hash,
        dataset_snapshot_key=dataset_key.key,
        evaluation_cutoff=plan.evaluation_cutoff,
        repository_commit_sha="c" * 40,
        uv_lock_sha256="d" * 64,
        python_version="3.14.7",
    )
    store = SQLiteEvaluationRunStore(tmp_path / "runs")
    store.open_run(run_identity)
    calls: list[int] = []

    def fake_outer_fold(*args, **kwargs):
        del kwargs
        fold = args[1]
        calls.append(fold.fold_index)
        configuration = global_v4._fallback_configuration(catalog, "build-1", fold, "test")
        return global_v4._invalid_outer_fold(fold, configuration, "synthetic test invalidity")

    fake_outer_fold.__module__ = global_v4.__name__
    process_pool_worker_counts: list[int] = []

    class _CompletedFuture:
        def __init__(self, value: object) -> None:
            self._value = value

        def result(self) -> object:
            return self._value

    class _InlineProcessPool:
        def __init__(self, *, max_workers: int, **_kwargs: object) -> None:
            process_pool_worker_counts.append(max_workers)

        def __enter__(self) -> _InlineProcessPool:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(self, function, fold):
            return _CompletedFuture(function(fold))

    monkeypatch.setattr(global_v4, "_evaluate_outer_fold", fake_outer_fold)
    monkeypatch.setattr(global_v4, "ProcessPoolExecutor", _InlineProcessPool)
    first = global_v4.evaluate_global_regime_v4(
        source_rows,
        catalog=catalog,
        profile=profile,
        run_store=store,
        run_identity=run_identity,
    )
    assert sorted(calls) == [1, 2]
    assert process_pool_worker_counts == [2]

    second = global_v4.evaluate_global_regime_v4(
        source_rows,
        catalog=catalog,
        profile=profile,
        run_store=store,
        run_identity=run_identity,
    )
    assert second == first
    assert sorted(calls) == [1, 2]
