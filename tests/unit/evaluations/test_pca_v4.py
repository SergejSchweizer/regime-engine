from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluations.global_regime_v4 import V4ConfigurationSelection
from market_regime_engine.evaluations.pca_v4 import (
    evaluate_global_regime_v4_with_pca,
    select_v4_configuration_with_pca,
)
from market_regime_engine.feature_discovery.contracts import AdaptiveEvaluationResult
from market_regime_engine.features.ports import FeatureCatalogEntry, FeatureCatalogSnapshot
from market_regime_engine.preprocessing import (
    PCAGeneratedFeatureSet,
    fit_pca_inner_train,
    materialize_pca_generated_features,
)
from market_regime_engine.profiles.config import ModelProfile, PCAConfig
from market_regime_engine.profiles.loader import load_profile

START = datetime(2026, 1, 1, tzinfo=UTC)


def _generated() -> PCAGeneratedFeatureSet:
    index = np.arange(40, dtype=np.float64)
    timestamps = tuple(START + timedelta(days=int(value)) for value in index)
    rows = np.column_stack((index, index * 2.0 + 1.0, np.sin(index / 4.0)))
    lineage = SourceLineage(
        source_dataset="synthetic-pca-v4",
        source_build_id="build-pca",
        data_sha256="a" * 64,
        schema_version=4,
        feature_version=4,
        source_table="regime_loader.features",
        synced_at_utc=START,
        row_count=len(rows),
        min_timestamp=timestamps[0],
        max_timestamp=timestamps[-1],
    )
    catalog = FeatureCatalogSnapshot.from_entries(
        lineage,
        "timestamp_m1",
        tuple(
            FeatureCatalogEntry(name, ordinal) for ordinal, name in enumerate(("a", "b", "c"), 1)
        ),
    )
    fit = fit_pca_inner_train(
        timestamps,
        rows,
        feature_order=catalog.feature_names,
        inner_fold_id="inner_fold_001",
        fit_start=START,
        fit_end=START + timedelta(days=20),
    )
    return materialize_pca_generated_features(catalog, fit, timestamps, rows)


def test_pca_generated_snapshot_reaches_shared_v4_selector_with_full_catalog() -> None:
    generated = _generated()
    calls: list[
        tuple[pd.DataFrame, FeatureCatalogSnapshot, int | None, str, tuple[str, ...], float]
    ] = []

    def selector(
        rows: pd.DataFrame,
        *,
        catalog: FeatureCatalogSnapshot,
        profile: ModelProfile,
        source_build_id: str,
        max_workers: int | None,
        pca_raw_feature_order: tuple[str, ...],
        pca_variance_threshold: float,
    ) -> V4ConfigurationSelection:
        calls.append(
            (
                rows,
                catalog,
                max_workers,
                source_build_id,
                pca_raw_feature_order,
                pca_variance_threshold,
            )
        )
        assert profile.profile_id == "xetra"
        return cast(V4ConfigurationSelection, sentinel)

    profile = replace(
        load_profile("configs/profiles/xetra_v4.yaml"),
        pca=PCAConfig(variance_threshold=0.90),
    )
    sentinel = object()
    result = select_v4_configuration_with_pca(
        generated,
        profile=profile,
        selector=selector,
        max_workers=86,
    )

    assert result is sentinel
    assert len(calls) == 1
    frame, catalog, worker_budget, source_build_id, raw_order, threshold = calls[0]
    assert tuple(frame.columns) == ("timestamp_m1", *generated.catalog.feature_names)
    assert len(frame) == generated.snapshot.row_count
    assert catalog == generated.catalog
    assert worker_budget == 86
    assert source_build_id == generated.catalog.lineage.source_build_id
    assert raw_order == generated.raw_feature_names
    assert threshold == 0.90


def test_pca_generated_snapshot_reaches_complete_global_policy() -> None:
    generated = _generated()
    calls: list[tuple[pd.DataFrame, FeatureCatalogSnapshot, int | None, str]] = []

    def evaluator(
        rows: pd.DataFrame,
        *,
        catalog: FeatureCatalogSnapshot,
        profile: ModelProfile,
        source_build_id: str,
        max_workers: int | None,
        pca_raw_feature_order: tuple[str, ...],
        pca_variance_threshold: float,
    ) -> AdaptiveEvaluationResult:
        calls.append((rows, catalog, max_workers, source_build_id))
        assert pca_raw_feature_order == generated.raw_feature_names
        assert pca_variance_threshold == 0.90
        return cast(AdaptiveEvaluationResult, sentinel)

    profile = replace(
        load_profile("configs/profiles/xetra_v4.yaml"),
        pca=PCAConfig(variance_threshold=0.90),
    )
    sentinel = object()
    result = evaluate_global_regime_v4_with_pca(
        generated,
        profile=profile,
        evaluator=evaluator,
        max_workers=86,
    )

    assert result is sentinel
    assert len(calls) == 1
    frame, catalog, worker_budget, source_build_id = calls[0]
    assert tuple(frame.columns) == ("timestamp_m1", *generated.catalog.feature_names)
    assert catalog == generated.catalog
    assert worker_budget == 86
    assert source_build_id == generated.catalog.lineage.source_build_id
