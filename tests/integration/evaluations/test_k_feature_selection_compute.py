from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import market_regime_engine.evaluation.walk_forward as walk_forward
from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluations.k_feature_selection import (
    run_real_k_feature_selection,
)
from market_regime_engine.features.ports import FeatureCatalogEntry, FeatureCatalogSnapshot
from market_regime_engine.models.protocols import FitResult
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.training.multistart import (
    MULTISTART_SEEDS,
    MultistartResult,
    StartDiagnostic,
)

pytestmark = pytest.mark.integration

START = datetime(2020, 1, 1, tzinfo=UTC)


def _rows() -> pd.DataFrame:
    count = 1323
    index = np.arange(count, dtype=np.float64)
    hidden = np.where((np.arange(count) // 47) % 2 == 0, -1.0, 1.0)
    noise = np.random.default_rng(9001)
    base = np.column_stack(
        (
            hidden + noise.normal(0.0, 0.15, count),
            0.75 * hidden + noise.normal(0.0, 0.15, count),
            np.sin(index / 19.0) + noise.normal(0.0, 0.05, count),
            *(noise.normal(0.0, 0.5, count) for _ in range(5)),
        )
    )
    raw = np.column_stack(
        (base, base + np.column_stack(tuple(noise.normal(0.0, 0.01, count) for _ in range(8))))
    )
    pca_weights = np.random.default_rng(9002).normal(size=(16, 8))
    return pd.DataFrame(
        {
            "timestamp_m1": tuple(START + timedelta(days=int(value)) for value in index),
            **{f"f{feature_index}": raw[:, feature_index] for feature_index in range(16)},
            **{
                f"pca_pc_{component:03d}": (raw @ pca_weights)[:, component - 1]
                for component in range(1, 9)
            },
        }
    )


def _catalog(rows: pd.DataFrame) -> FeatureCatalogSnapshot:
    names = tuple(column for column in rows.columns if column != "timestamp_m1")
    lineage = SourceLineage(
        source_dataset="macro_features_daily",
        source_build_id="fixed-k-build",
        data_sha256="d" * 64,
        schema_version=4,
        feature_version=3,
        source_table="macro_loader.macro_features_daily",
        synced_at_utc=START,
        row_count=len(rows),
        min_timestamp=rows["timestamp_m1"].iloc[0],
        max_timestamp=rows["timestamp_m1"].iloc[-1],
    )
    entries = tuple(
        FeatureCatalogEntry(
            name,
            ordinal,
            schema_name=("regime_engine" if name.startswith("pca_pc_") else "macro_loader"),
            relation_name=(
                "pca_generated_features" if name.startswith("pca_pc_") else "macro_features_daily"
            ),
            relation_kind=("MATERIALIZED VIEW" if name.startswith("pca_pc_") else "BASE TABLE"),
        )
        for ordinal, name in enumerate(names, 1)
    )
    return FeatureCatalogSnapshot.from_entries(lineage, "timestamp_m1", entries)


def _one_real_fit_multistart(train_rows, *, state_count, adapter_factory, **_kwargs):
    result: FitResult = adapter_factory().fit(train_rows, state_count, MULTISTART_SEEDS[0])
    diagnostics = tuple(
        StartDiagnostic(
            seed=seed,
            success=True,
            converged=True,
            iterations=result.iterations,
            train_log_likelihood=result.train_log_likelihood,
            artifact=result.artifact,
            failure_reason=None,
        )
        for seed in MULTISTART_SEEDS
    )
    return MultistartResult(state_count=state_count, winner=result, diagnostics=diagnostics)


def test_fixed_k_selection_runs_real_train_only_discovery_and_prefix(monkeypatch) -> None:
    monkeypatch.setattr(walk_forward, "run_multistart", _one_real_fit_multistart)
    rows = _rows()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    result = run_real_k_feature_selection(
        rows,
        catalog=_catalog(rows),
        profile=profile,
        source_snapshot_id="snapshot-fixed-k",
        source_build_id="fixed-k-build",
        validation_cutoff=rows["timestamp_m1"].iloc[-1],
        deployment_cutoff=rows["timestamp_m1"].iloc[-1] + timedelta(days=1),
        requested_state_counts=(2, 3, 4, 5),
        max_workers=4,
    )
    assert tuple(item.state_count for item in result) == (2, 3, 4, 5)
    assert tuple(item.eligible for item in result) == (True, True, False, True)
    assert tuple(item.selection.slot_id for item in result if item.selection) == (
        "k2",
        "k3",
        "k5",
    )
    assert tuple(item.selection.state_count for item in result if item.selection) == (2, 3, 5)
    assert all(
        item.selection.candidate_identity == f"gaussian_hmm_k{item.state_count}_full"
        for item in result
        if item.selection is not None
    )
    assert all(
        len(item.selection.feature_order) >= 2
        and item.selection.reference_teacher_id.startswith(
            f"gaussian_hmm_k{item.state_count}_full:"
        )
        for item in result
        if item.selection is not None
    )
    k4 = result[2]
    assert k4.selection is None
    assert k4.rejection_reason and "no eligible regime feature score" in k4.rejection_reason
