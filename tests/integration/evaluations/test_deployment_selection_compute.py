from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
import pytest

import market_regime_engine.evaluation.walk_forward as walk_forward
import market_regime_engine.evaluations.teacher_reference as teacher_reference
from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluations.deployment_selection import (
    select_deployment_configuration,
)
from market_regime_engine.feature_discovery.contracts import (
    AdaptiveEvaluationResult,
    FinalSelectedConfiguration,
    OuterFoldResult,
)
from market_regime_engine.features.ports import (
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
    FeatureRow,
    FeatureSnapshot,
)
from market_regime_engine.models.protocols import FitResult, GaussianHMMAdapter
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.training.multistart import (
    MULTISTART_SEEDS,
    MultistartResult,
    StartDiagnostic,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

START = datetime(2020, 1, 1, tzinfo=UTC)
HASH = "b" * 64


def _rows(count: int = 1_449) -> pd.DataFrame:
    index = np.arange(count, dtype=np.float64)
    regime = (index.astype(np.int64) // 35) % 2
    return pd.DataFrame(
        {
            "timestamp_m1": tuple(START + timedelta(days=int(value)) for value in index),
            "f0": np.where(regime == 0, -1.0, 1.0) + 0.12 * np.sin(index / 5.0),
            "f1": np.where(regime == 0, 1.0, -1.0) + 0.12 * np.cos(index / 7.0),
            "f2": np.sin(index / 13.0),
        }
    )


def _catalog(rows: pd.DataFrame) -> FeatureCatalogSnapshot:
    timestamps = tuple(rows["timestamp_m1"])
    lineage = SourceLineage(
        source_dataset="synthetic_deployment_selection",
        source_build_id="synthetic-build",
        data_sha256=HASH,
        schema_version=4,
        feature_version=4,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=START,
        row_count=len(rows),
        min_timestamp=timestamps[0],
        max_timestamp=timestamps[-1],
    )
    entries = tuple(
        FeatureCatalogEntry(name, ordinal) for ordinal, name in enumerate(("f0", "f1", "f2"), 1)
    )
    raw_catalog = FeatureCatalogSnapshot.from_entries(lineage, "timestamp_m1", entries)
    snapshot = FeatureSnapshot(
        lineage=lineage,
        feature_names=raw_catalog.feature_names,
        rows=tuple(
            FeatureRow(
                timestamp=(
                    timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
                ),
                values=tuple(float(rows.iloc[index][name]) for name in raw_catalog.feature_names),
            )
            for index, timestamp in enumerate(timestamps)
        ),
    )
    return raw_catalog.with_materialization(snapshot)


def _validation(catalog: FeatureCatalogSnapshot) -> AdaptiveEvaluationResult:
    configuration = FinalSelectedConfiguration(
        feature_order=("f0", "f1"),
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        model_family="gaussian_hmm",
        selected_prefix_length=2,
        feature_discovery_hash=HASH,
        source_build_id=catalog.lineage.source_build_id,
        catalog_hash=catalog.catalog_hash,
        selection_definition_hash=HASH,
        selection_execution_hash=HASH,
    )
    folds = tuple(
        OuterFoldResult(
            fold_index=index,
            train_start=START + timedelta(days=(index - 1) * 2),
            train_end=START + timedelta(days=(index - 1) * 2),
            test_start=START + timedelta(days=(index - 1) * 2 + 1),
            test_end=START + timedelta(days=(index - 1) * 2 + 1),
            final_configuration=configuration,
            oos_predictive_loglik_per_observation=-1.0,
            oos_timestamps=(START + timedelta(days=(index - 1) * 2 + 1),),
            oos_filtered_probabilities=((0.5, 0.5),),
            teacher_reference_hash=HASH,
            outer_teacher_final_soft_nmi=0.5,
            outer_shared_timestamp_count=42,
        )
        for index in range(1, 4)
    )
    return AdaptiveEvaluationResult(
        source_build_id=catalog.lineage.source_build_id,
        catalog_hash=catalog.catalog_hash,
        validation_evaluation_cutoff=START + timedelta(days=1_300),
        outer_folds=folds,
        valid_fold_count=len(folds),
        valid_fold_rate=1.0,
        soft_nmi_mean=0.5,
        soft_nmi_population_std=0.0,
        soft_nmi_worst=0.5,
        latest_complete_fold_valid=True,
        production_eligible=True,
        policy_hash=HASH,
    )


def _one_real_fit_multistart(
    train_rows: npt.ArrayLike,
    *,
    state_count: int,
    adapter_factory: Callable[[], GaussianHMMAdapter],
    **_kwargs: object,
) -> MultistartResult:
    """Keep this integration proof real while avoiding repeated seed cost."""

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


def test_full_synthetic_deployment_selection_uses_latest_source_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows()
    catalog = _catalog(rows)
    profile = load_profile("configs/profiles/xetra_v4.yaml")

    monkeypatch.setattr(walk_forward, "run_multistart", _one_real_fit_multistart)
    monkeypatch.setattr(teacher_reference, "run_multistart", _one_real_fit_multistart)
    result = select_deployment_configuration(
        rows,
        catalog=catalog,
        profile=profile,
        validation=_validation(catalog),
        max_workers=None,
    )

    assert result.deployment_selection_cutoff == rows["timestamp_m1"].iloc[-1]
    assert result.validation_evaluation_cutoff < result.deployment_selection_cutoff
    assert result.source_build_id == catalog.lineage.source_build_id
    assert result.source_catalog_hash == catalog.catalog_hash
    assert result.configuration.state_identity_scope == "model_version_local"
    assert result.configuration.catalog_hash == catalog.catalog_hash
    assert result.configuration.feature_order
