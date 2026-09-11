from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluations.deployment_selection import (
    select_deployment_configuration,
)
from market_regime_engine.feature_discovery.contracts import (
    AdaptiveEvaluationResult,
    FinalSelectedConfiguration,
    OuterFoldResult,
)
from market_regime_engine.features.ports import FeatureCatalogEntry, FeatureCatalogSnapshot
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile

PROFILE = load_profile(Path("configs/profiles/xetra_v4.yaml"))
HASH = "a" * 64
START = datetime(2026, 1, 1, tzinfo=UTC)


def _catalog(rows: pd.DataFrame) -> FeatureCatalogSnapshot:
    lineage = SourceLineage(
        source_dataset="regime_loader.regime_features_daily",
        source_build_id="build-1",
        data_sha256="b" * 64,
        schema_version=1,
        feature_version=1,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=START,
        row_count=len(rows),
        min_timestamp=rows["timestamp_m1"].iloc[0],
        max_timestamp=rows["timestamp_m1"].iloc[-1],
    )
    return FeatureCatalogSnapshot(
        lineage=lineage,
        timestamp_column="timestamp_m1",
        entries=tuple(
            FeatureCatalogEntry(feature_name=name, canonical_ordinal=index)
            for index, name in enumerate(("f0", "f1", "f2"), start=1)
        ),
        materialized_feature_data_sha256="c" * 64,
        materialized_row_count=len(rows),
        materialized_min_timestamp=rows["timestamp_m1"].iloc[0],
        materialized_max_timestamp=rows["timestamp_m1"].iloc[-1],
    )


def _configuration(scope: str) -> FinalSelectedConfiguration:
    return FinalSelectedConfiguration(
        feature_order=("f0", "f1"),
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        model_family="gaussian_hmm",
        selected_prefix_length=2,
        feature_discovery_hash=HASH,
        source_build_id="build-1",
        catalog_hash=HASH,
        selection_definition_hash=HASH,
        selection_execution_hash=HASH,
        state_identity_scope=scope,
    )


def _validation(catalog_hash: str = HASH) -> AdaptiveEvaluationResult:
    folds = tuple(
        OuterFoldResult(
            fold_index=index,
            train_start=START + timedelta(days=(index - 1) * 2),
            train_end=START + timedelta(days=(index - 1) * 2),
            test_start=START + timedelta(days=(index - 1) * 2 + 1),
            test_end=START + timedelta(days=(index - 1) * 2 + 1),
            final_configuration=_configuration("outer_fold_local"),
            oos_predictive_loglik_per_observation=-1.0,
            oos_timestamps=(START + timedelta(days=(index - 1) * 2 + 1),),
            oos_filtered_probabilities=((0.5, 0.5),),
            teacher_reference_hash=HASH,
            outer_teacher_final_soft_nmi=0.5,
            outer_shared_timestamp_count=63,
        )
        for index in range(1, 4)
    )
    return AdaptiveEvaluationResult(
        source_build_id="build-1",
        catalog_hash=catalog_hash,
        validation_evaluation_cutoff=folds[-1].test_end,
        outer_folds=folds,
        valid_fold_count=3,
        valid_fold_rate=1.0,
        soft_nmi_mean=0.5,
        soft_nmi_population_std=0.0,
        soft_nmi_worst=0.5,
        latest_complete_fold_valid=True,
        production_eligible=True,
        policy_hash=HASH,
    )


def _selector_result(catalog_hash: str = HASH) -> SimpleNamespace:
    candidate = ResolvedCandidateProfile(
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        covariance_type="full",
        feature_order=("f0", "f1"),
        feature_dimension=2,
        source_build_id="build-1",
        feature_selection_definition_hash=HASH,
        feature_selection_execution_hash=HASH,
        original_feature_universe=("f0", "f1", "f2"),
    )
    return SimpleNamespace(
        source_build_id="build-1",
        final_candidate=candidate,
        feature_discovery_hash=HASH,
        catalog_hash=catalog_hash,
    )


def test_deployment_selection_calls_shared_policy_through_source_maximum() -> None:
    rows = pd.DataFrame(
        {
            "timestamp_m1": [START + timedelta(days=index) for index in range(7)],
            "f0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "f1": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "f2": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        }
    )
    catalog = _catalog(rows)
    validation = _validation(catalog.catalog_hash)
    calls: list[tuple[pd.DataFrame, str, int | None]] = []

    def selector(source_rows, *, catalog, profile, source_build_id, max_workers):
        calls.append((source_rows, source_build_id, max_workers))
        assert catalog is not None
        assert profile is PROFILE
        return _selector_result(catalog.catalog_hash)

    result = select_deployment_configuration(
        rows,
        catalog=catalog,
        profile=PROFILE,
        validation=validation,
        selector=selector,
        max_workers=3,
    )

    assert len(calls) == 1
    assert calls[0][0] is rows
    assert calls[0][1] == "build-1"
    assert calls[0][2] == 3
    assert result.validation_evaluation_cutoff == START + timedelta(days=5)
    assert result.deployment_selection_cutoff == START + timedelta(days=6)
    assert result.configuration.state_identity_scope == "model_version_local"
    assert result.configuration.candidate_id == "gaussian_hmm_k2_full"


def test_deployment_selection_rejects_validation_catalog_drift() -> None:
    rows = pd.DataFrame(
        {
            "timestamp_m1": [START + timedelta(days=index) for index in range(7)],
            "f0": range(7),
            "f1": range(7),
            "f2": range(7),
        }
    )
    catalog = _catalog(rows)
    validation = _validation(catalog.catalog_hash)
    drifted = AdaptiveEvaluationResult(
        source_build_id=validation.source_build_id,
        catalog_hash="d" * 64,
        validation_evaluation_cutoff=validation.validation_evaluation_cutoff,
        outer_folds=validation.outer_folds,
        valid_fold_count=validation.valid_fold_count,
        valid_fold_rate=validation.valid_fold_rate,
        soft_nmi_mean=validation.soft_nmi_mean,
        soft_nmi_population_std=validation.soft_nmi_population_std,
        soft_nmi_worst=validation.soft_nmi_worst,
        latest_complete_fold_valid=validation.latest_complete_fold_valid,
        production_eligible=validation.production_eligible,
        policy_hash=validation.policy_hash,
    )
    with pytest.raises(ValueError, match="catalog differs"):
        select_deployment_configuration(
            rows,
            catalog=catalog,
            profile=PROFILE,
            validation=drifted,
            selector=lambda **_: _selector_result(),
        )


def test_deployment_selection_rejects_non_monotonic_source_rows() -> None:
    rows = pd.DataFrame(
        {
            "timestamp_m1": [
                START,
                START + timedelta(days=1),
                START + timedelta(days=3),
                START + timedelta(days=2),
                START + timedelta(days=4),
                START + timedelta(days=5),
                START + timedelta(days=6),
            ],
            "f0": range(7),
            "f1": range(7),
            "f2": range(7),
        }
    )
    catalog = _catalog(rows)
    validation = _validation(catalog.catalog_hash)
    with pytest.raises(ValueError, match="strictly increasing"):
        select_deployment_configuration(
            rows,
            catalog=catalog,
            profile=PROFILE,
            validation=validation,
            selector=lambda **_: _selector_result(),
        )
