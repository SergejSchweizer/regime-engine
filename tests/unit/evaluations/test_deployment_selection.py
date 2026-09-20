from __future__ import annotations

from dataclasses import replace
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


def _valid_deployment_inputs() -> tuple[
    pd.DataFrame, FeatureCatalogSnapshot, AdaptiveEvaluationResult
]:
    rows = pd.DataFrame(
        {
            "timestamp_m1": [START + timedelta(days=index) for index in range(7)],
            "f0": range(7),
            "f1": range(7),
            "f2": range(7),
        }
    )
    catalog = _catalog(rows)
    return rows, catalog, _validation(catalog.catalog_hash)


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
    assert result.configuration.source_build_id == validation.source_build_id
    assert result.configuration.catalog_hash == catalog.catalog_hash
    assert result.configuration.feature_discovery_hash == result.discovery_hash
    assert result.selection_hash


def test_deployment_selection_never_copies_the_last_outer_fold() -> None:
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
    last_outer_configuration = FinalSelectedConfiguration(
        feature_order=("f0", "f1", "f2"),
        candidate_id="gmm_hmm_k2_m2_full",
        state_count=2,
        model_family="gmm_hmm",
        selected_prefix_length=3,
        feature_discovery_hash="b" * 64,
        source_build_id="build-1",
        catalog_hash=catalog.catalog_hash,
        selection_definition_hash="b" * 64,
        selection_execution_hash="b" * 64,
    )
    validation = replace(
        validation,
        outer_folds=(
            *validation.outer_folds[:-1],
            replace(validation.outer_folds[-1], final_configuration=last_outer_configuration),
        ),
    )

    result = select_deployment_configuration(
        rows,
        catalog=catalog,
        profile=PROFILE,
        validation=validation,
        selector=lambda *_args, **_kwargs: _selector_result(catalog.catalog_hash),
    )

    assert result.configuration.candidate_id == "gaussian_hmm_k2_full"
    assert result.configuration.feature_order == ("f0", "f1")
    assert result.configuration.candidate_id != last_outer_configuration.candidate_id


@pytest.mark.parametrize(
    ("returned_source_build", "returned_catalog_hash", "message"),
    (
        ("other-build", HASH, "different source build"),
        ("build-1", "d" * 64, "different source catalog"),
    ),
)
def test_deployment_selection_rejects_selector_identity_drift(
    returned_source_build: str,
    returned_catalog_hash: str,
    message: str,
) -> None:
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
    selected = _selector_result(returned_catalog_hash)
    selected.source_build_id = returned_source_build

    with pytest.raises(ValueError, match=message):
        select_deployment_configuration(
            rows,
            catalog=catalog,
            profile=PROFILE,
            validation=validation,
            selector=lambda *_args, **_kwargs: selected,
        )


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


def test_deployment_selection_rejects_input_and_cutoff_contract_drift() -> None:
    rows, catalog, validation = _valid_deployment_inputs()

    def selector(**_kwargs: object) -> SimpleNamespace:
        return _selector_result(catalog.catalog_hash)

    with pytest.raises(TypeError, match="pandas DataFrame"):
        select_deployment_configuration(
            object(), catalog=catalog, profile=PROFILE, validation=validation, selector=selector
        )
    with pytest.raises(TypeError, match="feature catalog"):
        select_deployment_configuration(
            rows, catalog=object(), profile=PROFILE, validation=validation, selector=selector
        )
    with pytest.raises(ValueError, match="Xetra public profile"):
        select_deployment_configuration(
            rows,
            catalog=catalog,
            profile=replace(PROFILE, profile_id="other"),
            validation=validation,
            selector=selector,
        )
    with pytest.raises(TypeError, match="completed v4"):
        select_deployment_configuration(
            rows, catalog=catalog, profile=PROFILE, validation=object(), selector=selector
        )
    not_eligible = replace(
        validation,
        outer_folds=(
            *validation.outer_folds[:-1],
            replace(validation.outer_folds[-1], valid=False, failure_reason="failed"),
        ),
        valid_fold_count=2,
        valid_fold_rate=2 / 3,
        latest_complete_fold_valid=False,
        production_eligible=False,
    )
    with pytest.raises(ValueError, match="production-eligible"):
        select_deployment_configuration(
            rows,
            catalog=catalog,
            profile=PROFILE,
            validation=not_eligible,
            selector=selector,
        )
    with pytest.raises(ValueError, match="source build"):
        select_deployment_configuration(
            rows,
            catalog=catalog,
            profile=PROFILE,
            validation=replace(validation, source_build_id="other"),
            selector=selector,
        )
    with pytest.raises(ValueError, match="timestamp bounds"):
        select_deployment_configuration(
            rows,
            catalog=replace(catalog, materialized_max_timestamp=None),
            profile=PROFILE,
            validation=validation,
            selector=selector,
        )
    with pytest.raises(ValueError, match="after validation cutoff"):
        select_deployment_configuration(
            rows,
            catalog=catalog,
            profile=PROFILE,
            validation=replace(
                validation, validation_evaluation_cutoff=catalog.materialized_max_timestamp
            ),
            selector=selector,
        )
    with pytest.raises(ValueError, match="require timestamp_m1"):
        select_deployment_configuration(
            rows.drop(columns=["timestamp_m1"]),
            catalog=catalog,
            profile=PROFILE,
            validation=validation,
            selector=selector,
        )
