from dataclasses import replace
from datetime import UTC, datetime

import pytest

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.feature_discovery.feature_roles import (
    CORE_FEATURES,
    CORRELATION_ABS_THRESHOLD,
    CORRELATION_MIN_PAIR_ROWS,
    CORRELATION_MIN_SUBWINDOW_ROWS,
    CORRELATION_SUBWINDOW_ABS_THRESHOLD,
    CORRELATION_SUBWINDOWS,
    FAMILY_NEAR_DUPLICATE_ABS_THRESHOLD,
    FAMILY_NEAR_DUPLICATE_SUBWINDOW_ABS_THRESHOLD,
    FAMILY_PCA_MAX_COMPONENTS,
    SFFS_MAX_FEATURES,
    TEMPORAL_KEY,
    FeatureRole,
    FeatureSelectionProfile,
    FeatureStage,
    build_feature_role_contract,
    build_feature_role_contract_from_catalog,
    classify_feature_name,
    family_pc_name,
)
from market_regime_engine.features.ports import FeatureCatalogEntry, FeatureCatalogSnapshot


def test_canonical_core_inventory_is_exactly_twenty() -> None:
    assert len(CORE_FEATURES) == 20
    assert len(set(CORE_FEATURES)) == 20
    assert classify_feature_name("usd_broad_log_level").role is FeatureRole.CORE
    assert classify_feature_name("usd_broad_log_return_20obs").family == "usd_broad"


@pytest.mark.parametrize(
    "name",
    (
        "vix_delta_1obs",
        "vix9d_zscore_20obs",
        "vix3m_momentum_autocorr_60obs",
        "vix6m_return_geom_20obs",
        "vstoxx_delta_5obs",
        "ciss_zscore_252obs",
        "euro_hy_oas_return_geom_20obs",
        "us_2y_delta_1obs",
        "us_10y_momentum_autocorr_20obs",
        "estr_return_geom_5obs",
        "usd_broad_log_return_20obs",
    ),
)
def test_generated_transformations_are_family_inputs_not_direct_hmm_features(name: str) -> None:
    assignment = classify_feature_name(name)
    assert assignment.role is FeatureRole.TRANSFORMATION
    assert not assignment.direct_hmm_candidate
    assert assignment.family_pca_input


def test_temporal_key_is_neither_quality_or_model_input() -> None:
    assignment = classify_feature_name(TEMPORAL_KEY)
    assert assignment.role is FeatureRole.TEMPORAL_KEY
    assert not assignment.direct_hmm_candidate
    assert not assignment.family_pca_input

    contract = build_feature_role_contract((TEMPORAL_KEY, *CORE_FEATURES))
    with pytest.raises(ValueError, match="only core"):
        contract.validate_hmm_features((TEMPORAL_KEY,))


def test_contract_fails_closed_for_unknown_future_columns() -> None:
    with pytest.raises(ValueError, match="unclassifiable"):
        classify_feature_name("new_macro_level")


def test_contract_exposes_only_core_as_direct_hmm_candidates() -> None:
    contract = build_feature_role_contract(
        (TEMPORAL_KEY, *CORE_FEATURES, "usd_broad_log_return_20obs")
    )
    assert contract.temporal_keys == (TEMPORAL_KEY,)
    assert contract.core_features == CORE_FEATURES
    assert contract.direct_hmm_candidates == CORE_FEATURES
    assert contract.family_pca_inputs == ("usd_broad_log_return_20obs",)
    assert contract.validate_hmm_features((CORE_FEATURES[0],)) == (CORE_FEATURES[0],)
    with pytest.raises(ValueError, match="only core"):
        contract.validate_hmm_features(("usd_broad_log_return_20obs",))


def test_complete_catalog_requires_temporal_key_and_all_twenty_core_features() -> None:
    complete = build_feature_role_contract((TEMPORAL_KEY, *CORE_FEATURES))
    assert complete.validate_complete_catalog() == (TEMPORAL_KEY, *CORE_FEATURES)

    missing_core = build_feature_role_contract((TEMPORAL_KEY, *CORE_FEATURES[:-1]))
    with pytest.raises(ValueError, match="missing canonical core"):
        missing_core.validate_complete_catalog()

    no_temporal = build_feature_role_contract(CORE_FEATURES)
    with pytest.raises(ValueError, match="exactly timestamp_m1"):
        no_temporal.validate_complete_catalog()


def test_catalog_builder_classifies_every_discovered_non_core_column() -> None:
    catalog = FeatureCatalogSnapshot.from_entries(
        SourceLineage(
            source_dataset="macro_features",
            source_build_id="build-1",
            data_sha256="a" * 64,
            schema_version=1,
            feature_version=1,
            source_table="macro_loader.macro_features",
            synced_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        TEMPORAL_KEY,
        tuple(
            FeatureCatalogEntry(name, index + 1)
            for index, name in enumerate((*CORE_FEATURES, "usd_broad_log_return_20obs"))
        ),
    )

    contract = build_feature_role_contract_from_catalog(catalog)
    assert contract.validate_complete_catalog() == (
        TEMPORAL_KEY,
        *CORE_FEATURES,
        "usd_broad_log_return_20obs",
    )
    assert contract.family_pca_inputs == ("usd_broad_log_return_20obs",)


def test_catalog_builder_fails_closed_for_unknown_discovered_column() -> None:
    with pytest.raises(ValueError, match="unclassifiable"):
        build_feature_role_contract_from_catalog(
            FeatureCatalogSnapshot.from_entries(
                SourceLineage(
                    source_dataset="macro_features",
                    source_build_id="build-1",
                    data_sha256="a" * 64,
                    schema_version=1,
                    feature_version=1,
                    source_table="macro_loader.macro_features",
                    synced_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
                ),
                TEMPORAL_KEY,
                tuple(
                    FeatureCatalogEntry(name, index + 1)
                    for index, name in enumerate((*CORE_FEATURES, "future_unknown_column"))
                ),
            )
        )


def test_stage_boundaries_reject_temporal_and_direct_generated_inputs() -> None:
    generated = "usd_broad_log_return_20obs"
    contract = build_feature_role_contract((TEMPORAL_KEY, *CORE_FEATURES, generated))
    family_pc = family_pc_name("usd_broad", 1)

    assert contract.validate_stage_features(FeatureStage.QUALITY, (generated,)) == (generated,)
    assert contract.validate_stage_features(FeatureStage.FAMILY_PCA, (generated,)) == (generated,)
    assert contract.validate_stage_features(FeatureStage.CORRELATION, (family_pc,)) == (family_pc,)
    assert contract.validate_stage_features(FeatureStage.SFFS, (family_pc,)) == (family_pc,)
    assert contract.validate_stage_features(FeatureStage.HMM, (CORE_FEATURES[0], family_pc)) == (
        CORE_FEATURES[0],
        family_pc,
    )

    with pytest.raises(ValueError, match="temporal key"):
        contract.validate_stage_features(FeatureStage.QUALITY, (TEMPORAL_KEY,))
    with pytest.raises(ValueError, match="only as family PCs"):
        contract.validate_stage_features(FeatureStage.HMM, (generated,))
    with pytest.raises(ValueError, match="transformations only"):
        contract.validate_stage_features(FeatureStage.FAMILY_PCA, (CORE_FEATURES[0],))


def test_family_pc_identity_is_bounded_and_fail_closed() -> None:
    assert family_pc_name("usd_broad", 8) == "family_pc_usd_broad_8"
    with pytest.raises(ValueError):
        family_pc_name("unknown", 1)
    with pytest.raises(ValueError):
        family_pc_name("usd_broad", 9)


def test_profile_persists_the_canonical_selection_defaults() -> None:
    profile = FeatureSelectionProfile()
    assert (
        profile.family_near_duplicate_abs_threshold == FAMILY_NEAR_DUPLICATE_ABS_THRESHOLD == 0.995
    )
    assert (
        profile.family_near_duplicate_subwindow_abs_threshold
        == (FAMILY_NEAR_DUPLICATE_SUBWINDOW_ABS_THRESHOLD)
        == 0.99
    )
    assert profile.family_pca_max_components == FAMILY_PCA_MAX_COMPONENTS == 8
    assert profile.correlation_abs_threshold == CORRELATION_ABS_THRESHOLD == 0.95
    assert (
        profile.correlation_subwindow_abs_threshold == CORRELATION_SUBWINDOW_ABS_THRESHOLD == 0.90
    )
    assert profile.correlation_subwindows == CORRELATION_SUBWINDOWS == 3
    assert profile.correlation_min_pair_rows == CORRELATION_MIN_PAIR_ROWS == 30
    assert profile.correlation_min_subwindow_rows == CORRELATION_MIN_SUBWINDOW_ROWS == 10
    assert profile.sffs_max_features == SFFS_MAX_FEATURES == 10


def test_profile_and_role_mutations_change_hash() -> None:
    baseline = build_feature_role_contract((TEMPORAL_KEY, *CORE_FEATURES))
    changed_profile = replace(baseline.profile, correlation_abs_threshold=0.95)
    assert changed_profile.profile_hash == baseline.profile.profile_hash
    changed_assignment = replace(baseline.assignments[0], role=FeatureRole.CORE)
    changed = replace(baseline, assignments=(changed_assignment, *baseline.assignments[1:]))
    assert changed.profile_hash != baseline.profile_hash


def test_contract_exposes_complete_evidence_identity() -> None:
    contract = build_feature_role_contract((TEMPORAL_KEY, *CORE_FEATURES))
    metadata = contract.evidence_metadata()
    assert metadata["feature_selection_profile_version"] == "macro_regime_feature_selection_v1"
    assert metadata["feature_selection_profile_hash"] == contract.profile.profile_hash
    assert metadata["feature_role_contract_hash"] == contract.contract_hash
    assert len(metadata["feature_role_assignments"]) == 21  # type: ignore[arg-type]
