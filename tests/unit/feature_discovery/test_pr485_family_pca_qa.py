from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from market_regime_engine.feature_discovery.family_pca import (
    _canonicalize_signs,
    fit_family_pca,
)
from market_regime_engine.feature_discovery.feature_roles import (
    TEMPORAL_KEY,
    FeatureRoleContract,
    FeatureSelectionProfile,
    build_feature_role_contract,
)


def names(family: str, count: int) -> tuple[str, ...]:
    return tuple(f"{family}_delta_{index}obs" for index in range(1, count + 1))


def contract_for(family: str, count: int) -> tuple[tuple[str, ...], FeatureRoleContract]:
    feature_names = names(family, count)
    return feature_names, build_feature_role_contract((TEMPORAL_KEY, *feature_names))


def reference_svd(
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(rows, axis=0, dtype=np.float64)
    variances = np.var(rows, axis=0, ddof=0, dtype=np.float64)
    standardized = (rows - means) / np.sqrt(variances)
    _u, singular_values, components = np.linalg.svd(standardized, full_matrices=False)
    explained = np.square(singular_values, dtype=np.float64)
    return means, variances, components, explained / np.sum(explained)


def test_independent_svd_reference_reproduces_scaler_rank_and_loadings() -> None:
    feature_names, contract = contract_for("vix", 3)
    index = np.arange(150, dtype=np.float64)
    rows = np.column_stack((np.sin(index / 7.0), np.cos(index / 11.0), index / 50.0))
    artifact = fit_family_pca(rows, feature_names, contract)
    means, variances, components, explained = reference_svd(rows)

    assert artifact.scaler.means == pytest.approx(means)
    assert artifact.scaler.variances == pytest.approx(variances)
    assert artifact.numerical_rank == 3
    canonical_components = _canonicalize_signs(components, feature_names)
    assert np.asarray(artifact.components) == pytest.approx(canonical_components)
    assert artifact.explained_variance_ratio == pytest.approx(explained)


@pytest.mark.parametrize("feature_count", (1, 2, 8, 9, 101))
def test_rank_boundaries_never_emit_more_than_eight_components(feature_count: int) -> None:
    feature_names, contract = contract_for("vix", feature_count)
    rows = np.random.default_rng(1234).normal(size=(feature_count + 20, feature_count))
    artifact = fit_family_pca(rows, feature_names, contract)
    assert artifact.retained_component_count == min(feature_count, 8)
    assert len(artifact.generated_feature_names) <= 8


def test_test_mutation_cannot_change_train_fit_or_identity() -> None:
    feature_names, contract = contract_for("vix", 3)
    train = np.arange(90, dtype=np.float64).reshape(30, 3) + np.array([0.0, 1.0, 3.0])
    test_a = np.ones((10, 3), dtype=np.float64)
    test_b = np.full((10, 3), 1.0e12, dtype=np.float64)
    first = fit_family_pca(train, feature_names, contract)
    second = fit_family_pca(train, feature_names, contract)
    assert np.array_equal(first.transform(test_a), second.transform(test_a))
    assert first.fit_hash == second.fit_hash
    assert test_b.shape == test_a.shape


def test_sign_reversal_and_family_isolation_preserve_semantic_identity() -> None:
    feature_names, contract = contract_for("vix", 3)
    rows = np.random.default_rng(55).normal(size=(100, 3))
    artifact = fit_family_pca(rows, feature_names, contract)
    components = np.asarray(artifact.components)
    assert np.array_equal(
        _canonicalize_signs(components, feature_names),
        _canonicalize_signs(-components, feature_names),
    )

    usd_names, usd_contract = contract_for("usd_broad", 3)
    usd_artifact = fit_family_pca(rows + 4.0, usd_names, usd_contract)
    assert artifact.family == "vix"
    assert usd_artifact.family == "usd_broad"
    assert artifact.generated_feature_names != usd_artifact.generated_feature_names


def test_explained_variance_is_diagnostic_not_a_retention_threshold() -> None:
    feature_names, contract = contract_for("vix", 4)
    rows = np.random.default_rng(9).normal(size=(100, 4))
    artifact = fit_family_pca(rows, feature_names, contract)
    profile = replace(FeatureSelectionProfile(), sffs_max_features=10)
    assert artifact.retained_component_count == 4
    assert profile.family_pca_max_components == 8
