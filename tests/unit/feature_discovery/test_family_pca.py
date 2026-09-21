import numpy as np
import pytest

from market_regime_engine.feature_discovery.family_pca import fit_family_pca
from market_regime_engine.feature_discovery.feature_roles import (
    TEMPORAL_KEY,
    build_feature_role_contract,
)


def _names(count: int) -> tuple[str, ...]:
    return tuple(f"vix_delta_{index}obs" for index in range(1, count + 1))


def test_family_pca_retains_first_nonzero_rank_components_capped_at_eight() -> None:
    names = _names(10)
    index = np.arange(120, dtype=np.float64)
    rows = np.column_stack(
        [
            np.sin(index / 5.0),
            np.cos(index / 7.0),
            np.sin(index / 11.0),
            np.zeros_like(index),
            np.zeros_like(index),
            np.zeros_like(index),
            np.zeros_like(index),
            np.zeros_like(index),
            np.zeros_like(index),
            np.zeros_like(index),
        ]
    )
    # Make the zero columns eligible for matrix construction without changing
    # their rank; the first three columns provide the non-zero rank.
    rows[:, 3:] = rows[:, :1][:, :1] * np.arange(1, 8, dtype=np.float64)
    contract = build_feature_role_contract((TEMPORAL_KEY, *names))
    artifact = fit_family_pca(rows, names, contract)

    assert artifact.family == "vix"
    assert artifact.retained_component_count == 3
    assert artifact.generated_feature_names == (
        "family_pc_vix_1",
        "family_pc_vix_2",
        "family_pc_vix_3",
    )
    assert artifact.cumulative_explained_variance == pytest.approx(1.0)
    assert artifact.transform(rows).shape == (120, 3)


def test_family_pca_does_not_use_explained_variance_to_choose_components() -> None:
    names = _names(3)
    index = np.arange(100, dtype=np.float64)
    rows = np.column_stack((index, np.sin(index), np.cos(index)))
    contract = build_feature_role_contract((TEMPORAL_KEY, *names))
    artifact = fit_family_pca(rows, names, contract)
    assert artifact.retained_component_count == 3
    assert len(artifact.explained_variance_ratio) == 3


def test_family_pca_rejects_mixed_families_and_incomplete_train_rows() -> None:
    vix = "vix_delta_1obs"
    vix9d = "vix9d_delta_1obs"
    contract = build_feature_role_contract((TEMPORAL_KEY, vix, vix9d))
    with pytest.raises(ValueError, match="exactly one family"):
        fit_family_pca(np.ones((20, 2)), (vix, vix9d), contract)
    with pytest.raises(ValueError, match="complete finite"):
        fit_family_pca(
            np.array([[1.0], [np.nan], [3.0]]),
            (vix,),
            build_feature_role_contract((TEMPORAL_KEY, vix)),
        )


def test_family_pca_signs_are_canonical_and_all_loadings_are_exportable() -> None:
    names = _names(2)
    index = np.arange(120, dtype=np.float64)
    rows = np.column_stack((np.sin(index / 5.0), np.cos(index / 7.0)))
    contract = build_feature_role_contract((TEMPORAL_KEY, *names))
    artifact = fit_family_pca(rows, names, contract)

    for component in artifact.components:
        maximum = max(abs(value) for value in component)
        tied = [
            (name, value)
            for name, value in zip(names, component, strict=True)
            if abs(abs(value) - maximum) <= 1.0e-15
        ]
        assert next(value for name, value in tied if name == min(name for name, _ in tied)) > 0.0
    loadings = artifact.pca_loadings("fold-001", "source-build")
    assert len(loadings) == artifact.retained_component_count * len(names)
    assert all(item.profile_hash == contract.profile.profile_hash for item in loadings)
    assert all(item.squared_loading == pytest.approx(item.loading**2) for item in loadings)
