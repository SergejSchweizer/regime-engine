import pytest

from market_regime_engine.feature_discovery.feature_roles import (
    TEMPORAL_KEY,
    build_feature_role_contract,
    family_pc_name,
)
from market_regime_engine.feature_discovery.global_reduction import (
    prune_global_correlated_features,
)


def test_global_pruning_keeps_the_canonical_first_stable_representative() -> None:
    base = tuple(float(index) for index in range(30))
    inverse = tuple(-value for value in base)
    family_pc = family_pc_name("usd_broad", 1)
    names = ("vix_log_level", "us_10y_log_level", family_pc)
    contract = build_feature_role_contract((TEMPORAL_KEY, *names[:2]))
    result = prune_global_correlated_features(
        {names[0]: base, names[1]: inverse, names[2]: base},
        contract,
    )

    assert result.representatives == (names[0],)
    assert result.removed_features == (names[1], family_pc)
    assert len(result.evidence) == 2
    assert result.evidence[0].full_absolute_pearson == pytest.approx(1.0)
    assert result.evidence[0].subwindow_absolute_pearsons == pytest.approx((1.0, 1.0, 1.0))


def test_global_pruning_rejects_transformations_and_is_stable_only() -> None:
    names = ("vix_log_level", "vix_delta_1obs")
    contract = build_feature_role_contract((TEMPORAL_KEY, *names))
    values = tuple(float(index) for index in range(30))
    with pytest.raises(ValueError, match="family PCs"):
        prune_global_correlated_features({names[0]: values, names[1]: values}, contract)

    # Insufficient pair support is retained rather than guessed or imputed.
    result = prune_global_correlated_features(
        {names[0]: values[:20], "us_2y_log_level": values[:20]},
        build_feature_role_contract((TEMPORAL_KEY, names[0], "us_2y_log_level")),
    )
    assert result.representatives == (names[0], "us_2y_log_level")
    assert result.removed_features == ()


def test_global_pruning_does_not_accept_temporal_key() -> None:
    contract = build_feature_role_contract((TEMPORAL_KEY, "vix_log_level"))
    with pytest.raises(ValueError, match="temporal key"):
        prune_global_correlated_features(
            {TEMPORAL_KEY: tuple(float(index) for index in range(30))}, contract
        )
