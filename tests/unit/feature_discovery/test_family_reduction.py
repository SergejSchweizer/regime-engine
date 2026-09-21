import pytest

from market_regime_engine.feature_discovery.family_reduction import (
    prune_family_near_duplicates,
)
from market_regime_engine.feature_discovery.feature_roles import (
    TEMPORAL_KEY,
    build_feature_role_contract,
)


def _contract(*names: str):
    return build_feature_role_contract((TEMPORAL_KEY, *names))


def test_pruning_keeps_earliest_stable_leader_only_within_family() -> None:
    leader = tuple(float(index) for index in range(30))
    inverse = tuple(-value for value in leader)
    unstable = tuple(
        float(index) if index < 10 or index >= 20 else float(index % 2) for index in range(30)
    )
    other_family = tuple(float(index * 2) for index in range(30))
    names = (
        "vix_delta_1obs",
        "vix_delta_5obs",
        "vix_zscore_20obs",
        "vix9d_delta_1obs",
    )
    result = prune_family_near_duplicates(
        {
            names[0]: leader,
            names[1]: inverse,
            names[2]: unstable,
            names[3]: other_family,
        },
        _contract(*names),
    )

    assert result.retained_features == (names[0], names[2], names[3])
    assert result.removed_features == (names[1],)
    assert len(result.evidence) == 1
    assert result.evidence[0].full_absolute_pearson == pytest.approx(1.0)
    assert result.evidence[0].subwindow_absolute_pearsons == pytest.approx((1.0, 1.0, 1.0))


def test_pruning_is_redundancy_only_and_does_not_fill_or_guess_unsupported_pairs() -> None:
    values = tuple(float(index) for index in range(30))
    missing = tuple(None if index % 2 else float(index) for index in range(30))
    names = ("usd_broad_delta_1obs", "usd_broad_delta_5obs")
    result = prune_family_near_duplicates(
        {names[0]: values, names[1]: missing},
        _contract(*names),
    )
    assert result.retained_features == names
    assert result.removed_features == ()
    assert result.evidence == ()


def test_pruning_rejects_core_and_temporal_inputs() -> None:
    with pytest.raises(ValueError, match="transformations only"):
        prune_family_near_duplicates(
            {"vix_log_level": tuple(float(index) for index in range(30))},
            _contract("vix_log_level"),
        )
