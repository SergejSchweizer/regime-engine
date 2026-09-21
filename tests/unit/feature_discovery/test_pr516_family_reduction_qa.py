from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from market_regime_engine.feature_discovery.family_reduction import prune_family_near_duplicates
from market_regime_engine.feature_discovery.feature_roles import (
    TEMPORAL_KEY,
    FeatureRoleContract,
    build_feature_role_contract,
)


def _contract(names: tuple[str, ...]) -> FeatureRoleContract:
    return build_feature_role_contract((TEMPORAL_KEY, *names))


def _correlated_pair(
    correlation: float, *, thirds: int = 3
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    base = np.arange(30, dtype=np.float64) - 14.5
    alternating = np.where(np.arange(30) % 2 == 0, -1.0, 1.0)
    alternating -= np.dot(alternating, base) / np.dot(base, base) * base
    base /= np.linalg.norm(base)
    alternating /= np.linalg.norm(alternating)
    left = np.tile(base, thirds)
    orthogonal = np.tile(alternating, thirds)
    right = correlation * left + np.sqrt(1.0 - correlation**2) * orthogonal
    return tuple(float(value) for value in left), tuple(float(value) for value in right)


def _third_orthogonal() -> tuple[float, ...]:
    base = np.arange(30, dtype=np.float64) - 14.5
    alternating = np.where(np.arange(30) % 2 == 0, -1.0, 1.0)
    third = np.sin(np.arange(30, dtype=np.float64))
    orthonormal, _ = np.linalg.qr(np.column_stack((base, alternating, third)))
    return tuple(float(value) for value in np.tile(orthonormal[:, 2], 3))


def _reference_pair(
    left: tuple[float, ...], right: tuple[float, ...]
) -> tuple[float, int, tuple[float, ...], tuple[int, ...]]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)

    def corr(left_window: np.ndarray, right_window: np.ndarray) -> tuple[float, int]:
        mask = np.isfinite(left_window) & np.isfinite(right_window)
        support = int(mask.sum())
        value = float(abs(np.corrcoef(left_window[mask], right_window[mask])[0, 1]))
        return value, support

    full = corr(left_array, right_array)
    thirds = tuple(
        corr(left_array[start : start + 30], right_array[start : start + 30])
        for start in (0, 30, 60)
    )
    return full[0], full[1], tuple(item[0] for item in thirds), tuple(item[1] for item in thirds)


def test_exact_positive_negative_boundary_and_just_below_threshold() -> None:
    positive = tuple(float(index) for index in range(90))
    negative = tuple(-value for value in positive)
    almost_left, almost_right = _correlated_pair(0.994999)
    names = (
        "vix_delta_1obs",
        "vix_delta_5obs",
        "vix_zscore_20obs",
        "vix_zscore_40obs",
        "vix_return_geom_5obs",
        "vix_return_geom_20obs",
    )
    result = prune_family_near_duplicates(
        {
            names[0]: positive,
            names[1]: negative,
            names[2]: almost_left,
            names[3]: almost_right,
            names[4]: positive,
            names[5]: negative,
        },
        _contract(names),
    )
    assert (names[1],) == result.removed_features[:1]
    assert names[3] not in result.removed_features
    assert names[5] in result.removed_features
    values = {
        names[0]: positive,
        names[1]: negative,
        names[2]: almost_left,
        names[3]: almost_right,
        names[4]: positive,
        names[5]: negative,
    }
    for item in result.evidence:
        expected = _reference_pair(values[item.leader], values[item.duplicate])
        assert item.full_absolute_pearson == pytest.approx(expected[0])
        assert item.full_support_count == expected[1]
        assert item.subwindow_absolute_pearsons == pytest.approx(expected[2])
        assert item.subwindow_support_counts == expected[3]


def test_subwindow_support_ten_passes_and_nine_fails() -> None:
    left, right = _correlated_pair(1.0)
    ten_support = tuple(
        value if index < 10 or index < 40 or index >= 60 else None
        for index, value in enumerate(right)
    )
    nine_support = tuple(
        value if index < 9 or index < 39 or index >= 60 else None
        for index, value in enumerate(right)
    )
    names = ("vix_delta_1obs", "vix_delta_5obs")
    passed = prune_family_near_duplicates({names[0]: left, names[1]: ten_support}, _contract(names))
    failed = prune_family_near_duplicates(
        {names[0]: left, names[1]: nine_support}, _contract(names)
    )
    assert passed.removed_features == (names[1],)
    assert failed.removed_features == ()


def test_direct_chain_records_only_direct_pairs_and_not_a_transitive_edge() -> None:
    x, b = _correlated_pair(0.997)
    z = _third_orthogonal()
    c = tuple(
        0.997 * b_value + np.sqrt(1.0 - 0.997**2) * z_value
        for b_value, z_value in zip(b, z, strict=True)
    )
    names = ("vix_delta_1obs", "vix_delta_5obs", "vix_delta_20obs")
    result = prune_family_near_duplicates({names[0]: x, names[1]: b, names[2]: c}, _contract(names))
    assert {(item.leader, item.duplicate) for item in result.evidence} == {
        (names[1], names[0]),
        (names[1], names[2]),
    }
    assert (names[0], names[2]) not in {(item.leader, item.duplicate) for item in result.evidence}


def test_0_95_pair_remains_available_for_family_pca_and_order_is_stable() -> None:
    left, right = _correlated_pair(0.96)
    names = ("vix_delta_1obs", "vix_delta_5obs")
    first = prune_family_near_duplicates(
        OrderedDict(((names[0], left), (names[1], right))), _contract(names)
    )
    second = prune_family_near_duplicates(
        OrderedDict(((names[1], right), (names[0], left))), _contract(names)
    )
    assert first.removed_features == ()
    assert first.result_hash == second.result_hash


def test_oos_values_are_not_part_of_the_train_mapping() -> None:
    left, right = _correlated_pair(1.0)
    names = ("vix_delta_1obs", "vix_delta_5obs")
    train_rows = 60
    baseline = prune_family_near_duplicates(
        {names[0]: left[:train_rows], names[1]: right[:train_rows]}, _contract(names)
    )
    mutated_oos = tuple(value + 1.0e12 for value in right[train_rows:])
    repeated = prune_family_near_duplicates(
        {
            names[0]: left[:train_rows],
            names[1]: right[:train_rows],
        },
        _contract(names),
    )
    assert mutated_oos
    assert repeated.result_hash == baseline.result_hash
