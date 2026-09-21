from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from market_regime_engine.feature_discovery.feature_roles import (
    TEMPORAL_KEY,
    FeatureRoleContract,
    build_feature_role_contract,
)
from market_regime_engine.feature_discovery.global_reduction import (
    prune_global_correlated_features,
)


def contract_for(*core_names: str) -> FeatureRoleContract:
    return build_feature_role_contract((TEMPORAL_KEY, *core_names))


def orthogonal_series() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.arange(30, dtype=np.float64) - 14.5
    alternating = np.where(np.arange(30) % 2 == 0, -1.0, 1.0)
    third = np.sin(np.arange(30, dtype=np.float64))
    orthonormal, _ = np.linalg.qr(np.column_stack((base, alternating, third)))
    return tuple(np.tile(orthonormal[:, index], 3) for index in range(3))  # type: ignore[return-value]


def reference_pair(
    left: tuple[float, ...], right: tuple[float, ...]
) -> tuple[float, int, tuple[float, ...], tuple[int, ...]]:
    left_array = np.asarray(left)
    right_array = np.asarray(right)

    def calculate(left_window: np.ndarray, right_window: np.ndarray) -> tuple[float, int]:
        mask = np.isfinite(left_window) & np.isfinite(right_window)
        return float(abs(np.corrcoef(left_window[mask], right_window[mask])[0, 1])), int(mask.sum())

    full = calculate(left_array, right_array)
    thirds = tuple(
        calculate(left_array[start : start + 30], right_array[start : start + 30])
        for start in (0, 30, 60)
    )
    return full[0], full[1], tuple(item[0] for item in thirds), tuple(item[1] for item in thirds)


def test_independent_reference_covers_positive_negative_and_just_below_boundaries() -> None:
    base = tuple(float(index) for index in range(90))
    inverse = tuple(-value for value in base)
    left, orthogonal, _ = orthogonal_series()
    almost = tuple(
        0.949999 * a + np.sqrt(1.0 - 0.949999**2) * b for a, b in zip(left, orthogonal, strict=True)
    )
    names = (
        "vix_log_level",
        "us_10y_log_level",
        "family_pc_vix_1",
        "family_pc_vix_2",
    )
    values = OrderedDict(
        (
            (names[0], base),
            (names[1], inverse),
            (names[2], base),
            (names[3], almost),
        )
    )
    result = prune_global_correlated_features(values, contract_for(names[0], names[1]))
    assert result.representatives == (names[0], names[3])
    assert set(result.removed_features) == {names[1], names[2]}
    for item in result.evidence:
        expected = reference_pair(values[item.leader], values[item.removed])
        assert item.full_absolute_pearson == pytest.approx(expected[0])
        assert item.full_support_count == expected[1]
        assert item.subwindow_absolute_pearsons == pytest.approx(expected[2])
        assert item.subwindow_support_counts == expected[3]


def test_direct_chain_does_not_remove_c_when_a_is_the_leader() -> None:
    a, orthogonal_b, orthogonal_c = orthogonal_series()
    leader_r = 0.96
    chain_r = 0.959
    b = tuple(
        leader_r * left + np.sqrt(1.0 - leader_r**2) * right
        for left, right in zip(a, orthogonal_b, strict=True)
    )
    d = tuple(
        leader_r * left - np.sqrt(1.0 - leader_r**2) * right
        for left, right in zip(a, orthogonal_b, strict=True)
    )
    c = tuple(
        chain_r * left + np.sqrt(1.0 - chain_r**2) * right
        for left, right in zip(b, orthogonal_c, strict=True)
    )
    names = ("vix_log_level", "us_10y_log_level", "family_pc_vix_1", "family_pc_vix_2")
    values: dict[str, tuple[float | None, ...]] = {
        names[0]: tuple(float(value) for value in a),
        names[1]: d,
        names[2]: b,
        names[3]: c,
    }
    result = prune_global_correlated_features(
        values,
        contract_for(names[0], names[1]),
    )
    assert result.representatives == (names[0], names[3])
    assert {(item.leader, item.removed) for item in result.evidence} == {
        (names[0], names[1]),
        (names[0], names[2]),
    }


def test_unstable_third_and_insufficient_support_never_remove() -> None:
    base = tuple(float(index) for index in range(90))
    unstable = tuple(
        value if index < 30 else float((index * 17) % 11) for index, value in enumerate(base)
    )
    short = tuple(value if index < 20 else None for index, value in enumerate(base))
    names = ("vix_log_level", "us_10y_log_level", "family_pc_vix_1")
    values: dict[str, tuple[float | None, ...]] = {
        names[0]: base,
        names[1]: unstable,
        names[2]: short,
    }
    result = prune_global_correlated_features(
        values,
        contract_for(names[0], names[1]),
    )
    assert result.removed_features == ()


def test_reversing_mapping_and_mutating_oos_rows_does_not_change_train_hash() -> None:
    base = tuple(float(index) for index in range(90))
    inverse = tuple(-value for value in base)
    names = ("vix_log_level", "us_10y_log_level")
    contract = contract_for(*names)
    train_rows = 60
    first = prune_global_correlated_features(
        OrderedDict(((names[0], base[:train_rows]), (names[1], inverse[:train_rows]))), contract
    )
    second = prune_global_correlated_features(
        OrderedDict(((names[1], inverse[:train_rows]), (names[0], base[:train_rows]))), contract
    )
    mutated_oos = tuple(value + 1.0e12 for value in inverse[train_rows:])
    assert mutated_oos
    assert first.result_hash == second.result_hash
