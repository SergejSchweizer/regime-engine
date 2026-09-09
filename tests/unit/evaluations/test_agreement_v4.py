from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import log

import pytest

from market_regime_engine.evaluations.agreement_v4 import compute_soft_regime_nmi


def times(count: int = 4) -> tuple[datetime, ...]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return tuple(start + timedelta(days=index) for index in range(count))


def test_one_hot_perfect_relabeling_is_soft_nmi_one() -> None:
    timestamps = times()
    result = compute_soft_regime_nmi(
        timestamps,
        ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, 1.0)),
        timestamps,
        ((0.0, 1.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
    )

    assert result.soft_regime_nmi == pytest.approx(1.0, abs=1e-12)
    assert result.mutual_information == pytest.approx(result.left_entropy, abs=1e-12)
    assert result.shared_timestamp_count == 4
    assert len(result.agreement_hash) == 64


def test_independent_one_hot_sequences_have_zero_soft_nmi() -> None:
    timestamps = times()
    result = compute_soft_regime_nmi(
        timestamps,
        ((1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0)),
        timestamps,
        ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, 1.0)),
    )

    assert result.soft_regime_nmi == pytest.approx(0.0, abs=1e-12)
    for row in result.joint_matrix:
        assert row == pytest.approx((0.25, 0.25))


def test_uncertain_soft_joint_matches_independent_reference_calculation() -> None:
    timestamps = times(2)
    result = compute_soft_regime_nmi(
        timestamps,
        ((0.8, 0.2), (0.2, 0.8)),
        timestamps,
        ((0.8, 0.2), (0.2, 0.8)),
    )

    expected_joint = ((0.34, 0.16), (0.16, 0.34))
    expected_mi = sum(mass * log(mass / 0.25) for row in expected_joint for mass in row)
    for actual_row, expected_row in zip(result.joint_matrix, expected_joint, strict=True):
        assert actual_row == pytest.approx(expected_row)
    assert result.mutual_information == pytest.approx(expected_mi)
    assert result.soft_regime_nmi == pytest.approx(expected_mi / log(2.0))


def test_unequal_state_spaces_and_timestamp_order_are_supported() -> None:
    timestamp_values = times(3)
    result = compute_soft_regime_nmi(
        timestamp_values[::-1],
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        timestamp_values[1:],
        ((0.0, 1.0), (1.0, 0.0)),
    )

    assert (result.left_state_count, result.right_state_count) == (3, 2)
    assert result.shared_timestamps == timestamp_values[1:]
    assert result.shared_timestamp_count == 2


@pytest.mark.parametrize(
    ("left_probabilities", "right_probabilities", "message"),
    [
        (((1.0, -1.0),), ((1.0, 0.0),), "non-negative"),
        (((0.5, 0.5000000002),), ((1.0, 0.0),), "normalized"),
    ],
)
def test_probability_validation_is_explicit(
    left_probabilities: tuple[tuple[float, ...], ...],
    right_probabilities: tuple[tuple[float, ...], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_soft_regime_nmi(times(1), left_probabilities, times(1), right_probabilities)


def test_zero_shared_support_and_degenerate_entropy_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="zero shared timestamp"):
        compute_soft_regime_nmi(
            times(1), ((1.0, 0.0),), (datetime(2027, 1, 1, tzinfo=UTC),), ((1.0, 0.0),)
        )
    with pytest.raises(ValueError, match="non-degenerate marginal entropy"):
        compute_soft_regime_nmi(
            times(2), ((1.0, 0.0), (1.0, 0.0)), times(2), ((0.5, 0.5), (0.5, 0.5))
        )
