from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import log

import numpy as np
import pytest

from market_regime_engine.evaluation.diagnostics import (
    LOW_CONFIDENCE_THRESHOLD,
    MINIMUM_TRAIN_HARD_OCCUPANCY,
    MINIMUM_TRAIN_SOFT_OCCUPANCY,
    dominant_state_durations,
    gaussian_hmm_parameter_count,
    gmm_hmm_parameter_count,
    information_criteria,
    occupancy,
    switches_per_year,
    uncertainty,
    validate_full_covariances,
    validate_train_occupancy,
)
from market_regime_engine.models.artifacts import GaussianHMMArtifact


def artifact() -> GaussianHMMArtifact:
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=("a", "b"),
        start_probabilities=(0.6, 0.4),
        transition_matrix=((0.9, 0.1), (0.2, 0.8)),
        means=((-1.0, 0.0), (1.0, 0.0)),
        full_covariances=(
            ((1.0, 0.2), (0.2, 2.0)),
            ((2.0, -0.1), (-0.1, 1.0)),
        ),
    )


def test_parameter_count_aic_and_bic_match_exact_formula() -> None:
    count = gaussian_hmm_parameter_count(3, 4)
    expected = (3 - 1) + 3 * (3 - 1) + 3 * 4 + 3 * 4 * 5 // 2
    assert count == expected
    result = information_criteria(-123.5, 600, 3, 4)
    assert result.parameter_count == expected
    assert result.aic == pytest.approx(2 * expected - 2 * -123.5)
    assert result.bic == pytest.approx(expected * log(600) - 2 * -123.5)


def test_gmm_hmm_information_criteria_include_mixture_weights_and_emissions() -> None:
    count = gmm_hmm_parameter_count(2, 4, 2)
    expected = (2 - 1) + 2 * (2 - 1) + 2 * (2 - 1) + 2 * 2 * 4 + 2 * 2 * 4 * 5 // 2
    assert count == expected
    result = information_criteria(-123.5, 600, 2, 4, mixture_count=2)
    assert result.parameter_count == expected


def test_information_criteria_fail_closed() -> None:
    for state_count, dimension in ((0, 1), (1, 0)):
        with pytest.raises(ValueError, match="positive"):
            gaussian_hmm_parameter_count(state_count, dimension)
    with pytest.raises(ValueError, match="finite"):
        information_criteria(float("nan"), 600, 2, 1)
    with pytest.raises(ValueError, match="positive"):
        information_criteria(-1.0, 0, 2, 1)


def test_train_hard_and_soft_occupancy_are_distinct() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.1],
            [0.6, 0.4],
            [0.4, 0.6],
            [0.2, 0.8],
        ]
    )
    result = occupancy(probabilities)
    assert result.hard == pytest.approx((0.5, 0.5))
    assert result.soft == pytest.approx((0.525, 0.475))
    assert MINIMUM_TRAIN_HARD_OCCUPANCY == 0.03
    assert MINIMUM_TRAIN_SOFT_OCCUPANCY == 0.05
    assert validate_train_occupancy(probabilities) == result


def test_train_occupancy_gates_fail_separately() -> None:
    hard_fail = np.tile([0.51, 0.49], (100, 1))
    with pytest.raises(ValueError, match="hard occupancy"):
        validate_train_occupancy(hard_fail)

    soft_fail = np.vstack((np.tile([0.999, 0.001], (97, 1)), np.tile([0.49, 0.51], (3, 1))))
    with pytest.raises(ValueError, match="soft occupancy"):
        validate_train_occupancy(soft_fail)


def test_entropy_confidence_and_low_confidence_use_natural_log() -> None:
    probabilities = np.asarray([[0.5, 0.5], [0.8, 0.2], [1.0, 0.0]])
    result = uncertainty(probabilities)
    assert LOW_CONFIDENCE_THRESHOLD == 0.60
    assert result.confidence == pytest.approx((0.5, 0.8, 1.0))
    assert result.entropy[0] == pytest.approx(log(2.0))
    assert result.entropy[1] == pytest.approx(-(0.8 * log(0.8) + 0.2 * log(0.2)))
    assert result.entropy[2] == pytest.approx(0.0)
    assert result.low_confidence == (True, False, False)


def test_retained_observation_durations_and_actual_time_switches_per_year() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.7, 0.3],
        ]
    )
    assert dominant_state_durations(probabilities) == (2, 2, 1)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    timestamps = tuple(start + timedelta(days=value) for value in (0, 1, 10, 11, 20))
    expected = 2 / 20 * 365.2425
    assert switches_per_year(timestamps, probabilities) == pytest.approx(expected)


def test_switch_rate_zero_elapsed_and_invalid_time_inputs() -> None:
    probabilities = np.asarray([[0.9, 0.1]])
    timestamp = datetime(2025, 1, 1, tzinfo=UTC)
    assert switches_per_year((timestamp,), probabilities) is None
    with pytest.raises(ValueError, match="count"):
        switches_per_year((timestamp, timestamp + timedelta(days=1)), probabilities)
    with pytest.raises(ValueError, match="timezone-aware"):
        switches_per_year((datetime(2025, 1, 1),), probabilities)

    two = np.asarray([[0.9, 0.1], [0.1, 0.9]])
    with pytest.raises(ValueError, match="strictly increasing"):
        switches_per_year((timestamp, timestamp), two)


def test_probability_rows_validation_is_shared_by_diagnostics() -> None:
    for invalid in (
        np.asarray([]),
        np.asarray([[0.5, float("nan")]]),
        np.asarray([[-0.1, 1.1]]),
        np.asarray([[0.4, 0.4]]),
    ):
        with pytest.raises(ValueError):
            occupancy(invalid)


def test_full_covariance_validation_records_exact_diagnostics() -> None:
    result = validate_full_covariances(artifact())
    assert result.maximum_absolute_asymmetry == pytest.approx((0.0, 0.0))
    assert result.minimum_diagonal_variance == pytest.approx((1.0, 1.0))


def test_cholesky_failure_is_not_jittered() -> None:
    valid = artifact()
    object.__setattr__(
        valid,
        "full_covariances",
        (
            ((1.0, 2.0), (2.0, 1.0)),
            ((2.0, -0.1), (-0.1, 1.0)),
        ),
    )
    with pytest.raises(ValueError, match="Cholesky"):
        validate_full_covariances(valid)
