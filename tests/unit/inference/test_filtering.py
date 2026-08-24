from __future__ import annotations

from math import log, pi

import numpy as np
import pytest

from market_regime_engine.inference.filtering import causal_filter
from market_regime_engine.inference.predictive_likelihood import (
    continued_test_predictive_likelihood,
)
from market_regime_engine.models.artifacts import GaussianHMMArtifact


def equal_emission_artifact() -> GaussianHMMArtifact:
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=("x",),
        start_probabilities=(0.7, 0.3),
        transition_matrix=((0.9, 0.1), (0.2, 0.8)),
        means=((0.0,), (0.0,)),
        full_covariances=(((1.0,),), ((1.0,),)),
    )


def separated_artifact() -> GaussianHMMArtifact:
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=("x",),
        start_probabilities=(0.99, 0.01),
        transition_matrix=((0.99, 0.01), (0.01, 0.99)),
        means=((-5.0,), (5.0,)),
        full_covariances=(((1.0,),), ((1.0,),)),
    )


def test_stabilized_alpha_recursion_matches_hand_transition() -> None:
    result = causal_filter([[0.0], [0.0]], equal_emission_artifact())
    assert result.filtered_probabilities[0] == pytest.approx((0.7, 0.3))
    assert result.filtered_probabilities[1] == pytest.approx((0.69, 0.31))
    assert result.terminal_probabilities == pytest.approx((0.69, 0.31))
    assert np.allclose(result.filtered_probabilities.sum(axis=1), 1.0)


def test_supplied_terminal_train_alpha_causes_exactly_one_transition() -> None:
    result = causal_filter(
        [[0.0]],
        equal_emission_artifact(),
        initial_filtered_probabilities=(0.7, 0.3),
    )
    assert result.filtered_probabilities[0] == pytest.approx((0.69, 0.31))


def test_test_predictive_likelihood_sums_only_test_normalizers() -> None:
    scored = continued_test_predictive_likelihood(
        [[0.0], [0.0]],
        [[0.0], [0.0], [0.0]],
        equal_emission_artifact(),
    )
    expected_one = -0.5 * log(2.0 * pi)
    assert scored.test_observation_count == 3
    assert scored.test_log_likelihood == pytest.approx(3.0 * expected_one)
    assert scored.test_log_likelihood_per_observation == pytest.approx(expected_one)
    assert scored.train_terminal_probabilities == pytest.approx((0.69, 0.31))


def test_continuation_is_not_backend_style_test_restart() -> None:
    artifact = separated_artifact()
    continued = continued_test_predictive_likelihood(
        [[5.0], [5.0]],
        [[5.0]],
        artifact,
    )
    reset = causal_filter([[5.0]], artifact)
    assert continued.test_log_likelihood > reset.log_likelihood


def test_future_rows_cannot_change_earlier_filtered_probabilities() -> None:
    artifact = separated_artifact()
    prefix = causal_filter([[-5.0], [-4.5], [5.0]], artifact)
    extended = causal_filter([[-5.0], [-4.5], [5.0], [-5.0], [5.0]], artifact)
    assert extended.filtered_probabilities[:3] == pytest.approx(prefix.filtered_probabilities)


def test_invalid_empty_test_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        continued_test_predictive_likelihood(
            [[0.0]],
            np.empty((0, 1)),
            equal_emission_artifact(),
        )
