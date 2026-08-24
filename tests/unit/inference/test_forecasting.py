from __future__ import annotations

import numpy as np
import pytest

from market_regime_engine.inference.forecasting import transition_horizon_forecast
from market_regime_engine.models.artifacts import GaussianHMMArtifact


def artifact() -> GaussianHMMArtifact:
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=("x",),
        start_probabilities=(0.5, 0.5),
        transition_matrix=((0.9, 0.1), (0.2, 0.8)),
        means=((-1.0,), (1.0,)),
        full_covariances=(((1.0,),), ((1.0,),)),
    )


def test_horizon_zero_is_current_distribution() -> None:
    current = (0.25, 0.75)
    assert transition_horizon_forecast(current, artifact(), 0) == current


def test_positive_horizon_matches_matrix_power() -> None:
    current = np.asarray((0.25, 0.75))
    transition = np.asarray(artifact().transition_matrix)
    for horizon in (1, 2, 7):
        expected = current @ np.linalg.matrix_power(transition, horizon)
        actual = transition_horizon_forecast(tuple(current), artifact(), horizon)
        assert actual == pytest.approx(expected)
        assert sum(actual) == pytest.approx(1.0, abs=1e-10)
        assert all(value >= 0.0 for value in actual)


def test_one_step_is_one_retained_observation_transition() -> None:
    assert transition_horizon_forecast((1.0, 0.0), artifact(), 1) == pytest.approx((0.9, 0.1))


def test_invalid_horizon_and_probability_vector_fail_closed() -> None:
    for horizon in (-1, 1.5, True):
        with pytest.raises(ValueError, match="non-negative integer"):
            transition_horizon_forecast((0.5, 0.5), artifact(), horizon)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        transition_horizon_forecast((1.0,), artifact(), 1)
    with pytest.raises(ValueError, match="nonfinite"):
        transition_horizon_forecast((float("nan"), 1.0), artifact(), 1)
    with pytest.raises(ValueError, match="normalized"):
        transition_horizon_forecast((0.4, 0.4), artifact(), 1)
    with pytest.raises(ValueError, match="normalized"):
        transition_horizon_forecast((-0.1, 1.1), artifact(), 1)
