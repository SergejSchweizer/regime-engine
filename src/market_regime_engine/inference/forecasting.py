"""Retained-observation transition-horizon forecasts for Gaussian HMM states."""

from __future__ import annotations

from math import isfinite

import numpy as np

from market_regime_engine.models.artifacts import GaussianHMMArtifact

_PROBABILITY_TOLERANCE = 1e-10


def transition_horizon_forecast(
    current_filtered_probabilities: tuple[float, ...],
    artifact: GaussianHMMArtifact,
    horizon: int,
) -> tuple[float, ...]:
    """Forecast state probabilities ``horizon`` retained observations ahead.

    Horizon zero is the current filtered distribution. A positive horizon applies
    the HMM transition matrix once per future retained model observation; it has
    no calendar-day interpretation.
    """

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
        raise ValueError("forecast horizon must be a non-negative integer")
    probabilities = np.asarray(current_filtered_probabilities, dtype=np.float64)
    if probabilities.shape != (artifact.state_count,) or not np.all(np.isfinite(probabilities)):
        raise ValueError("current filtered probabilities have invalid shape or nonfinite values")
    if np.any(probabilities < 0.0) or not np.isclose(
        float(probabilities.sum()),
        1.0,
        rtol=0.0,
        atol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError("current filtered probabilities must be normalized within 1e-10")

    if horizon == 0:
        result = probabilities.copy()
    else:
        transition = np.asarray(artifact.transition_matrix, dtype=np.float64)
        result = probabilities @ np.linalg.matrix_power(transition, horizon)
    if not np.all(np.isfinite(result)) or np.any(result < -_PROBABILITY_TOLERANCE):
        raise ValueError("forecast probabilities must be finite and nonnegative")
    total = float(result.sum())
    if not isfinite(total) or not np.isclose(
        total,
        1.0,
        rtol=0.0,
        atol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError("forecast probabilities must remain normalized within 1e-10")
    return tuple(float(value) for value in result)
