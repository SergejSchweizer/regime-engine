"""TRAIN-continuation out-of-sample predictive likelihood."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import numpy.typing as npt

from market_regime_engine.inference.filtering import causal_filter
from market_regime_engine.models.artifacts import GaussianHMMArtifact


@dataclass(frozen=True, slots=True)
class ContinuedPredictiveLikelihood:
    train_terminal_probabilities: tuple[float, ...]
    test_terminal_probabilities: tuple[float, ...]
    test_log_likelihood: float
    test_observation_count: int
    test_log_likelihood_per_observation: float

    def __post_init__(self) -> None:
        if self.test_observation_count < 1:
            raise ValueError("TEST predictive likelihood requires at least one retained observation")
        if not isfinite(self.test_log_likelihood):
            raise ValueError("TEST predictive log likelihood must be finite")
        if not isfinite(self.test_log_likelihood_per_observation):
            raise ValueError("per-observation TEST predictive log likelihood must be finite")


def continued_test_predictive_likelihood(
    train_rows: npt.ArrayLike,
    test_rows: npt.ArrayLike,
    artifact: GaussianHMMArtifact,
) -> ContinuedPredictiveLikelihood:
    """Score TEST by continuing the causal filter from terminal TRAIN alpha only."""

    train = causal_filter(train_rows, artifact)
    test_array = np.asarray(test_rows, dtype=np.float64)
    if test_array.ndim != 2 or test_array.shape[0] < 1:
        raise ValueError("TEST rows must contain at least one retained observation")
    test = causal_filter(
        test_array,
        artifact,
        initial_filtered_probabilities=train.terminal_probabilities,
    )
    count = int(test_array.shape[0])
    per_observation = test.log_likelihood / count
    return ContinuedPredictiveLikelihood(
        train_terminal_probabilities=train.terminal_probabilities,
        test_terminal_probabilities=test.terminal_probabilities,
        test_log_likelihood=test.log_likelihood,
        test_observation_count=count,
        test_log_likelihood_per_observation=per_observation,
    )
