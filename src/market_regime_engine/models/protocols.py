"""Model/backend ports used by training, evaluation, and causal inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from market_regime_engine.models.artifacts import GaussianHMMArtifact

ArrayF64 = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FitResult:
    artifact: GaussianHMMArtifact
    train_log_likelihood: float
    converged: bool
    iterations: int
    seed: int
    em_log_likelihood_history: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        history = self.em_log_likelihood_history
        if history and len(history) != self.iterations:
            raise ValueError("EM log-likelihood history length must equal completed iterations")
        if history and not all(np.isfinite(value) for value in history):
            raise ValueError("EM log-likelihood history must contain only finite values")


@dataclass(frozen=True, slots=True)
class FilterResult:
    filtered_probabilities: ArrayF64
    log_likelihood: float
    terminal_probabilities: tuple[float, ...]


class GaussianHMMAdapter(Protocol):
    def fit(self, train_rows: npt.ArrayLike, state_count: int, seed: int) -> FitResult: ...

    def extract(self) -> GaussianHMMArtifact: ...

    def reconstruct(self, artifact: GaussianHMMArtifact) -> None: ...

    def causal_filter(
        self,
        rows: npt.ArrayLike,
        initial_filtered_probabilities: tuple[float, ...] | None = None,
    ) -> FilterResult: ...


class PredictiveLikelihood(Protocol):
    def score_continuation(
        self,
        test_rows: npt.ArrayLike,
        terminal_train_alpha: tuple[float, ...],
    ) -> float: ...
