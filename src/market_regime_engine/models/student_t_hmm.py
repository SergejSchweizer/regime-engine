"""Deterministic full-covariance multivariate Student-t HMM estimator."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import numpy.typing as npt
from hmmlearn.hmm import GaussianHMM  # type: ignore[import-untyped]
from scipy.optimize import brentq  # type: ignore[import-untyped]
from scipy.special import digamma, logsumexp  # type: ignore[import-untyped]

from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.gaussian_hmm import (
    _validated_em_history,
    forward_filter,
    gaussian_log_emissions,
)
from market_regime_engine.models.protocols import ArrayF64, FilterResult, FitResult


@dataclass(frozen=True, slots=True)
class StudentTHMMSettings:
    n_iter: int = 200
    tol: float = 1e-4
    min_covar: float = 1e-6
    initial_nu: float = 10.0
    minimum_nu: float = 2.05
    maximum_nu: float = 200.0

    def __post_init__(self) -> None:
        if self.n_iter < 1 or self.tol <= 0.0 or self.min_covar <= 0.0:
            raise ValueError("Student-t iteration settings must be positive")
        if not 2.0 < self.minimum_nu < self.initial_nu < self.maximum_nu:
            raise ValueError("Student-t degrees-of-freedom bounds are inconsistent")


def _rows(values: npt.ArrayLike, dimension: int) -> ArrayF64:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 2 or result.shape[1] != dimension:
        raise ValueError("Student-t HMM rows must have shape (n>=2, exact feature dimension)")
    if not np.all(np.isfinite(result)):
        raise ValueError("Student-t HMM rows must be finite")
    return result


def _artifact(
    feature_order: tuple[str, ...],
    start: ArrayF64,
    transition: ArrayF64,
    means: ArrayF64,
    scales: ArrayF64,
    nu: ArrayF64,
) -> GaussianHMMArtifact:
    return GaussianHMMArtifact(
        state_count=len(start),
        feature_order=feature_order,
        start_probabilities=tuple(float(value) for value in start),
        transition_matrix=tuple(tuple(float(value) for value in row) for row in transition),
        means=tuple(tuple(float(value) for value in row) for row in means),
        full_covariances=tuple(
            tuple(tuple(float(value) for value in row) for row in matrix) for matrix in scales
        ),
        model_family="student_t_hmm",
        degrees_of_freedom=tuple(float(value) for value in nu),
    )


def _expectation(
    values: ArrayF64,
    artifact: GaussianHMMArtifact,
) -> tuple[ArrayF64, ArrayF64, float, ArrayF64]:
    emissions = gaussian_log_emissions(values, artifact)
    # Exact zero start/transition probabilities map to -inf in log-space; they
    # are valid model parameters rather than a numerical failure.
    with np.errstate(divide="ignore"):
        log_start = np.log(np.asarray(artifact.start_probabilities))
        log_transition = np.log(np.asarray(artifact.transition_matrix))
    count, states = emissions.shape
    alpha = np.empty_like(emissions)
    alpha[0] = log_start + emissions[0]
    for index in range(1, count):
        alpha[index] = emissions[index] + logsumexp(
            alpha[index - 1][:, None] + log_transition, axis=0
        )
    likelihood = float(logsumexp(alpha[-1]))
    beta = np.zeros_like(emissions)
    for index in range(count - 2, -1, -1):
        beta[index] = logsumexp(
            log_transition + emissions[index + 1][None, :] + beta[index + 1][None, :],
            axis=1,
        )
    gamma = np.exp(alpha + beta - likelihood)
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi_sum = np.zeros((states, states), dtype=np.float64)
    for index in range(count - 1):
        log_xi = (
            alpha[index][:, None]
            + log_transition
            + emissions[index + 1][None, :]
            + beta[index + 1][None, :]
            - likelihood
        )
        xi_sum += np.exp(log_xi)
    return gamma, xi_sum, likelihood, emissions


def _update_nu(
    expected_log_precision_minus_precision: float,
    settings: StudentTHMMSettings,
) -> float:
    def objective(value: float) -> float:
        return float(
            np.log(value / 2.0)
            - digamma(value / 2.0)
            + 1.0
            + expected_log_precision_minus_precision
        )

    lower = settings.minimum_nu
    upper = settings.maximum_nu
    lower_value = float(objective(lower))
    upper_value = float(objective(upper))
    if lower_value * upper_value > 0.0:
        return lower if abs(lower_value) < abs(upper_value) else upper
    return float(brentq(objective, lower, upper, xtol=1e-8, rtol=1e-10))


class StudentTHMMAdapter:
    """Baum-Welch Student-t HMM with one state-specific t emission and nu."""

    def __init__(
        self,
        feature_order: tuple[str, ...],
        settings: StudentTHMMSettings | None = None,
    ) -> None:
        if not feature_order or len(set(feature_order)) != len(feature_order):
            raise ValueError("feature_order must be non-empty and duplicate-free")
        self.feature_order = feature_order
        self.settings = settings or StudentTHMMSettings()
        self._artifact: GaussianHMMArtifact | None = None

    def fit(self, train_rows: npt.ArrayLike, state_count: int, seed: int) -> FitResult:
        if state_count not in (2, 3, 4, 5):
            raise ValueError("Student-t HMM state_count must be K=2,3,4,5")
        values = _rows(train_rows, len(self.feature_order))
        initializer = GaussianHMM(
            n_components=state_count,
            covariance_type="full",
            min_covar=self.settings.min_covar,
            random_state=seed,
            n_iter=100,
            tol=self.settings.tol,
            implementation="log",
        ).fit(values)
        start = np.asarray(initializer.startprob_, dtype=np.float64)
        transition = np.asarray(initializer.transmat_, dtype=np.float64)
        means = np.asarray(initializer.means_, dtype=np.float64)
        scales = np.asarray(initializer.covars_, dtype=np.float64)
        nu = np.full(state_count, self.settings.initial_nu, dtype=np.float64)
        previous = -np.inf
        converged = False
        iterations = 0
        dimension = values.shape[1]
        identity = np.eye(dimension)
        history: list[float] = []
        for iteration in range(1, self.settings.n_iter + 1):
            iterations = iteration
            current = _artifact(self.feature_order, start, transition, means, scales, nu)
            gamma, xi_sum, likelihood, _ = _expectation(values, current)
            history.append(float(likelihood))
            start = np.maximum(gamma[0], 1e-12)
            start /= start.sum()
            transition = np.maximum(xi_sum, 1e-12)
            transition /= transition.sum(axis=1, keepdims=True)
            for state in range(state_count):
                centered = values - means[state]
                solved = np.linalg.solve(scales[state], centered.T).T
                delta = np.einsum("ij,ij->i", centered, solved)
                precision = (nu[state] + dimension) / (nu[state] + delta)
                log_precision = digamma((nu[state] + dimension) / 2.0) - np.log(
                    (nu[state] + delta) / 2.0
                )
                weighted = gamma[:, state] * precision
                means[state] = (weighted[:, None] * values).sum(axis=0) / weighted.sum()
                centered = values - means[state]
                scales[state] = (
                    np.einsum("n,ni,nj->ij", weighted, centered, centered) / gamma[:, state].sum()
                    + self.settings.min_covar * identity
                )
                expectation = float(
                    np.sum(gamma[:, state] * (log_precision - precision)) / gamma[:, state].sum()
                )
                nu[state] = _update_nu(expectation, self.settings)
            if isfinite(previous) and abs(likelihood - previous) <= self.settings.tol * (
                1.0 + abs(previous)
            ):
                converged = True
                break
            previous = likelihood
        result = _artifact(self.feature_order, start, transition, means, scales, nu)
        final_likelihood = forward_filter(values, result).log_likelihood
        em_history = _validated_em_history(history, iterations)
        self._artifact = result
        return FitResult(
            artifact=result,
            train_log_likelihood=final_likelihood,
            converged=converged,
            iterations=iterations,
            seed=seed,
            em_log_likelihood_history=em_history,
        )

    def extract(self) -> GaussianHMMArtifact:
        if self._artifact is None:
            raise ValueError("Student-t model has not been fitted or reconstructed")
        return self._artifact

    def reconstruct(self, artifact: GaussianHMMArtifact) -> None:
        if artifact.model_family != "student_t_hmm":
            raise ValueError("Student-t adapter requires a Student-t artifact")
        if artifact.feature_order != self.feature_order:
            raise ValueError("artifact feature order does not match adapter feature order")
        self._artifact = artifact

    def causal_filter(
        self,
        rows: npt.ArrayLike,
        initial_filtered_probabilities: tuple[float, ...] | None = None,
    ) -> FilterResult:
        return forward_filter(
            rows,
            self.extract(),
            terminal_train_alpha=initial_filtered_probabilities,
        )

    def score_continuation(
        self,
        test_rows: npt.ArrayLike,
        terminal_train_alpha: tuple[float, ...],
    ) -> float:
        return self.causal_filter(test_rows, terminal_train_alpha).log_likelihood
