"""Pinned Gaussian-HMM diagnostic formulas and hard TRAIN occupancy gates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from math import isfinite, log

import numpy as np
import numpy.typing as npt

from market_regime_engine.models.artifacts import GaussianHMMArtifact

MINIMUM_TRAIN_HARD_OCCUPANCY = 0.03
MINIMUM_TRAIN_SOFT_OCCUPANCY = 0.05
LOW_CONFIDENCE_THRESHOLD = 0.60
COVARIANCE_ASYMMETRY_TOLERANCE = 1e-10
MINIMUM_COVARIANCE_DIAGONAL_VARIANCE = 1e-12


@dataclass(frozen=True, slots=True)
class InformationCriteria:
    parameter_count: int
    aic: float
    bic: float


@dataclass(frozen=True, slots=True)
class OccupancyDiagnostics:
    hard: tuple[float, ...]
    soft: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class UncertaintyDiagnostics:
    confidence: tuple[float, ...]
    entropy: tuple[float, ...]
    low_confidence: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class CovarianceDiagnostics:
    maximum_absolute_asymmetry: tuple[float, ...]
    minimum_diagonal_variance: tuple[float, ...]


def gaussian_hmm_parameter_count(state_count: int, feature_dimension: int) -> int:
    if state_count < 1 or feature_dimension < 1:
        raise ValueError("state count and feature dimension must be positive")
    return (
        (state_count - 1)
        + state_count * (state_count - 1)
        + state_count * feature_dimension
        + state_count * feature_dimension * (feature_dimension + 1) // 2
    )


def information_criteria(
    train_log_likelihood: float,
    train_observation_count: int,
    state_count: int,
    feature_dimension: int,
) -> InformationCriteria:
    if not isfinite(train_log_likelihood):
        raise ValueError("TRAIN log likelihood must be finite")
    if train_observation_count < 1:
        raise ValueError("TRAIN observation count must be positive")
    parameter_count = gaussian_hmm_parameter_count(state_count, feature_dimension)
    aic = 2.0 * parameter_count - 2.0 * train_log_likelihood
    bic = parameter_count * log(train_observation_count) - 2.0 * train_log_likelihood
    return InformationCriteria(parameter_count, aic, bic)


def _probability_rows(filtered_probabilities: npt.ArrayLike) -> np.ndarray:
    values = np.asarray(filtered_probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("filtered probabilities must be a non-empty 2D matrix")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("filtered probabilities must be finite and nonnegative")
    sums = values.sum(axis=1)
    if not np.all(np.isclose(sums, 1.0, rtol=0.0, atol=1e-10)):
        raise ValueError("every filtered-probability row must sum to one within 1e-10")
    return values


def occupancy(filtered_probabilities: npt.ArrayLike) -> OccupancyDiagnostics:
    values = _probability_rows(filtered_probabilities)
    hard_states = np.argmax(values, axis=1)
    hard = tuple(float(np.mean(hard_states == state)) for state in range(values.shape[1]))
    soft = tuple(float(value) for value in values.mean(axis=0))
    return OccupancyDiagnostics(hard=hard, soft=soft)


def validate_train_occupancy(filtered_probabilities: npt.ArrayLike) -> OccupancyDiagnostics:
    result = occupancy(filtered_probabilities)
    if any(value < MINIMUM_TRAIN_HARD_OCCUPANCY for value in result.hard):
        raise ValueError("TRAIN hard occupancy is below 0.03")
    if any(value < MINIMUM_TRAIN_SOFT_OCCUPANCY for value in result.soft):
        raise ValueError("TRAIN soft occupancy is below 0.05")
    return result


def uncertainty(filtered_probabilities: npt.ArrayLike) -> UncertaintyDiagnostics:
    values = _probability_rows(filtered_probabilities)
    confidence_values = np.max(values, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(values > 0.0, values * np.log(values), 0.0)
    entropy_values = -terms.sum(axis=1)
    return UncertaintyDiagnostics(
        confidence=tuple(float(value) for value in confidence_values),
        entropy=tuple(float(value) for value in entropy_values),
        low_confidence=tuple(bool(value < LOW_CONFIDENCE_THRESHOLD) for value in confidence_values),
    )


def dominant_state_durations(filtered_probabilities: npt.ArrayLike) -> tuple[int, ...]:
    values = _probability_rows(filtered_probabilities)
    states = np.argmax(values, axis=1)
    durations: list[int] = []
    current = int(states[0])
    length = 1
    for raw_state in states[1:]:
        state = int(raw_state)
        if state == current:
            length += 1
        else:
            durations.append(length)
            current = state
            length = 1
    durations.append(length)
    return tuple(durations)


def switches_per_year(
    timestamps: tuple[datetime, ...],
    filtered_probabilities: npt.ArrayLike,
) -> float | None:
    values = _probability_rows(filtered_probabilities)
    if len(timestamps) != values.shape[0]:
        raise ValueError("timestamp count must match filtered observations")
    if any(timestamp.tzinfo is None for timestamp in timestamps):
        raise ValueError("switch timestamps must be timezone-aware")
    if any(left >= right for left, right in pairwise(timestamps)):
        raise ValueError("switch timestamps must be strictly increasing")
    elapsed_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86_400.0
    if elapsed_days == 0.0:
        return None
    states = np.argmax(values, axis=1)
    switch_count = int(np.sum(states[1:] != states[:-1]))
    return switch_count / elapsed_days * 365.2425


def validate_full_covariances(artifact: GaussianHMMArtifact) -> CovarianceDiagnostics:
    asymmetries: list[float] = []
    minimum_variances: list[float] = []
    dimension = artifact.feature_dimension
    for covariance_rows in artifact.full_covariances:
        covariance = np.asarray(covariance_rows, dtype=np.float64)
        if covariance.shape != (dimension, dimension) or not np.all(np.isfinite(covariance)):
            raise ValueError("full covariance must have exact finite d x d shape")
        asymmetry = float(np.max(np.abs(covariance - covariance.T)))
        if asymmetry > COVARIANCE_ASYMMETRY_TOLERANCE:
            raise ValueError("full covariance asymmetry exceeds 1e-10")
        diagonal_minimum = float(np.min(np.diag(covariance)))
        if diagonal_minimum < MINIMUM_COVARIANCE_DIAGONAL_VARIANCE:
            raise ValueError("full covariance diagonal variance is below 1e-12")
        symmetric = (covariance + covariance.T) / 2.0
        try:
            np.linalg.cholesky(symmetric)
        except np.linalg.LinAlgError as exc:
            raise ValueError("full covariance must pass Cholesky without jitter") from exc
        asymmetries.append(asymmetry)
        minimum_variances.append(diagonal_minimum)
    return CovarianceDiagnostics(tuple(asymmetries), tuple(minimum_variances))
