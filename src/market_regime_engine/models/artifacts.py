"""Backend-neutral Gaussian HMM parameter artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite

_PROB_TOL = 1e-10
_ASYMMETRY_TOL = 1e-10
_MIN_VARIANCE = 1e-12


def _finite_vector(values: tuple[float, ...], name: str) -> None:
    if not values or any(not isfinite(value) for value in values):
        raise ValueError(f"{name} must be non-empty and finite")


def _probability_vector(values: tuple[float, ...], name: str) -> None:
    _finite_vector(values, name)
    if any(value < 0.0 for value in values):
        raise ValueError(f"{name} cannot contain negative probabilities")
    if not isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=_PROB_TOL):
        raise ValueError(f"{name} must sum to one within 1e-10")


@dataclass(frozen=True, slots=True)
class GaussianHMMArtifact:
    state_count: int
    feature_order: tuple[str, ...]
    start_probabilities: tuple[float, ...]
    transition_matrix: tuple[tuple[float, ...], ...]
    means: tuple[tuple[float, ...], ...]
    full_covariances: tuple[tuple[tuple[float, ...], ...], ...]
    covariance_type: str = "full"

    def __post_init__(self) -> None:
        if self.state_count not in (2, 3, 4):
            raise ValueError("Gaussian MVP state_count must be 2, 3, or 4")
        if self.covariance_type != "full":
            raise ValueError("covariance_type must be exactly full")
        dimension = len(self.feature_order)
        if dimension < 1 or len(set(self.feature_order)) != dimension:
            raise ValueError("feature_order must be non-empty and duplicate-free")
        if len(self.start_probabilities) != self.state_count:
            raise ValueError("start probability dimension mismatch")
        _probability_vector(self.start_probabilities, "start_probabilities")
        if len(self.transition_matrix) != self.state_count:
            raise ValueError("transition matrix row count mismatch")
        for row in self.transition_matrix:
            if len(row) != self.state_count:
                raise ValueError("transition matrix must be square KxK")
            _probability_vector(row, "transition row")
        if len(self.means) != self.state_count:
            raise ValueError("mean state dimension mismatch")
        for mean in self.means:
            if len(mean) != dimension:
                raise ValueError("mean feature dimension mismatch")
            _finite_vector(mean, "state mean")
        if len(self.full_covariances) != self.state_count:
            raise ValueError("covariance state dimension mismatch")
        for covariance in self.full_covariances:
            if len(covariance) != dimension or any(len(row) != dimension for row in covariance):
                raise ValueError("each covariance must have exact d x d shape")
            flattened = tuple(value for row in covariance for value in row)
            if any(not isfinite(value) for value in flattened):
                raise ValueError("covariance values must be finite")
            max_asymmetry = max(
                abs(covariance[i][j] - covariance[j][i])
                for i in range(dimension)
                for j in range(dimension)
            )
            if max_asymmetry > _ASYMMETRY_TOL:
                raise ValueError("full covariance asymmetry exceeds 1e-10")
            if any(covariance[i][i] < _MIN_VARIANCE for i in range(dimension)):
                raise ValueError("full covariance diagonal variance is below 1e-12")

    @property
    def feature_dimension(self) -> int:
        return len(self.feature_order)
