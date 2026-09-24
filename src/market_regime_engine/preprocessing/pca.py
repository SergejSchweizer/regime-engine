"""Deterministic TRAIN-only standardized PCA artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isclose, isfinite

import numpy as np
import numpy.typing as npt

from market_regime_engine.evaluation.errors import RecoverableEvaluationInvalidity
from market_regime_engine.preprocessing.scaling import StandardScalerArtifact, fit_standard_scaler

ArrayF64 = npt.NDArray[np.float64]
_DEFAULT_VARIANCE_THRESHOLD = 0.90
_ORTHONORMAL_TOLERANCE = 1.0e-10


def _matrix(rows: npt.ArrayLike, dimension: int, name: str) -> ArrayF64:
    if dimension < 1:
        raise ValueError("feature_order cannot be empty")
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ValueError(f"{name} must be a two-dimensional matrix in exact feature order")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty")
    if np.any(~np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite complete-case values")
    return matrix


def _validate_threshold(value: float) -> None:
    if not isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("variance_threshold must be finite and in (0, 1]")


@dataclass(frozen=True, slots=True)
class PCAArtifact:
    """Immutable standardized PCA fit on one complete TRAIN matrix.

    Component signs are canonicalized by making the loading with the largest
    absolute magnitude positive (ties choose the first feature).  The full
    explained-variance-ratio vector is retained for auditability.  Automatic
    component selection stores the minimal prefix meeting
    ``variance_threshold``; an explicit component count is a fixed-dimension
    contract and records the threshold diagnostically without requiring that
    fixed prefix to reach it.
    """

    scaler: StandardScalerArtifact
    components: tuple[tuple[float, ...], ...]
    explained_variance_ratio: tuple[float, ...]
    retained_component_count: int
    variance_threshold: float = _DEFAULT_VARIANCE_THRESHOLD
    variance_threshold_enforced: bool = True

    def __post_init__(self) -> None:
        _validate_threshold(self.variance_threshold)
        dimension = len(self.scaler.feature_order)
        if not self.components or len(self.components) > dimension:
            raise ValueError("PCA components must contain between one and d rows")
        if len(self.explained_variance_ratio) != dimension:
            raise ValueError("explained variance ratio must contain one value per input feature")
        if self.retained_component_count != len(self.components):
            raise ValueError("retained component count does not match component rows")
        if any(
            len(component) != dimension or any(not isfinite(value) for value in component)
            for component in self.components
        ):
            raise ValueError("PCA components must be finite and have exact input dimension")
        if any(
            not isfinite(value) or value < 0.0 for value in self.explained_variance_ratio
        ) or not isclose(sum(self.explained_variance_ratio), 1.0, abs_tol=1.0e-10, rel_tol=0.0):
            raise ValueError("explained variance ratios must be finite, nonnegative and sum to one")
        cumulative = float(sum(self.explained_variance_ratio[: self.retained_component_count]))
        if (
            self.variance_threshold_enforced
            and cumulative + _ORTHONORMAL_TOLERANCE < self.variance_threshold
        ):
            raise RecoverableEvaluationInvalidity(
                "retained PCA components do not meet the variance threshold"
            )
        matrix = np.asarray(self.components, dtype=np.float64)
        if not np.allclose(
            matrix @ matrix.T,
            np.eye(self.retained_component_count, dtype=np.float64),
            rtol=0.0,
            atol=_ORTHONORMAL_TOLERANCE,
        ):
            raise ValueError("PCA components must be orthonormal")

    @property
    def feature_order(self) -> tuple[str, ...]:
        return self.scaler.feature_order

    @property
    def feature_dimension(self) -> int:
        return len(self.feature_order)

    @property
    def generated_feature_names(self) -> tuple[str, ...]:
        return tuple(
            f"pca_pc_{component_index:03d}"
            for component_index in range(1, self.retained_component_count + 1)
        )

    @property
    def cumulative_explained_variance(self) -> float:
        return float(sum(self.explained_variance_ratio[: self.retained_component_count]))

    def transform(self, rows: npt.ArrayLike) -> ArrayF64:
        """Transform rows using only parameters frozen in this artifact."""

        matrix = _matrix(rows, self.feature_dimension, "rows")
        return self.scaler.transform(matrix) @ np.asarray(self.components, dtype=np.float64).T

    def to_canonical_json(self) -> str:
        payload = {
            "artifact_schema": "RegimeEnginePCA.v2",
            "components_hex": [
                [value.hex() for value in component] for component in self.components
            ],
            "explained_variance_ratio_hex": [
                value.hex() for value in self.explained_variance_ratio
            ],
            "feature_order": list(self.feature_order),
            "retained_component_count": self.retained_component_count,
            "scaler": json.loads(self.scaler.to_canonical_json()),
            "variance_threshold_hex": self.variance_threshold.hex(),
            "variance_threshold_enforced": self.variance_threshold_enforced,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_canonical_json(cls, payload: str) -> PCAArtifact:
        raw = json.loads(payload)
        expected = {
            "artifact_schema",
            "components_hex",
            "explained_variance_ratio_hex",
            "feature_order",
            "retained_component_count",
            "scaler",
            "variance_threshold_hex",
            "variance_threshold_enforced",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("unknown/missing PCA serialization fields")
        scaler_payload = raw["scaler"]
        if not isinstance(scaler_payload, dict):
            raise ValueError("PCA scaler serialization must be an object")
        scaler = StandardScalerArtifact.from_canonical_json(
            json.dumps(scaler_payload, sort_keys=True, separators=(",", ":"))
        )
        if tuple(raw["feature_order"]) != scaler.feature_order:
            raise ValueError("PCA feature order does not match scaler feature order")
        if raw["artifact_schema"] != "RegimeEnginePCA.v2":
            raise ValueError("unsupported PCA artifact schema")
        return cls(
            scaler=scaler,
            components=tuple(
                tuple(float.fromhex(value) for value in component)
                for component in raw["components_hex"]
            ),
            explained_variance_ratio=tuple(
                float.fromhex(value) for value in raw["explained_variance_ratio_hex"]
            ),
            retained_component_count=int(raw["retained_component_count"]),
            variance_threshold=float.fromhex(raw["variance_threshold_hex"]),
            variance_threshold_enforced=bool(raw["variance_threshold_enforced"]),
        )


def _canonicalize_component_signs(components: ArrayF64) -> ArrayF64:
    oriented = np.array(components, dtype=np.float64, copy=True)
    for index in range(oriented.shape[0]):
        pivot = int(np.argmax(np.abs(oriented[index])))
        if oriented[index, pivot] < 0.0:
            oriented[index] *= -1.0
    return oriented


def fit_pca_transformer(
    retained_train_rows: npt.ArrayLike,
    feature_order: tuple[str, ...],
    *,
    variance_threshold: float = _DEFAULT_VARIANCE_THRESHOLD,
    component_count: int | None = None,
) -> PCAArtifact:
    """Fit standardized PCA exclusively on the supplied complete TRAIN rows."""

    _validate_threshold(variance_threshold)
    matrix = _matrix(retained_train_rows, len(feature_order), "retained_train_rows")
    scaler = fit_standard_scaler(matrix, feature_order, allow_constant_features=True)
    standardized = scaler.transform(matrix)
    _u, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    explained = np.square(singular_values, dtype=np.float64)
    total = float(np.sum(explained, dtype=np.float64))
    if not isfinite(total) or total <= 0.0:
        raise ValueError("PCA explained variance must be positive and finite")
    explained_ratio = explained / total
    if explained_ratio.size < matrix.shape[1]:
        explained_ratio = np.pad(
            explained_ratio,
            (0, matrix.shape[1] - explained_ratio.size),
            mode="constant",
        )
    cumulative = np.cumsum(explained_ratio, dtype=np.float64)
    if component_count is not None and component_count < 1:
        raise ValueError("component_count must be positive")
    retained = (
        int(component_count)
        if component_count is not None
        else int(np.searchsorted(cumulative, variance_threshold, side="left")) + 1
    )
    retained = min(retained, vt.shape[0])
    components = _canonicalize_component_signs(vt[:retained])
    return PCAArtifact(
        scaler=scaler,
        components=tuple(tuple(float(value) for value in row) for row in components),
        explained_variance_ratio=tuple(float(value) for value in explained_ratio),
        retained_component_count=retained,
        variance_threshold=variance_threshold,
        variance_threshold_enforced=component_count is None,
    )


__all__ = ["PCAArtifact", "fit_pca_transformer"]
