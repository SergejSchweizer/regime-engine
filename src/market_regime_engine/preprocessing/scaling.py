"""Train-only deterministic standardization for retained HMM observations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite

import numpy as np
import numpy.typing as npt

ArrayF64 = npt.NDArray[np.float64]
_MIN_VARIANCE = 1e-12


@dataclass(frozen=True, slots=True)
class StandardScalerArtifact:
    feature_order: tuple[str, ...]
    means: tuple[float, ...]
    variances: tuple[float, ...]
    scales: tuple[float, ...]
    constant_feature_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        dimension = len(self.feature_order)
        if dimension == 0 or len(set(self.feature_order)) != dimension:
            raise ValueError("feature_order must be non-empty and duplicate-free")
        if not (len(self.means) == len(self.variances) == len(self.scales) == dimension):
            raise ValueError("scaler parameter dimensions do not match feature_order")
        if any(not isfinite(value) for value in (*self.means, *self.variances, *self.scales)):
            raise ValueError("scaler parameters must be finite")
        if tuple(sorted(set(self.constant_feature_indices))) != self.constant_feature_indices:
            raise ValueError("constant_feature_indices must be sorted and duplicate-free")
        if any(index < 0 or index >= dimension for index in self.constant_feature_indices):
            raise ValueError("constant_feature_indices are outside feature_order")
        constants = set(self.constant_feature_indices)
        if any(
            value <= _MIN_VARIANCE and index not in constants
            for index, value in enumerate(self.variances)
        ):
            raise ValueError("training population variance must be greater than 1e-12")
        if any(value <= 0.0 for value in self.scales):
            raise ValueError("scaler standard deviations must be positive")
        if any(self.variances[index] != 0.0 or self.scales[index] != 1.0 for index in constants):
            raise ValueError("constant scaler features must have zero variance and unit scale")

    def transform(self, rows: npt.ArrayLike) -> ArrayF64:
        matrix = _matrix(rows, len(self.feature_order), "rows")
        means = np.asarray(self.means, dtype=np.float64)
        scales = np.asarray(self.scales, dtype=np.float64)
        return (matrix - means) / scales

    def to_canonical_json(self) -> str:
        payload = {
            "feature_order": list(self.feature_order),
            "means_hex": [value.hex() for value in self.means],
            "scales_hex": [value.hex() for value in self.scales],
            "constant_feature_indices": list(self.constant_feature_indices),
            "variances_hex": [value.hex() for value in self.variances],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_canonical_json(cls, payload: str) -> StandardScalerArtifact:
        raw = json.loads(payload)
        expected = {
            "constant_feature_indices",
            "feature_order",
            "means_hex",
            "scales_hex",
            "variances_hex",
        }
        if set(raw) != expected:
            raise ValueError("unknown/missing scaler serialization fields")
        return cls(
            feature_order=tuple(raw["feature_order"]),
            means=tuple(float.fromhex(value) for value in raw["means_hex"]),
            variances=tuple(float.fromhex(value) for value in raw["variances_hex"]),
            scales=tuple(float.fromhex(value) for value in raw["scales_hex"]),
            constant_feature_indices=tuple(int(value) for value in raw["constant_feature_indices"]),
        )


def fit_standard_scaler(
    retained_train_rows: npt.ArrayLike,
    feature_order: tuple[str, ...],
    *,
    allow_constant_features: bool = False,
) -> StandardScalerArtifact:
    """Fit exclusively on the supplied retained TRAIN matrix using population variance."""
    matrix = _matrix(retained_train_rows, len(feature_order), "retained_train_rows")
    if matrix.shape[0] == 0:
        raise ValueError("retained TRAIN matrix cannot be empty")
    means = np.mean(matrix, axis=0, dtype=np.float64)
    variances = np.var(matrix, axis=0, ddof=0, dtype=np.float64)
    if np.any(~np.isfinite(means)) or np.any(~np.isfinite(variances)):
        raise ValueError("TRAIN scaler parameters must be finite")
    constant_indices = tuple(int(index) for index, value in enumerate(variances) if value == 0.0)
    if np.any(variances <= _MIN_VARIANCE) and (
        not allow_constant_features or np.any((variances <= _MIN_VARIANCE) & (variances != 0.0))
    ):
        raise ValueError("training population variance must be greater than 1e-12")
    scales = np.sqrt(variances, where=variances > 0.0, out=np.ones_like(variances))
    return StandardScalerArtifact(
        feature_order=feature_order,
        means=tuple(float(value) for value in means),
        variances=tuple(float(value) for value in variances),
        scales=tuple(float(value) for value in scales),
        constant_feature_indices=constant_indices if allow_constant_features else (),
    )


def _matrix(rows: npt.ArrayLike, dimension: int, name: str) -> ArrayF64:
    if dimension < 1:
        raise ValueError("feature_order cannot be empty")
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ValueError(f"{name} must be a two-dimensional matrix in exact feature order")
    if np.any(~np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite complete-case values")
    return matrix
