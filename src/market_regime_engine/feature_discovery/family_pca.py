"""Family-local TRAIN-only standardization and fixed-rank PCA."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

import numpy as np
import numpy.typing as npt

from market_regime_engine.feature_discovery.feature_roles import (
    FAMILY_PCA_MAX_COMPONENTS,
    FeatureRoleContract,
    FeatureSelectionProfile,
    FeatureStage,
    family_pc_name,
)
from market_regime_engine.preprocessing.scaling import (
    StandardScalerArtifact,
    fit_standard_scaler,
)

ArrayF64 = npt.NDArray[np.float64]
_RANK_TOLERANCE = 1.0e-12


@dataclass(frozen=True, slots=True)
class FamilyPCAArtifact:
    family: str
    feature_order: tuple[str, ...]
    scaler: StandardScalerArtifact
    components: tuple[tuple[float, ...], ...]
    explained_variance_ratio: tuple[float, ...]
    numerical_rank: int
    profile_hash: str

    def __post_init__(self) -> None:
        if not self.feature_order or len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("family PCA feature order must be non-empty and duplicate-free")
        if self.scaler.feature_order != self.feature_order:
            raise ValueError("family PCA scaler order differs from source order")
        if not 1 <= self.numerical_rank <= len(self.feature_order):
            raise ValueError("family PCA numerical rank is out of bounds")
        if len(self.components) != self.numerical_rank:
            raise ValueError("family PCA component count must equal numerical rank")
        if len(self.components) > FAMILY_PCA_MAX_COMPONENTS:
            raise ValueError("family PCA retained more than eight components")
        if len(self.explained_variance_ratio) != len(self.feature_order):
            raise ValueError("family PCA diagnostic variance dimension is invalid")
        if any(
            len(component) != len(self.feature_order)
            or any(not isfinite(value) for value in component)
            for component in self.components
        ):
            raise ValueError("family PCA components must be finite and dimensionally aligned")
        if any(not isfinite(value) or value < 0.0 for value in self.explained_variance_ratio):
            raise ValueError("family PCA explained variance must be finite and non-negative")
        matrix = np.asarray(self.components, dtype=np.float64)
        if not np.allclose(
            matrix @ matrix.T,
            np.eye(self.numerical_rank, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ValueError("family PCA components must be orthonormal")
        if len(self.profile_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.profile_hash
        ):
            raise ValueError("family PCA profile hash must be a lowercase SHA-256")

    @property
    def generated_feature_names(self) -> tuple[str, ...]:
        return tuple(
            family_pc_name(self.family, index) for index in range(1, self.numerical_rank + 1)
        )

    @property
    def retained_component_count(self) -> int:
        return self.numerical_rank

    @property
    def cumulative_explained_variance(self) -> float:
        return float(sum(self.explained_variance_ratio[: self.numerical_rank]))

    @property
    def fit_hash(self) -> str:
        payload = {
            "family": self.family,
            "feature_order": self.feature_order,
            "components": self.components,
            "explained_variance_ratio": self.explained_variance_ratio,
            "numerical_rank": self.numerical_rank,
            "profile_hash": self.profile_hash,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()

    def transform(self, rows: npt.ArrayLike) -> ArrayF64:
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_order):
            raise ValueError("family PCA rows must match the exact TRAIN feature order")
        if np.any(~np.isfinite(matrix)):
            raise ValueError("family PCA transform rows must be complete and finite")
        return self.scaler.transform(matrix) @ np.asarray(self.components, dtype=np.float64).T


def _canonicalize_signs(components: ArrayF64) -> ArrayF64:
    oriented = np.array(components, dtype=np.float64, copy=True)
    for index in range(oriented.shape[0]):
        pivot = int(np.argmax(np.abs(oriented[index])))
        if oriented[index, pivot] < 0.0:
            oriented[index] *= -1.0
    return oriented


def fit_family_pca(
    train_rows: npt.ArrayLike,
    feature_order: Sequence[str],
    contract: FeatureRoleContract,
    *,
    profile: FeatureSelectionProfile | None = None,
) -> FamilyPCAArtifact:
    """Fit one family PCA on complete TRAIN rows only.

    The retained count is ``min(numerical_rank, 8)``.  No explained-variance
    target, labels, future data, HMM score, likelihood, AIC or BIC enters the
    count or the component orientation.
    """

    order = tuple(feature_order)
    if not order:
        raise ValueError("family PCA requires at least one source feature")
    contract.validate_stage_features(FeatureStage.FAMILY_PCA, order)
    families = {contract.assignment(name).family for name in order}
    if len(families) != 1:
        raise ValueError("family PCA feature order must contain exactly one family")
    family = next(iter(families))
    assert family is not None
    resolved_profile = contract.profile if profile is None else profile
    if resolved_profile.profile_hash != contract.profile.profile_hash:
        raise ValueError("family PCA profile must match the role contract profile")
    matrix = np.asarray(train_rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(order):
        raise ValueError("family PCA TRAIN rows must match the exact feature order")
    if matrix.shape[0] < 1 or np.any(~np.isfinite(matrix)):
        raise ValueError("family PCA requires non-empty complete finite TRAIN rows")
    scaler = fit_standard_scaler(matrix, order)
    standardized = scaler.transform(matrix)
    _u, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise ValueError("family PCA requires positive TRAIN variation")
    rank_tolerance = max(standardized.shape) * singular_values[0] * _RANK_TOLERANCE
    numerical_rank = int(np.count_nonzero(singular_values > rank_tolerance))
    if numerical_rank < 1:
        raise ValueError("family PCA numerical rank is zero")
    retained_count = min(numerical_rank, resolved_profile.family_pca_max_components)
    components = _canonicalize_signs(vt[:retained_count])
    explained = np.square(singular_values, dtype=np.float64)
    total = float(np.sum(explained, dtype=np.float64))
    if not isfinite(total) or total <= 0.0:
        raise ValueError("family PCA explained variance must be positive and finite")
    ratios = explained / total
    return FamilyPCAArtifact(
        family=family,
        feature_order=order,
        scaler=scaler,
        components=tuple(tuple(float(value) for value in row) for row in components),
        explained_variance_ratio=tuple(float(value) for value in ratios),
        numerical_rank=retained_count,
        profile_hash=resolved_profile.profile_hash,
    )


__all__ = ["FamilyPCAArtifact", "fit_family_pca"]
