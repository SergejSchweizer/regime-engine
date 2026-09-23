"""Fold-local two-stage PCA and HMM standardization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from market_regime_engine.preprocessing.scaling import (
    StandardScalerArtifact,
    fit_standard_scaler,
)

ArrayF64 = npt.NDArray[np.float64]

if TYPE_CHECKING:
    from market_regime_engine.feature_discovery.family_pca import FamilyPCAArtifact


def _raw_matrix(rows: npt.ArrayLike, dimension: int) -> ArrayF64:
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ValueError("raw PCA/HMM rows must preserve exact raw feature order")
    if matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("raw PCA/HMM rows must be non-empty and finite")
    return matrix


@dataclass(frozen=True, slots=True)
class FamilyPCATwoStageScalerArtifact:
    """Frozen canonical family-PCA transform followed by HMM scaling."""

    raw_feature_order: tuple[str, ...]
    family_pca: tuple[FamilyPCAArtifact, ...]
    hmm_scaler: StandardScalerArtifact
    selected_feature_order: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.raw_feature_order or len(set(self.raw_feature_order)) != len(
            self.raw_feature_order
        ):
            raise ValueError("family PCA raw feature order must be unique and non-empty")
        generated: tuple[str, ...] = tuple(
            name for artifact in self.family_pca for name in artifact.generated_feature_names
        )
        full_order = self.raw_feature_order + generated
        if len(set(generated)) != len(generated):
            raise ValueError("family PCA generated feature names must be unique")
        if not self.selected_feature_order or len(set(self.selected_feature_order)) != len(
            self.selected_feature_order
        ):
            raise ValueError("family PCA model feature order must be unique and non-empty")
        if any(name not in full_order for name in self.selected_feature_order):
            raise ValueError("family PCA model feature order contains unknown features")
        if self.hmm_scaler.feature_order != self.selected_feature_order:
            raise ValueError("HMM scaler order must equal selected family PCA features")

    @property
    def model_feature_order(self) -> tuple[str, ...]:
        return self.selected_feature_order

    @property
    def fit_hash(self) -> str:
        return sha256(self.to_canonical_json().encode("utf-8")).hexdigest()

    def transform(self, raw_rows: npt.ArrayLike) -> ArrayF64:
        raw = _raw_matrix(raw_rows, len(self.raw_feature_order))
        positions = {name: index for index, name in enumerate(self.raw_feature_order)}
        generated_columns: list[ArrayF64] = []
        generated_names: list[str] = []
        for artifact in self.family_pca:
            indices = [positions[name] for name in artifact.feature_order]
            transformed = artifact.transform(raw[:, indices])
            generated_columns.extend(transformed[:, index] for index in range(transformed.shape[1]))
            generated_names.extend(artifact.generated_feature_names)
        full = np.column_stack((raw, *generated_columns)) if generated_columns else raw
        full_order = self.raw_feature_order + tuple(generated_names)
        selected = full[:, [full_order.index(name) for name in self.model_feature_order]]
        return self.hmm_scaler.transform(selected)

    def to_canonical_json(self) -> str:
        return json.dumps(
            {
                "artifact_schema": "RegimeEngineFamilyPCATwoStageScaler.v1",
                "family_pca": [json.loads(item.to_canonical_json()) for item in self.family_pca],
                "hmm_scaler": json.loads(self.hmm_scaler.to_canonical_json()),
                "raw_feature_order": list(self.raw_feature_order),
                "selected_feature_order": list(self.model_feature_order),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @classmethod
    def from_canonical_json(cls, payload: str) -> FamilyPCATwoStageScalerArtifact:
        from market_regime_engine.feature_discovery.family_pca import FamilyPCAArtifact

        raw = json.loads(payload)
        expected = {
            "artifact_schema",
            "family_pca",
            "hmm_scaler",
            "raw_feature_order",
            "selected_feature_order",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("unknown/missing family PCA two-stage scaler fields")
        if raw["artifact_schema"] != "RegimeEngineFamilyPCATwoStageScaler.v1":
            raise ValueError("unsupported family PCA two-stage scaler schema")
        family_payload = raw["family_pca"]
        scaler_payload = raw["hmm_scaler"]
        if not isinstance(family_payload, list) or not isinstance(scaler_payload, dict):
            raise ValueError("family PCA two-stage payloads have invalid types")
        artifacts = tuple(
            FamilyPCAArtifact.from_canonical_json(
                json.dumps(item, sort_keys=True, separators=(",", ":"))
            )
            for item in family_payload
            if isinstance(item, dict)
        )
        if len(artifacts) != len(family_payload):
            raise ValueError("family PCA artifacts must be objects")
        scaler = StandardScalerArtifact.from_canonical_json(
            json.dumps(scaler_payload, sort_keys=True, separators=(",", ":"))
        )
        return cls(
            raw_feature_order=tuple(raw["raw_feature_order"]),
            family_pca=artifacts,
            hmm_scaler=scaler,
            selected_feature_order=tuple(raw["selected_feature_order"]),
        )


def fit_family_pca_hmm_scaler(
    raw_train_rows: npt.ArrayLike,
    *,
    raw_feature_order: tuple[str, ...],
    family_pca: tuple[FamilyPCAArtifact, ...],
    model_feature_order: tuple[str, ...],
) -> FamilyPCATwoStageScalerArtifact:
    """Fit HMM scaling on a canonical family-PCA materialized TRAIN matrix."""

    raw = _raw_matrix(raw_train_rows, len(raw_feature_order))
    positions = {name: index for index, name in enumerate(raw_feature_order)}
    generated_columns: list[ArrayF64] = []
    generated_names: list[str] = []
    for artifact in family_pca:
        indices = [positions[name] for name in artifact.feature_order]
        transformed = artifact.transform(raw[:, indices])
        generated_columns.extend(transformed[:, index] for index in range(transformed.shape[1]))
        generated_names.extend(artifact.generated_feature_names)
    full = np.column_stack((raw, *generated_columns)) if generated_columns else raw
    full_order = raw_feature_order + tuple(generated_names)
    if any(name not in full_order for name in model_feature_order):
        raise ValueError("family PCA model feature order contains unknown features")
    selected = full[:, [full_order.index(name) for name in model_feature_order]]
    scaler = fit_standard_scaler(selected, model_feature_order)
    return FamilyPCATwoStageScalerArtifact(
        raw_feature_order=raw_feature_order,
        family_pca=family_pca,
        hmm_scaler=scaler,
        selected_feature_order=model_feature_order,
    )


__all__ = [
    "FamilyPCATwoStageScalerArtifact",
    "fit_family_pca_hmm_scaler",
]
