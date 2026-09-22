"""Fold-local two-stage PCA and HMM standardization."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from market_regime_engine.preprocessing.pca_policy import PCAFitResult, fit_pca_inner_train
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
class PCATwoStageScalerArtifact:
    """Frozen PCA transform followed by a separately fitted HMM scaler."""

    pca_fit: PCAFitResult
    hmm_scaler: StandardScalerArtifact
    selected_feature_order: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        raw_order = self.pca_fit.feature_order
        full_order = raw_order + self.pca_fit.artifact.generated_feature_names
        model_order = (
            full_order if self.selected_feature_order is None else self.selected_feature_order
        )
        if not model_order or len(set(model_order)) != len(model_order):
            raise ValueError("PCA/HMM model feature order must be unique and non-empty")
        if any(feature not in full_order for feature in model_order):
            raise ValueError("PCA/HMM model feature order contains unknown features")
        if self.hmm_scaler.feature_order != model_order:
            raise ValueError("HMM scaler order must equal the selected PCA model features")

    @property
    def raw_feature_order(self) -> tuple[str, ...]:
        return self.pca_fit.feature_order

    @property
    def model_feature_order(self) -> tuple[str, ...]:
        return self.hmm_scaler.feature_order

    @property
    def fit_hash(self) -> str:
        return sha256(self.to_canonical_json().encode("utf-8")).hexdigest()

    def transform(self, raw_rows: npt.ArrayLike) -> ArrayF64:
        raw = _raw_matrix(raw_rows, len(self.raw_feature_order))
        generated = self.pca_fit.transform(raw)
        full = np.column_stack((raw, generated))
        full_order = self.raw_feature_order + self.pca_fit.artifact.generated_feature_names
        selected = full[:, [full_order.index(name) for name in self.model_feature_order]]
        return self.hmm_scaler.transform(selected)

    def to_canonical_json(self) -> str:
        return json.dumps(
            {
                "artifact_schema": "RegimeEnginePCATwoStageScaler.v1",
                "hmm_scaler": json.loads(self.hmm_scaler.to_canonical_json()),
                "pca_fit": json.loads(self.pca_fit.to_canonical_json()),
                "selected_feature_order": list(self.model_feature_order),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @classmethod
    def from_canonical_json(cls, payload: str) -> PCATwoStageScalerArtifact:
        raw = json.loads(payload)
        expected = {"artifact_schema", "hmm_scaler", "pca_fit", "selected_feature_order"}
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("unknown/missing PCA two-stage scaler fields")
        if raw["artifact_schema"] != "RegimeEnginePCATwoStageScaler.v1":
            raise ValueError("unsupported PCA two-stage scaler schema")
        pca_payload = raw["pca_fit"]
        scaler_payload = raw["hmm_scaler"]
        selected_payload = raw["selected_feature_order"]
        if not isinstance(pca_payload, dict) or not isinstance(scaler_payload, dict):
            raise ValueError("PCA two-stage scaler payloads must be objects")
        if (
            not isinstance(selected_payload, list)
            or not selected_payload
            or any(not isinstance(value, str) for value in selected_payload)
        ):
            raise ValueError("PCA two-stage selected feature order must be a string list")
        pca_fit = PCAFitResult.from_canonical_json(
            json.dumps(pca_payload, sort_keys=True, separators=(",", ":"))
        )
        hmm_scaler = StandardScalerArtifact.from_canonical_json(
            json.dumps(scaler_payload, sort_keys=True, separators=(",", ":"))
        )
        return cls(
            pca_fit=pca_fit,
            hmm_scaler=hmm_scaler,
            selected_feature_order=tuple(selected_payload),
        )


def fit_pca_hmm_scaler(
    timestamps: Sequence[datetime],
    raw_train_rows: npt.ArrayLike,
    *,
    raw_feature_order: tuple[str, ...],
    inner_fold_id: str,
    fit_start: datetime,
    fit_end: datetime,
    variance_threshold: float = 0.90,
    component_count: int | None = None,
    model_feature_order: tuple[str, ...] | None = None,
) -> PCATwoStageScalerArtifact:
    """Fit PCA on frozen Inner-TRAIN, then fit HMM scaling on PCA-augmented TRAIN."""

    raw = _raw_matrix(raw_train_rows, len(raw_feature_order))
    if raw.shape[0] != len(timestamps):
        raise ValueError("raw rows and timestamps must have equal length")
    pca_fit = fit_pca_inner_train(
        timestamps,
        raw,
        feature_order=raw_feature_order,
        inner_fold_id=inner_fold_id,
        fit_start=fit_start,
        fit_end=fit_end,
        variance_threshold=variance_threshold,
        component_count=component_count,
    )
    by_timestamp = {timestamp: index for index, timestamp in enumerate(timestamps)}
    selected_indices = tuple(by_timestamp[timestamp] for timestamp in pca_fit.selected_timestamps)
    selected_raw = raw[np.asarray(selected_indices, dtype=np.intp)]
    selected_pca = pca_fit.transform(selected_raw)
    full_order = raw_feature_order + pca_fit.artifact.generated_feature_names
    model_order = full_order if model_feature_order is None else model_feature_order
    if not model_order or len(set(model_order)) != len(model_order):
        raise ValueError("PCA/HMM model feature order must be unique and non-empty")
    if any(feature not in full_order for feature in model_order):
        raise ValueError("PCA/HMM model feature order contains unknown features")
    selected = np.column_stack((selected_raw, selected_pca))[
        :, [full_order.index(name) for name in model_order]
    ]
    hmm_scaler = fit_standard_scaler(
        selected,
        model_order,
    )
    return PCATwoStageScalerArtifact(
        pca_fit=pca_fit,
        hmm_scaler=hmm_scaler,
        selected_feature_order=model_order,
    )


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
    "PCATwoStageScalerArtifact",
    "fit_family_pca_hmm_scaler",
    "fit_pca_hmm_scaler",
]
