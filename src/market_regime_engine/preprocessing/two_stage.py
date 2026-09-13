"""Fold-local two-stage PCA and HMM standardization."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256

import numpy as np
import numpy.typing as npt

from market_regime_engine.preprocessing.pca_policy import PCAFitResult, fit_pca_inner_train
from market_regime_engine.preprocessing.scaling import (
    StandardScalerArtifact,
    fit_standard_scaler,
)

ArrayF64 = npt.NDArray[np.float64]


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

    def __post_init__(self) -> None:
        raw_order = self.pca_fit.feature_order
        expected_order = raw_order + self.pca_fit.artifact.generated_feature_names
        if self.hmm_scaler.feature_order != expected_order:
            raise ValueError("HMM scaler order must be raw features followed by PCA components")

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
        return self.hmm_scaler.transform(np.column_stack((raw, generated)))

    def to_canonical_json(self) -> str:
        return json.dumps(
            {
                "artifact_schema": "RegimeEnginePCATwoStageScaler.v1",
                "hmm_scaler": json.loads(self.hmm_scaler.to_canonical_json()),
                "pca_fit": json.loads(self.pca_fit.to_canonical_json()),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @classmethod
    def from_canonical_json(cls, payload: str) -> PCATwoStageScalerArtifact:
        raw = json.loads(payload)
        expected = {"artifact_schema", "hmm_scaler", "pca_fit"}
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("unknown/missing PCA two-stage scaler fields")
        if raw["artifact_schema"] != "RegimeEnginePCATwoStageScaler.v1":
            raise ValueError("unsupported PCA two-stage scaler schema")
        pca_payload = raw["pca_fit"]
        scaler_payload = raw["hmm_scaler"]
        if not isinstance(pca_payload, dict) or not isinstance(scaler_payload, dict):
            raise ValueError("PCA two-stage scaler payloads must be objects")
        pca_fit = PCAFitResult.from_canonical_json(
            json.dumps(pca_payload, sort_keys=True, separators=(",", ":"))
        )
        hmm_scaler = StandardScalerArtifact.from_canonical_json(
            json.dumps(scaler_payload, sort_keys=True, separators=(",", ":"))
        )
        return cls(pca_fit=pca_fit, hmm_scaler=hmm_scaler)


def fit_pca_hmm_scaler(
    timestamps: Sequence[datetime],
    raw_train_rows: npt.ArrayLike,
    *,
    raw_feature_order: tuple[str, ...],
    inner_fold_id: str,
    fit_start: datetime,
    fit_end: datetime,
    variance_threshold: float = 0.90,
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
    )
    by_timestamp = {timestamp: index for index, timestamp in enumerate(timestamps)}
    selected_indices = tuple(by_timestamp[timestamp] for timestamp in pca_fit.selected_timestamps)
    selected_raw = raw[np.asarray(selected_indices, dtype=np.intp)]
    selected_pca = pca_fit.transform(selected_raw)
    model_order = raw_feature_order + pca_fit.artifact.generated_feature_names
    hmm_scaler = fit_standard_scaler(
        np.column_stack((selected_raw, selected_pca)),
        model_order,
    )
    return PCATwoStageScalerArtifact(pca_fit=pca_fit, hmm_scaler=hmm_scaler)


__all__ = ["PCATwoStageScalerArtifact", "fit_pca_hmm_scaler"]
