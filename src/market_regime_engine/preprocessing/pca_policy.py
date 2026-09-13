"""Explicit PCA source-universe and frozen Inner-TRAIN fit-clock contracts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from market_regime_engine.preprocessing.pca import PCAArtifact, fit_pca_transformer

ArrayF64 = npt.NDArray[np.float64]
PCA_SOURCE_UNIVERSE = "catalog_feature_order_complete_case"


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{name} must be timezone-aware UTC")
    return value


def _feature_order(feature_order: tuple[str, ...]) -> None:
    if (
        not feature_order
        or len(set(feature_order)) != len(feature_order)
        or any(not name or name.strip() != name for name in feature_order)
    ):
        raise ValueError("PCA source feature_order must be non-empty and duplicate-free")


@dataclass(frozen=True, slots=True)
class PCAFitClock:
    """The immutable inner-training time interval used by one PCA fit."""

    inner_fold_id: str
    fit_start: datetime
    fit_end: datetime

    def __post_init__(self) -> None:
        if not self.inner_fold_id or self.inner_fold_id.strip() != self.inner_fold_id:
            raise ValueError("inner_fold_id must be a non-empty trimmed string")
        _utc(self.fit_start, "fit_start")
        _utc(self.fit_end, "fit_end")
        if self.fit_start > self.fit_end:
            raise ValueError("PCA fit_start must not be after fit_end")


@dataclass(frozen=True, slots=True)
class PCAFitResult:
    """PCA artifact plus the exact source universe and rows used to fit it."""

    clock: PCAFitClock
    feature_order: tuple[str, ...]
    artifact: PCAArtifact
    selected_timestamps: tuple[datetime, ...]
    skipped_incomplete_row_count: int
    source_universe: str = PCA_SOURCE_UNIVERSE

    def __post_init__(self) -> None:
        _feature_order(self.feature_order)
        if self.source_universe != PCA_SOURCE_UNIVERSE:
            raise ValueError("unsupported PCA source universe")
        if self.artifact.feature_order != self.feature_order:
            raise ValueError("PCA artifact feature order differs from fit source universe")
        if not self.selected_timestamps:
            raise ValueError("PCA fit must retain at least one complete TRAIN row")
        if self.skipped_incomplete_row_count < 0:
            raise ValueError("skipped incomplete row count cannot be negative")
        previous: datetime | None = None
        for timestamp in self.selected_timestamps:
            _utc(timestamp, "selected timestamp")
            if not self.clock.fit_start <= timestamp <= self.clock.fit_end:
                raise ValueError("selected PCA timestamp lies outside the frozen fit clock")
            if previous is not None and timestamp <= previous:
                raise ValueError("selected PCA timestamps must be strictly increasing")
            previous = timestamp

    @property
    def selected_row_count(self) -> int:
        return len(self.selected_timestamps)

    @property
    def fit_hash(self) -> str:
        return sha256(self.to_canonical_json().encode("utf-8")).hexdigest()

    def transform(self, rows: npt.ArrayLike) -> ArrayF64:
        return self.artifact.transform(rows)

    def to_canonical_json(self) -> str:
        payload = {
            "artifact": json.loads(self.artifact.to_canonical_json()),
            "clock": {
                "fit_end": self.clock.fit_end.isoformat(),
                "fit_start": self.clock.fit_start.isoformat(),
                "inner_fold_id": self.clock.inner_fold_id,
            },
            "feature_order": list(self.feature_order),
            "selected_timestamps": [value.isoformat() for value in self.selected_timestamps],
            "skipped_incomplete_row_count": self.skipped_incomplete_row_count,
            "source_universe": self.source_universe,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_canonical_json(cls, payload: str) -> PCAFitResult:
        raw = json.loads(payload)
        expected = {
            "artifact",
            "clock",
            "feature_order",
            "selected_timestamps",
            "skipped_incomplete_row_count",
            "source_universe",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("unknown/missing PCA fit serialization fields")
        clock_payload = raw["clock"]
        if not isinstance(clock_payload, dict) or set(clock_payload) != {
            "fit_end",
            "fit_start",
            "inner_fold_id",
        }:
            raise ValueError("unknown/missing PCA fit clock fields")
        artifact_payload = raw["artifact"]
        if not isinstance(artifact_payload, dict):
            raise ValueError("PCA fit artifact must be an object")
        artifact = PCAArtifact.from_canonical_json(
            json.dumps(artifact_payload, sort_keys=True, separators=(",", ":"))
        )
        feature_order = tuple(raw["feature_order"])
        return cls(
            clock=PCAFitClock(
                inner_fold_id=str(clock_payload["inner_fold_id"]),
                fit_start=datetime.fromisoformat(str(clock_payload["fit_start"])),
                fit_end=datetime.fromisoformat(str(clock_payload["fit_end"])),
            ),
            feature_order=feature_order,
            artifact=artifact,
            selected_timestamps=tuple(
                datetime.fromisoformat(str(value)) for value in raw["selected_timestamps"]
            ),
            skipped_incomplete_row_count=int(raw["skipped_incomplete_row_count"]),
            source_universe=str(raw["source_universe"]),
        )


def fit_pca_inner_train(
    timestamps: Sequence[datetime],
    source_rows: npt.ArrayLike,
    *,
    feature_order: tuple[str, ...],
    inner_fold_id: str,
    fit_start: datetime,
    fit_end: datetime,
    variance_threshold: float = 0.90,
) -> PCAFitResult:
    """Fit PCA on complete rows in one frozen Inner-TRAIN interval only.

    The caller supplies the complete catalog-derived feature order.  Rows
    outside the clock are ignored, while incomplete rows inside it are
    excluded explicitly and counted; no fill, interpolation, or carry is
    performed.
    """

    _feature_order(feature_order)
    clock = PCAFitClock(inner_fold_id, fit_start, fit_end)
    timestamp_values = tuple(_utc(value, "source timestamp") for value in timestamps)
    if any(current <= previous for previous, current in pairwise(timestamp_values)):
        raise ValueError("PCA source timestamps must be strictly increasing and unique")
    matrix = np.asarray(source_rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (len(timestamp_values), len(feature_order)):
        raise ValueError("PCA source rows must match timestamp count and exact feature order")
    in_clock = np.asarray(
        [clock.fit_start <= timestamp <= clock.fit_end for timestamp in timestamp_values],
        dtype=bool,
    )
    clock_rows = matrix[in_clock]
    complete = np.all(np.isfinite(clock_rows), axis=1)
    selected_rows = clock_rows[complete]
    complete_mask = np.zeros(len(timestamp_values), dtype=bool)
    complete_mask[np.flatnonzero(in_clock)] = complete
    selected_timestamps = tuple(
        timestamp
        for timestamp, is_complete in zip(timestamp_values, complete_mask, strict=True)
        if is_complete
    )
    if selected_rows.shape[0] == 0:
        raise ValueError("PCA fit clock contains no complete TRAIN rows")
    artifact = fit_pca_transformer(
        selected_rows,
        feature_order,
        variance_threshold=variance_threshold,
    )
    return PCAFitResult(
        clock=clock,
        feature_order=feature_order,
        artifact=artifact,
        selected_timestamps=selected_timestamps,
        skipped_incomplete_row_count=int(clock_rows.shape[0] - selected_rows.shape[0]),
    )


__all__ = [
    "PCA_SOURCE_UNIVERSE",
    "PCAFitClock",
    "PCAFitResult",
    "fit_pca_inner_train",
]
