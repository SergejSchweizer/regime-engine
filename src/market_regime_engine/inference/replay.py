"""Fixed-model replay on the complete-case retained-observation clock."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from market_regime_engine.features.ports import FeatureSnapshot
from market_regime_engine.inference.filtering import causal_filter
from market_regime_engine.models.production_artifact import ProductionModelArtifact


@dataclass(frozen=True, slots=True)
class ReplayInferenceResult:
    timestamps: tuple[datetime, ...]
    filtered_probabilities: tuple[tuple[float, ...], ...]
    warmup_observation_count: int

    def __post_init__(self) -> None:
        if not self.timestamps or len(self.timestamps) != len(self.filtered_probabilities):
            raise ValueError("replay inference requires matching non-empty timestamps/probabilities")
        if self.warmup_observation_count < 0:
            raise ValueError("warmup_observation_count cannot be negative")


def _validate_snapshot(
    artifact: ProductionModelArtifact,
    snapshot: FeatureSnapshot,
) -> None:
    if snapshot.feature_names != artifact.feature_order:
        raise ValueError("serving source feature order differs from production artifact")
    if snapshot.lineage.schema_version != artifact.source_schema_version:
        raise ValueError("serving source schema version is incompatible with production artifact")
    if snapshot.lineage.feature_version != artifact.source_feature_version:
        raise ValueError("serving source feature version is incompatible with production artifact")
    if snapshot.lineage.data_time_semantics != artifact.data_time_semantics:
        raise ValueError("serving source time semantics differ from production artifact")


def fixed_model_replay(
    artifact: ProductionModelArtifact,
    snapshot: FeatureSnapshot,
    *,
    start: datetime,
    end: datetime,
    deadline_check: Callable[[], None] = lambda: None,
) -> ReplayInferenceResult:
    """Replay without ever treating the caller's start as a fresh HMM initial condition."""

    if start > end:
        raise ValueError("replay start must not be after end")
    _validate_snapshot(artifact, snapshot)
    continuation = start > artifact.trained_through_timestamp
    origin = (
        artifact.trained_through_timestamp
        if continuation
        else artifact.inference_origin_timestamp
    )
    rows = tuple(
        row
        for row in snapshot.rows
        if row.timestamp >= origin
        and row.timestamp <= end
        and (not continuation or row.timestamp > artifact.trained_through_timestamp)
    )
    if not rows:
        raise ValueError("no_complete_observations")

    alpha = artifact.terminal_filtered_probabilities if continuation else None
    output_timestamps: list[datetime] = []
    output_probabilities: list[tuple[float, ...]] = []
    warmup = 0
    for row in rows:
        deadline_check()
        if any(value is None for value in row.values):
            raise ValueError("resolved-model snapshot cannot contain incomplete feature rows")
        matrix = np.asarray([[float(value) for value in row.values]], dtype=np.float64)
        scaled = artifact.scaler.transform(matrix)
        filtered = causal_filter(
            scaled,
            artifact.hmm,
            initial_filtered_probabilities=alpha,
        )
        alpha = filtered.terminal_probabilities
        if row.timestamp < start:
            warmup += 1
        else:
            output_timestamps.append(row.timestamp)
            output_probabilities.append(alpha)
    deadline_check()
    if not output_timestamps:
        raise ValueError("no_complete_observations")
    return ReplayInferenceResult(
        timestamps=tuple(output_timestamps),
        filtered_probabilities=tuple(output_probabilities),
        warmup_observation_count=warmup,
    )
