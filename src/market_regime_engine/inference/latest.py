"""Causal latest prediction on the complete-case retained-observation clock."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from market_regime_engine.features.ports import FeatureRow, FeatureSnapshot
from market_regime_engine.inference.filtering import causal_filter
from market_regime_engine.models.production_artifact import ProductionModelArtifact


@dataclass(frozen=True, slots=True)
class LatestInferenceResult:
    timestamp: datetime
    filtered_probabilities: tuple[float, ...]
    warmup_observation_count: int

    def __post_init__(self) -> None:
        if not self.filtered_probabilities:
            raise ValueError("latest inference probabilities cannot be empty")
        if self.warmup_observation_count < 0:
            raise ValueError("warmup_observation_count cannot be negative")


def _validate_snapshot(artifact: ProductionModelArtifact, snapshot: FeatureSnapshot) -> None:
    if snapshot.feature_names != artifact.feature_order:
        raise ValueError("serving source feature order differs from production artifact")
    if snapshot.lineage.schema_version != artifact.source_schema_version:
        raise ValueError("serving source schema version is incompatible with production artifact")
    if snapshot.lineage.feature_version != artifact.source_feature_version:
        raise ValueError("serving source feature version is incompatible with production artifact")
    if snapshot.lineage.data_time_semantics != artifact.data_time_semantics:
        raise ValueError("serving source time semantics differ from production artifact")


def _filter_rows(
    artifact: ProductionModelArtifact,
    rows: tuple[FeatureRow, ...],
    *,
    initial: tuple[float, ...] | None,
) -> tuple[tuple[float, ...], int]:
    alpha = initial
    count = 0
    for row in rows:
        complete_values = tuple(value for value in row.values if value is not None)
        if len(complete_values) != len(row.values):
            raise ValueError("resolved-model snapshot cannot contain incomplete feature rows")
        matrix = np.asarray([complete_values], dtype=np.float64)
        filtered = causal_filter(
            artifact.scaler.transform(matrix),
            artifact.hmm,
            initial_filtered_probabilities=alpha,
        )
        alpha = filtered.terminal_probabilities
        count += 1
    if alpha is None:
        raise ValueError("no_complete_observations")
    return alpha, count


def latest_prediction(
    artifact: ProductionModelArtifact,
    snapshot: FeatureSnapshot,
    *,
    as_of: datetime,
) -> LatestInferenceResult:
    _validate_snapshot(artifact, snapshot)
    bounded = tuple(row for row in snapshot.rows if row.timestamp <= as_of)
    if as_of <= artifact.trained_through_timestamp:
        rows = tuple(row for row in bounded if row.timestamp >= artifact.inference_origin_timestamp)
        if not rows:
            raise ValueError("no_complete_observations")
        alpha, count = _filter_rows(artifact, rows, initial=None)
        return LatestInferenceResult(rows[-1].timestamp, alpha, count - 1)

    subsequent = tuple(row for row in bounded if row.timestamp > artifact.trained_through_timestamp)
    if subsequent:
        alpha, count = _filter_rows(
            artifact,
            subsequent,
            initial=artifact.terminal_filtered_probabilities,
        )
        return LatestInferenceResult(subsequent[-1].timestamp, alpha, count - 1)

    if any(row.timestamp == artifact.trained_through_timestamp for row in bounded):
        return LatestInferenceResult(
            artifact.trained_through_timestamp,
            artifact.terminal_filtered_probabilities,
            0,
        )
    raise ValueError("no_complete_observations")
