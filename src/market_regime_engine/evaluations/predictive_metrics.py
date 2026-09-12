"""Leak-free retained-observation forecast metrics for explicit target data."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from itertools import pairwise
from math import isfinite, sqrt

import numpy as np

from market_regime_engine.mlflow_support.metric_catalog import validate_metric_points
from market_regime_engine.mlflow_support.ports import MetricPoint

_PROBABILITY_TOLERANCE = 1.0e-10


def _timestamp_ms(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("forecast timestamps must be timezone-aware UTC")
    result = int(value.timestamp() * 1000)
    if result < 0:
        raise ValueError("forecast timestamps must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class ForecastContract:
    """Explicit boundary and scaling contract for a forecast target."""

    target_name: str
    horizons: tuple[int, ...] = (1,)
    horizon_units: str = "retained_model_observations"
    information_boundary: str = "origin_filtered_state_and_frozen_transition"
    missing_target_policy: str = "omit_incomplete_horizon"
    scaling_semantics: str = "target_units"

    def __post_init__(self) -> None:
        if not self.target_name or self.target_name.strip() != self.target_name:
            raise ValueError("forecast target_name must be non-empty and trimmed")
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("forecast horizons must be sorted, unique and non-empty")
        if any(isinstance(value, bool) or value < 1 for value in self.horizons):
            raise ValueError("forecast horizons must be positive integers")
        if self.horizon_units != "retained_model_observations":
            raise ValueError("forecast horizon units must be retained model observations")
        if self.information_boundary != "origin_filtered_state_and_frozen_transition":
            raise ValueError("unsupported forecast information boundary")
        if self.missing_target_policy != "omit_incomplete_horizon":
            raise ValueError("unsupported missing-target policy")
        if self.scaling_semantics != "target_units":
            raise ValueError("unsupported forecast scaling semantics")


@dataclass(frozen=True, slots=True)
class ForecastObservation:
    horizon: int
    origin_timestamp: datetime
    target_timestamp: datetime
    predicted: float
    actual: float
    residual: float


@dataclass(frozen=True, slots=True)
class ForecastHorizonSummary:
    horizon: int
    observation_count: int
    missing_target_count: int
    rmse: float
    mae: float
    mape: float | None
    r2: float | None
    mape_unavailable_reason: str | None
    r2_unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class ForecastEvidence:
    status: str
    contract: ForecastContract
    observations: tuple[ForecastObservation, ...]
    summaries: tuple[ForecastHorizonSummary, ...]
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"available", "not_available"}:
            raise ValueError("forecast status must be available or not_available")
        if self.status == "available" and not self.observations:
            raise ValueError("available forecast evidence requires observations")
        if self.status == "not_available" and not self.unavailable_reason:
            raise ValueError("unavailable forecast evidence requires a reason")
        if tuple(summary.horizon for summary in self.summaries) != self.contract.horizons:
            raise ValueError("forecast summaries must cover the exact configured horizons")

    @property
    def metric_points(self) -> tuple[MetricPoint, ...]:
        """Return raw histories and aggregates through standard Model Metrics."""

        if self.status != "available":
            return ()
        points: list[MetricPoint] = []
        for observation_index, observation in enumerate(self.observations, start=1):
            timestamp_ms = _timestamp_ms(observation.target_timestamp)
            prefix = f"predictive_forecast_h{observation.horizon}"
            for suffix, value in (
                ("predicted", observation.predicted),
                ("actual", observation.actual),
                ("residual", observation.residual),
            ):
                points.append(
                    MetricPoint(
                        key=f"{prefix}_{suffix}",
                        value=value,
                        step=observation_index,
                        timestamp_ms=timestamp_ms,
                    )
                )
        for summary in self.summaries:
            timestamp_ms = (
                _timestamp_ms(self.observations[-1].target_timestamp) if self.observations else 0
            )
            prefix = f"predictive_h{summary.horizon}"
            summary_values: tuple[tuple[str, float | None], ...] = (
                ("observation_count", float(summary.observation_count)),
                ("missing_target_count", float(summary.missing_target_count)),
                ("rmse", summary.rmse),
                ("mae", summary.mae),
                ("mape", summary.mape),
                ("r2", summary.r2),
            )
            for suffix, summary_value in summary_values:
                if summary_value is not None:
                    points.append(
                        MetricPoint(
                            key=f"{prefix}_{suffix}",
                            value=float(summary_value),
                            step=0,
                            timestamp_ms=timestamp_ms,
                        )
                    )
        result = tuple(sorted(points, key=lambda point: (point.key, point.step)))
        validate_metric_points(result)
        return result

    @property
    def canonical_json(self) -> str:
        payload = {
            "status": self.status,
            "contract": asdict(self.contract),
            "observations": [
                {
                    "horizon": item.horizon,
                    "origin_timestamp": item.origin_timestamp.isoformat().replace("+00:00", "Z"),
                    "target_timestamp": item.target_timestamp.isoformat().replace("+00:00", "Z"),
                    "predicted_hex": item.predicted.hex(),
                    "actual_hex": item.actual.hex(),
                    "residual_hex": item.residual.hex(),
                }
                for item in self.observations
            ],
            "summaries": [asdict(summary) for summary in self.summaries],
            "unavailable_reason": self.unavailable_reason,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @property
    def source_hash(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()


def _validate_inputs(
    timestamps: Sequence[datetime],
    probabilities: np.ndarray,
    transition: np.ndarray,
    state_target_means: np.ndarray,
    target_values: Sequence[float | None],
) -> None:
    if len(timestamps) != len(target_values) or len(timestamps) != probabilities.shape[0]:
        raise ValueError("forecast timestamps, probabilities and targets must have equal length")
    if not timestamps:
        raise ValueError("forecast inputs cannot be empty")
    if any(left >= right for left, right in pairwise(timestamps)):
        raise ValueError("forecast timestamps must be strictly increasing")
    for timestamp in timestamps:
        _timestamp_ms(timestamp)
    if transition.shape != (probabilities.shape[1], probabilities.shape[1]):
        raise ValueError("forecast transition matrix must be square and match state count")
    if state_target_means.shape != (probabilities.shape[1],):
        raise ValueError("state target means must match state count")
    if not np.all(np.isfinite(transition)) or np.any(transition < 0.0):
        raise ValueError("forecast transition matrix must be finite and non-negative")
    if not np.allclose(transition.sum(axis=1), 1.0, rtol=0.0, atol=_PROBABILITY_TOLERANCE):
        raise ValueError("forecast transition rows must sum to one")
    if not np.all(np.isfinite(state_target_means)):
        raise ValueError("state target means must be finite")
    for value in target_values:
        if value is not None and not isfinite(value):
            raise ValueError("forecast target values must be finite or None")


def build_predictive_forecast_evidence(
    *,
    timestamps: Sequence[datetime],
    filtered_probabilities: Sequence[Sequence[float]],
    transition_matrix: Sequence[Sequence[float]],
    state_target_means: Sequence[float],
    target_values: Sequence[float | None],
    contract: ForecastContract,
) -> ForecastEvidence:
    """Build forecasts using only the filtered origin state and frozen model.

    A target at ``origin + horizon`` is used only as the observed value.  It
    cannot affect the prediction, scaling, state probabilities or transition
    matrix.  Horizons count retained model observations rather than calendar
    days, matching the HMM inference contract.
    """

    probabilities = np.asarray(filtered_probabilities, dtype=np.float64)
    transition = np.asarray(transition_matrix, dtype=np.float64)
    means = np.asarray(state_target_means, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] < 1:
        raise ValueError("filtered probabilities must be a non-empty T x K matrix")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("filtered probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=_PROBABILITY_TOLERANCE):
        raise ValueError("filtered probability rows must sum to one")
    _validate_inputs(timestamps, probabilities, transition, means, target_values)

    observations: list[ForecastObservation] = []
    summaries: list[ForecastHorizonSummary] = []
    for horizon in contract.horizons:
        horizon_observations: list[ForecastObservation] = []
        missing = 0
        transition_power = np.linalg.matrix_power(transition, horizon)
        for origin, target_index in enumerate(range(horizon, len(timestamps))):
            target = target_values[target_index]
            if target is None:
                missing += 1
                continue
            predicted = float(probabilities[origin] @ transition_power @ means)
            residual = float(target - predicted)
            horizon_observations.append(
                ForecastObservation(
                    horizon=horizon,
                    origin_timestamp=timestamps[origin],
                    target_timestamp=timestamps[target_index],
                    predicted=predicted,
                    actual=float(target),
                    residual=residual,
                )
            )
        observations.extend(horizon_observations)
        if horizon_observations:
            actual = np.asarray([item.actual for item in horizon_observations], dtype=np.float64)
            residual_values = np.asarray(
                [item.residual for item in horizon_observations], dtype=np.float64
            )
            rmse = float(sqrt(np.mean(residual_values**2)))
            mae = float(np.mean(np.abs(residual_values)))
            if np.any(actual == 0.0):
                mape = None
                mape_reason = "MAPE is undefined because at least one target is zero"
            else:
                mape = float(np.mean(np.abs(residual / actual)) * 100.0)
                mape_reason = None
            centered = actual - float(np.mean(actual))
            total = float(np.dot(centered, centered))
            if total == 0.0:
                r2 = None
                r2_reason = "R2 is undefined because observed targets have zero variance"
            else:
                r2 = float(1.0 - np.dot(residual_values, residual_values) / total)
                r2_reason = None
        else:
            rmse = mae = 0.0
            mape = r2 = None
            mape_reason = r2_reason = "no complete target horizon is available"
        summaries.append(
            ForecastHorizonSummary(
                horizon=horizon,
                observation_count=len(horizon_observations),
                missing_target_count=missing,
                rmse=rmse,
                mae=mae,
                mape=mape,
                r2=r2,
                mape_unavailable_reason=mape_reason,
                r2_unavailable_reason=r2_reason,
            )
        )
    if not observations:
        return ForecastEvidence(
            status="not_available",
            contract=contract,
            observations=(),
            summaries=tuple(summaries),
            unavailable_reason="no complete target horizon is available",
        )
    return ForecastEvidence(
        status="available",
        contract=contract,
        observations=tuple(observations),
        summaries=tuple(summaries),
    )


__all__ = [
    "ForecastContract",
    "ForecastEvidence",
    "ForecastHorizonSummary",
    "ForecastObservation",
    "build_predictive_forecast_evidence",
]
