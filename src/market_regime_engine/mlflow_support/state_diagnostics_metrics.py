"""State-posterior, transition, and emission diagnostics for v4 models.

The functions in this module are deliberately independent of MLflow.  They
turn one already-fitted, already-aligned fold into finite ``MetricPoint``
values and a canonical mapping artifact.  No source rows are read and no
model is fitted here, which makes the projection safe to run in parallel with
other LoggedModel projections.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite, log
from typing import Any

import numpy as np
import numpy.typing as npt

from market_regime_engine.mlflow_support.metric_catalog import validate_metric_points
from market_regime_engine.mlflow_support.ports import MetricPoint
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.states.alignment import StateAlignment

_PROBABILITY_TOLERANCE = 1.0e-10
_MIN_EIGENVALUE = 1.0e-12


def _utc_millis(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("diagnostic timestamps must be timezone-aware UTC")
    result = int(value.timestamp() * 1000)
    if result < 0:
        raise ValueError("diagnostic timestamps must be non-negative")
    return result


def _probability_matrix(
    values: npt.ArrayLike,
    *,
    state_count: int,
    name: str,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] != state_count:
        raise ValueError(f"{name} must be a non-empty T x K matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=_PROBABILITY_TOLERANCE):
        raise ValueError(f"{name} rows must sum to one within 1e-10")
    return matrix


def _aligned_matrix(values: np.ndarray, alignment: StateAlignment) -> np.ndarray:
    mapping = np.asarray(alignment.persistent_to_fitted, dtype=np.intp)
    if mapping.shape != (values.shape[1],) or not np.array_equal(
        np.sort(mapping), np.arange(values.shape[1])
    ):
        raise ValueError("state alignment is not a fitted-state permutation")
    return np.ascontiguousarray(values[:, mapping])


def _aligned_transition(
    artifact: GaussianHMMArtifact,
    alignment: StateAlignment,
) -> np.ndarray:
    transition = np.asarray(artifact.transition_matrix, dtype=np.float64)
    mapping = np.asarray(alignment.persistent_to_fitted, dtype=np.intp)
    aligned = np.ascontiguousarray(transition[np.ix_(mapping, mapping)])
    if not np.all(np.isfinite(aligned)) or np.any(aligned < 0.0):
        raise ValueError("aligned transition matrix must be finite and non-negative")
    if not np.allclose(aligned.sum(axis=1), 1.0, rtol=0.0, atol=_PROBABILITY_TOLERANCE):
        raise ValueError("aligned transition rows must sum to one within 1e-10")
    return aligned


def _mean_durations(viterbi: np.ndarray, state_count: int) -> tuple[float, ...]:
    if viterbi.ndim != 1 or viterbi.size == 0:
        raise ValueError("Viterbi states must be a non-empty vector")
    durations: list[list[int]] = [[] for _ in range(state_count)]
    starts = np.flatnonzero(np.r_[True, viterbi[1:] != viterbi[:-1]])
    ends = np.r_[starts[1:], viterbi.size]
    for start, end in zip(starts, ends, strict=True):
        state = int(viterbi[start])
        if state < 0 or state >= state_count:
            raise ValueError("Viterbi state index is outside the model state range")
        durations[state].append(int(end - start))
    return tuple(float(np.mean(values)) if values else 0.0 for values in durations)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0)
    result = -terms.sum(axis=1)
    if not np.all(np.isfinite(result)):
        raise ValueError("posterior entropy must be finite")
    return result


def _point(
    key: str,
    value: float,
    *,
    step: int,
    timestamp_ms: int,
) -> MetricPoint:
    if not isfinite(value):
        raise ValueError(f"state diagnostic {key} must be finite")
    return MetricPoint(key=key, value=float(value), step=step, timestamp_ms=timestamp_ms)


@dataclass(frozen=True, slots=True)
class StateDiagnosticEvidence:
    """Canonical state diagnostics for one model/fold."""

    fold_id: str
    model_family: str
    state_count: int
    feature_order: tuple[str, ...]
    persistent_to_fitted: tuple[int, ...]
    oos_timestamps: tuple[datetime, ...]
    posterior_probabilities: tuple[tuple[float, ...], ...]
    viterbi_states: tuple[int, ...]
    train_hard_occupancy: tuple[float, ...]
    train_soft_occupancy: tuple[float, ...]
    oos_hard_occupancy: tuple[float, ...]
    oos_soft_occupancy: tuple[float, ...]
    entropy: tuple[float, ...]
    confidence: tuple[float, ...]
    low_confidence_count: int
    expected_duration: tuple[float, ...]
    transition_matrix: tuple[tuple[float, ...], ...]
    transition_row_entropy: tuple[float, ...]
    emission_means: tuple[tuple[float, ...], ...]
    emission_variances: tuple[tuple[float, ...], ...]
    emission_covariances: tuple[tuple[tuple[float, ...], ...], ...]
    covariance_eigenvalues: tuple[tuple[float, ...], ...]
    covariance_condition_numbers: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.fold_id or self.state_count not in (2, 3, 4, 5):
            raise ValueError("state diagnostic identity is invalid")
        if len(self.persistent_to_fitted) != self.state_count or tuple(
            sorted(self.persistent_to_fitted)
        ) != tuple(range(self.state_count)):
            raise ValueError("persistent_to_fitted must be a state permutation")
        if len(self.oos_timestamps) != len(self.posterior_probabilities):
            raise ValueError("posterior timestamps and rows must have equal length")
        if len(self.viterbi_states) != len(self.posterior_probabilities):
            raise ValueError("Viterbi history and posterior history must have equal length")
        if any(
            left >= right
            for left, right in zip(self.oos_timestamps, self.oos_timestamps[1:], strict=False)
        ):
            raise ValueError("OOS diagnostic timestamps must be strictly increasing")
        if self.low_confidence_count < 0 or self.low_confidence_count > len(self.viterbi_states):
            raise ValueError("low-confidence count is outside its observation bounds")

    @property
    def mapping_artifact(self) -> dict[str, Any]:
        """Return state mapping metadata without implying cross-model identity."""

        return {
            "model_family": self.model_family,
            "state_count": self.state_count,
            "state_identity_scope": "model_version_local",
            "fold_id": self.fold_id,
            "persistent_to_fitted": list(self.persistent_to_fitted),
            "persistent_state_ids": [f"state_{index}" for index in range(self.state_count)],
        }

    @property
    def mapping_artifact_sha256(self) -> str:
        encoded = json.dumps(self.mapping_artifact, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return sha256(encoded).hexdigest()

    def metric_points(self) -> tuple[MetricPoint, ...]:
        """Return all finite diagnostics with deterministic key/step order."""

        points: list[MetricPoint] = []
        default_timestamp = _utc_millis(self.oos_timestamps[-1]) if self.oos_timestamps else 0

        def add(key: str, value: float, step: int, timestamp_ms: int = default_timestamp) -> None:
            points.append(_point(key, value, step=step, timestamp_ms=timestamp_ms))

        for state, value in enumerate(self.train_hard_occupancy):
            add(f"state_diag_train_hard_occupancy_state_{state}", value, 0)
        for state, value in enumerate(self.train_soft_occupancy):
            add(f"state_diag_train_soft_occupancy_state_{state}", value, 0)
        for state, value in enumerate(self.oos_hard_occupancy):
            add(f"state_diag_oos_hard_occupancy_state_{state}", value, 0)
        for state, value in enumerate(self.oos_soft_occupancy):
            add(f"state_diag_oos_soft_occupancy_state_{state}", value, 0)
        add("state_diag_low_confidence_count", float(self.low_confidence_count), 0)
        add(
            "state_diag_low_confidence_rate",
            self.low_confidence_count / len(self.viterbi_states),
            0,
        )
        for state, value in enumerate(self.expected_duration):
            add(f"state_diag_expected_duration_state_{state}", value, 0)
        for state, value in enumerate(self.entropy):
            add(
                "state_diag_posterior_entropy",
                value,
                state,
                _utc_millis(self.oos_timestamps[state]),
            )
        for state, value in enumerate(self.confidence):
            add(
                "state_diag_posterior_confidence",
                value,
                state,
                _utc_millis(self.oos_timestamps[state]),
            )
        for observation, (timestamp, probabilities, viterbi) in enumerate(
            zip(
                self.oos_timestamps,
                self.posterior_probabilities,
                self.viterbi_states,
                strict=True,
            ),
            start=1,
        ):
            timestamp_ms = _utc_millis(timestamp)
            for state, value in enumerate(probabilities):
                add(
                    f"state_diag_posterior_probability_state_{state}",
                    value,
                    observation,
                    timestamp_ms,
                )
            add("state_diag_viterbi_state", float(viterbi), observation, timestamp_ms)
        for row, values in enumerate(self.transition_matrix):
            add(
                f"state_diag_transition_row_entropy_state_{row}",
                self.transition_row_entropy[row],
                0,
            )
            for column, value in enumerate(values):
                add(f"state_diag_transition_probability_state_{row}_to_state_{column}", value, 0)
        for state, (mean, variance) in enumerate(
            zip(self.emission_means, self.emission_variances, strict=True)
        ):
            for feature, value in enumerate(mean):
                add(f"state_diag_emission_mean_state_{state}_feature_{feature}", value, 0)
            for feature, value in enumerate(variance):
                add(f"state_diag_emission_variance_state_{state}_feature_{feature}", value, 0)
        for state, eigenvalues in enumerate(self.covariance_eigenvalues):
            for eigenvalue, value in enumerate(eigenvalues):
                add(f"state_diag_covariance_eigenvalue_state_{state}_{eigenvalue}", value, 0)
            add(
                f"state_diag_covariance_condition_number_state_{state}",
                self.covariance_condition_numbers[state],
                0,
            )
        result = tuple(
            sorted(points, key=lambda point: (point.key, point.step, point.timestamp_ms))
        )
        validate_metric_points(result)
        return result

    def canonical_json(self) -> str:
        payload = asdict(self)
        payload["oos_timestamps"] = [
            value.isoformat().replace("+00:00", "Z") for value in self.oos_timestamps
        ]
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_state_diagnostic_evidence(
    *,
    fold_id: str,
    artifact: GaussianHMMArtifact,
    alignment: StateAlignment,
    oos_timestamps: tuple[datetime, ...],
    oos_filtered_probabilities: npt.ArrayLike,
    train_hard_occupancy: tuple[float, ...],
    train_soft_occupancy: tuple[float, ...],
    oos_hard_occupancy: tuple[float, ...] | None = None,
    oos_soft_occupancy: tuple[float, ...] | None = None,
) -> StateDiagnosticEvidence:
    """Build aligned diagnostics from one already-fitted fold.

    Only OOS posterior rows are retained by the walk-forward contract.  TRAIN
    occupancy is therefore supplied by the fold result, while OOS occupancy
    is independently recomputed from the OOS posterior matrix.
    """

    if len(alignment.persistent_to_fitted) != artifact.state_count:
        raise ValueError("state alignment dimension differs from artifact")
    probabilities = _aligned_matrix(
        _probability_matrix(
            oos_filtered_probabilities,
            state_count=artifact.state_count,
            name="OOS filtered probabilities",
        ),
        alignment,
    )
    if len(oos_timestamps) != probabilities.shape[0]:
        raise ValueError("OOS timestamps must match posterior rows")
    for timestamp in oos_timestamps:
        _utc_millis(timestamp)
    train_hard = tuple(float(value) for value in train_hard_occupancy)
    train_soft = tuple(float(value) for value in train_soft_occupancy)
    if len(train_hard) != artifact.state_count or len(train_soft) != artifact.state_count:
        raise ValueError("TRAIN occupancy vectors must match state count")
    if any(not isfinite(value) or value < 0.0 for value in (*train_hard, *train_soft)):
        raise ValueError("TRAIN occupancy values must be finite and non-negative")
    oos_hard_values = np.mean(
        np.argmax(probabilities, axis=1)[:, None] == np.arange(artifact.state_count),
        axis=0,
    )
    oos_soft_values = np.mean(probabilities, axis=0)
    # The fold stores aggregate occupancy as evidence, but the posterior rows
    # are the authoritative source for this family.  Older dossiers can carry
    # rounded aggregates, so they are deliberately not used to override the
    # exact vectorized recomputation above.

    entropy = _entropy(probabilities)
    confidence = np.max(probabilities, axis=1)
    viterbi = np.argmax(probabilities, axis=1).astype(np.intp)
    transition = _aligned_transition(artifact, alignment)
    transition_entropy = tuple(
        float(-sum(float(value) * log(float(value)) for value in row if value > 0.0))
        for row in transition
    )
    covariances = np.asarray(artifact.distribution_covariances(), dtype=np.float64)[
        list(alignment.persistent_to_fitted)
    ]
    if not np.all(np.isfinite(covariances)):
        raise ValueError("emission covariance values must be finite")
    symmetric = (covariances + np.swapaxes(covariances, 1, 2)) / 2.0
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if np.any(eigenvalues <= _MIN_EIGENVALUE):
        raise ValueError("emission covariance must be positive definite")
    condition_numbers = tuple(float(values[-1] / values[0]) for values in eigenvalues)
    return StateDiagnosticEvidence(
        fold_id=fold_id,
        model_family=artifact.model_family,
        state_count=artifact.state_count,
        feature_order=artifact.feature_order,
        persistent_to_fitted=tuple(alignment.persistent_to_fitted),
        oos_timestamps=oos_timestamps,
        posterior_probabilities=tuple(
            tuple(float(value) for value in row) for row in probabilities
        ),
        viterbi_states=tuple(int(value) for value in viterbi),
        train_hard_occupancy=train_hard,
        train_soft_occupancy=train_soft,
        oos_hard_occupancy=tuple(float(value) for value in oos_hard_values),
        oos_soft_occupancy=tuple(float(value) for value in oos_soft_values),
        entropy=tuple(float(value) for value in entropy),
        confidence=tuple(float(value) for value in confidence),
        low_confidence_count=int(np.count_nonzero(confidence < 0.60)),
        expected_duration=_mean_durations(viterbi, artifact.state_count),
        transition_matrix=tuple(tuple(float(value) for value in row) for row in transition),
        transition_row_entropy=transition_entropy,
        emission_means=tuple(
            tuple(float(value) for value in artifact.means[index])
            for index in alignment.persistent_to_fitted
        ),
        emission_variances=tuple(
            tuple(float(value) for value in np.diag(covariance)) for covariance in covariances
        ),
        emission_covariances=tuple(
            tuple(tuple(float(value) for value in row) for row in covariance)
            for covariance in covariances
        ),
        covariance_eigenvalues=tuple(
            tuple(float(value) for value in values) for values in eigenvalues
        ),
        covariance_condition_numbers=condition_numbers,
    )


__all__ = ["StateDiagnosticEvidence", "build_state_diagnostic_evidence"]
