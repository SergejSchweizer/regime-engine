"""Optional external-label quality metrics for v4 state assignments."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from itertools import pairwise
from math import fsum, isfinite, log

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from market_regime_engine.mlflow_support.metric_catalog import validate_metric_points
from market_regime_engine.mlflow_support.ports import MetricPoint

_PROBABILITY_TOLERANCE = 1.0e-10


@dataclass(frozen=True, slots=True)
class ClassificationContract:
    """Identity and alignment rules for optional external labels."""

    label_identity: str
    label_vocabulary: tuple[str, ...]
    unknown_label_policy: str = "omit_unknown"
    state_id_policy: str = "model_version_local_persistent_ids"
    selection_policy: str = "diagnostic_only"

    def __post_init__(self) -> None:
        if not self.label_identity or self.label_identity.strip() != self.label_identity:
            raise ValueError("label identity must be non-empty and trimmed")
        if not self.label_vocabulary or len(set(self.label_vocabulary)) != len(
            self.label_vocabulary
        ):
            raise ValueError("label vocabulary must be non-empty and duplicate-free")
        if any(not label or label.strip() != label for label in self.label_vocabulary):
            raise ValueError("label vocabulary entries must be non-empty and trimmed")
        if self.unknown_label_policy not in {"omit_unknown", "fail_closed"}:
            raise ValueError("unsupported unknown-label policy")
        if self.state_id_policy != "model_version_local_persistent_ids":
            raise ValueError("state IDs must remain model-version-local persistent IDs")
        if self.selection_policy != "diagnostic_only":
            raise ValueError("labeled metrics cannot be used for unsupervised selection")


@dataclass(frozen=True, slots=True)
class ClassificationEvidence:
    status: str
    contract: ClassificationContract
    timestamps: tuple[datetime, ...]
    state_count: int
    state_ids: tuple[int, ...]
    labels: tuple[str, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    hard_ari: float | None
    hard_nmi: float | None
    hard_accuracy: float | None
    hard_purity: float | None
    soft_nmi: float | None
    state_to_label_assignment: tuple[tuple[int, str], ...]
    omitted_unknown_count: int
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"available", "not_available"}:
            raise ValueError("classification status must be available or not_available")
        if self.status == "not_available":
            if self.unavailable_reason is None or any(
                value is not None
                for value in (
                    self.hard_ari,
                    self.hard_nmi,
                    self.hard_accuracy,
                    self.hard_purity,
                    self.soft_nmi,
                )
            ):
                raise ValueError("unavailable classification evidence cannot contain scores")
            return
        if not self.labels or len(self.labels) != len(self.timestamps):
            raise ValueError("available classification evidence requires aligned labels")
        scores = (self.hard_ari, self.hard_nmi, self.hard_accuracy, self.hard_purity, self.soft_nmi)
        if any(value is None or not isfinite(value) for value in scores):
            raise ValueError("available classification evidence requires finite scores")

    @property
    def metric_points(self) -> tuple[MetricPoint, ...]:
        if self.status != "available":
            return ()
        timestamp_ms = int(self.timestamps[-1].timestamp() * 1000)
        values = {
            "classification_hard_ari": self.hard_ari,
            "classification_hard_nmi": self.hard_nmi,
            "classification_hard_accuracy": self.hard_accuracy,
            "classification_hard_purity": self.hard_purity,
            "classification_soft_nmi": self.soft_nmi,
            "classification_label_count": float(len(self.labels)),
            "classification_omitted_unknown_count": float(self.omitted_unknown_count),
        }
        points = tuple(
            MetricPoint(key=key, value=float(value), step=0, timestamp_ms=timestamp_ms)
            for key, value in sorted(values.items())
            if value is not None
        )
        validate_metric_points(points)
        return points

    @property
    def canonical_json(self) -> str:
        payload = asdict(self)
        payload["timestamps"] = [
            value.isoformat().replace("+00:00", "Z") for value in self.timestamps
        ]
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @property
    def source_hash(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()


def _validate_timestamps(timestamps: Sequence[datetime]) -> None:
    if not timestamps:
        raise ValueError("classification timestamps cannot be empty")
    if any(left >= right for left, right in pairwise(timestamps)):
        raise ValueError("classification timestamps must be strictly increasing")
    for timestamp in timestamps:
        if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(timestamp):
            raise ValueError("classification timestamps must be timezone-aware UTC")


def _nmi_from_joint(joint: np.ndarray) -> float:
    left = joint.sum(axis=1)
    right = joint.sum(axis=0)
    mutual_information = fsum(
        float(value) * log(float(value) / float(left[row] * right[column]))
        for row in range(joint.shape[0])
        for column, value in enumerate(joint[row])
        if value > 0.0
    )
    left_entropy = -fsum(float(value) * log(float(value)) for value in left if value > 0.0)
    right_entropy = -fsum(float(value) * log(float(value)) for value in right if value > 0.0)
    denominator = left_entropy + right_entropy
    if denominator == 0.0:
        return 1.0
    return float(2.0 * mutual_information / denominator)


def _adjusted_rand_index(
    labels: np.ndarray, states: np.ndarray, class_count: int, state_count: int
) -> float:
    matrix = np.zeros((class_count, state_count), dtype=np.int64)
    np.add.at(matrix, (labels, states), 1)

    def comb2(values: np.ndarray) -> np.ndarray:
        return np.asarray(values * (values - 1) // 2)

    index = int(np.sum(comb2(matrix)))
    row_sum = int(np.sum(comb2(matrix.sum(axis=1))))
    column_sum = int(np.sum(comb2(matrix.sum(axis=0))))
    total = len(labels) * (len(labels) - 1) // 2
    if total == 0:
        return 1.0
    expected = row_sum * column_sum / total
    maximum = 0.5 * (row_sum + column_sum)
    denominator = maximum - expected
    if denominator == 0.0:
        return 1.0 if index == maximum else 0.0
    return float((index - expected) / denominator)


def build_state_classification_evidence(
    *,
    timestamps: Sequence[datetime],
    filtered_probabilities: Sequence[Sequence[float]],
    labels: Sequence[str | None] | None,
    contract: ClassificationContract,
    state_ids: Sequence[int] | None = None,
) -> ClassificationEvidence:
    """Compare optional labels with hard and soft model state assignments."""

    _validate_timestamps(timestamps)
    probabilities = np.asarray(filtered_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(timestamps):
        raise ValueError("classification posterior rows must match timestamps")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("classification posterior rows must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=_PROBABILITY_TOLERANCE):
        raise ValueError("classification posterior rows must sum to one")
    state_count = probabilities.shape[1]
    persistent_ids = tuple(range(state_count)) if state_ids is None else tuple(state_ids)
    if persistent_ids != tuple(range(state_count)):
        raise ValueError("state IDs must be canonical model-local state_0..state_(K-1) ordinals")
    if labels is None:
        return ClassificationEvidence(
            status="not_available",
            contract=contract,
            timestamps=tuple(timestamps),
            state_count=state_count,
            state_ids=persistent_ids,
            labels=(),
            confusion_matrix=(),
            hard_ari=None,
            hard_nmi=None,
            hard_accuracy=None,
            hard_purity=None,
            soft_nmi=None,
            state_to_label_assignment=(),
            omitted_unknown_count=0,
            unavailable_reason="external labels were not supplied",
        )
    if len(labels) != len(timestamps):
        raise ValueError("classification labels must align one-to-one with timestamps")
    vocabulary = {label: index for index, label in enumerate(contract.label_vocabulary)}
    known_indices: list[int] = []
    known_labels: list[str] = []
    omitted = 0
    for label in labels:
        if label in vocabulary:
            known_indices.append(vocabulary[label])
            known_labels.append(label)
        elif contract.unknown_label_policy == "fail_closed":
            raise ValueError(f"unknown classification label: {label!r}")
        else:
            omitted += 1
    if not known_indices:
        return ClassificationEvidence(
            status="not_available",
            contract=contract,
            timestamps=tuple(timestamps),
            state_count=state_count,
            state_ids=persistent_ids,
            labels=(),
            confusion_matrix=(),
            hard_ari=None,
            hard_nmi=None,
            hard_accuracy=None,
            hard_purity=None,
            soft_nmi=None,
            state_to_label_assignment=(),
            omitted_unknown_count=omitted,
            unavailable_reason="no labels remain after unknown-label policy",
        )
    keep = np.asarray([label in vocabulary for label in labels], dtype=bool)
    hard_states = np.argmax(probabilities[keep], axis=1).astype(np.intp)
    label_indices = np.asarray(known_indices, dtype=np.intp)
    class_count = len(contract.label_vocabulary)
    confusion = np.zeros((class_count, state_count), dtype=np.int64)
    np.add.at(confusion, (label_indices, hard_states), 1)
    joint_soft = np.zeros((class_count, state_count), dtype=np.float64)
    for label_index, row in zip(label_indices, probabilities[keep], strict=True):
        joint_soft[label_index] += row / len(label_indices)
    hard_joint = confusion.astype(np.float64) / len(label_indices)
    row_indices, column_indices = linear_sum_assignment(-confusion)
    assignment = tuple(
        (int(column), contract.label_vocabulary[int(row)])
        for row, column in sorted(
            zip(row_indices, column_indices, strict=True), key=lambda item: item[1]
        )
    )
    matched = sum(
        int(confusion[row, column]) for row, column in zip(row_indices, column_indices, strict=True)
    )
    purity = float(np.sum(np.max(confusion, axis=0)) / len(label_indices))
    return ClassificationEvidence(
        status="available",
        contract=contract,
        timestamps=tuple(
            timestamp for timestamp, selected in zip(timestamps, keep, strict=True) if selected
        ),
        state_count=state_count,
        state_ids=persistent_ids,
        labels=tuple(known_labels),
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in confusion),
        hard_ari=_adjusted_rand_index(label_indices, hard_states, class_count, state_count),
        hard_nmi=_nmi_from_joint(hard_joint),
        hard_accuracy=float(matched / len(label_indices)),
        hard_purity=purity,
        soft_nmi=_nmi_from_joint(joint_soft),
        state_to_label_assignment=assignment,
        omitted_unknown_count=omitted,
    )


__all__ = [
    "ClassificationContract",
    "ClassificationEvidence",
    "build_state_classification_evidence",
]
