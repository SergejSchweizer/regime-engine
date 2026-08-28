"""Immutable, evaluation-scoped complete-case observation clocks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from itertools import pairwise

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.walk_forward_splits import WalkForwardFold, WalkForwardPlan
from market_regime_engine.evaluations.contracts import EvaluationId, FeatureSpec

_TIMESTAMP_COLUMN = "timestamp_m1"


def _utc(value: object, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != UTC.utcoffset(value)
    ):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return value


@dataclass(frozen=True, slots=True)
class ClockFoldEvidence:
    fold_id: str
    retained_train_timestamps: tuple[datetime, ...]
    skipped_train_timestamps: tuple[datetime, ...]
    retained_test_timestamps: tuple[datetime, ...]
    skipped_test_timestamps: tuple[datetime, ...]

    def __post_init__(self) -> None:
        if not self.fold_id:
            raise ValueError("clock fold_id must be non-empty")
        for field_name in (
            "retained_train_timestamps",
            "skipped_train_timestamps",
            "retained_test_timestamps",
            "skipped_test_timestamps",
        ):
            timestamps = getattr(self, field_name)
            if any(
                current <= previous
                for previous, current in pairwise(_utc(value, field_name) for value in timestamps)
            ):
                raise ValueError(f"{field_name} must be strictly increasing")


@dataclass(frozen=True, slots=True)
class EvaluationClock:
    evaluation_id: EvaluationId
    feature_order: tuple[str, ...]
    retained_mask_hash: str
    fold_evidence: tuple[ClockFoldEvidence, ...]

    def __post_init__(self) -> None:
        if not self.feature_order or len(self.feature_order) != len(set(self.feature_order)):
            raise ValueError("clock feature_order must be non-empty and duplicate-free")
        if len(self.retained_mask_hash) != 64:
            raise ValueError("retained_mask_hash must be a SHA-256 digest")
        if tuple(item.fold_id for item in self.fold_evidence) != tuple(
            sorted(item.fold_id for item in self.fold_evidence)
        ):
            raise ValueError("clock fold evidence must be ordered by fold ID")

    @property
    def clock_hash(self) -> str:
        return self.retained_mask_hash


def _validate_source_rows(
    source_rows: pd.DataFrame, feature_order: tuple[str, ...]
) -> tuple[datetime, ...]:
    required = (_TIMESTAMP_COLUMN, *feature_order)
    missing = tuple(column for column in required if column not in source_rows.columns)
    if missing:
        raise ValueError(f"source rows are missing required columns: {missing}")
    timestamps = tuple(_utc(value, _TIMESTAMP_COLUMN) for value in source_rows[_TIMESTAMP_COLUMN])
    if any(current <= previous for previous, current in pairwise(timestamps)):
        raise ValueError("source timestamps must be strictly increasing and unique")
    return timestamps


def _window_indices(
    timestamps: tuple[datetime, ...], fold: WalkForwardFold
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(timestamps, dtype=object)
    train = np.flatnonzero((values >= fold.train_start) & (values <= fold.train_end))
    test = np.flatnonzero((values >= fold.test_start) & (values <= fold.test_end))
    if len(train) != fold.train_source_observations or len(test) != fold.test_source_observations:
        raise ValueError(f"{fold.fold_id} source-row counts do not match planned fold")
    if (
        timestamps[train[0]],
        timestamps[train[-1]],
        timestamps[test[0]],
        timestamps[test[-1]],
    ) != (fold.train_start, fold.train_end, fold.test_start, fold.test_end):
        raise ValueError(f"{fold.fold_id} source timestamps do not match evaluation plan")
    return train, test


def _retained_mask(source_rows: pd.DataFrame, feature_order: tuple[str, ...]) -> np.ndarray:
    selected = source_rows.loc[:, list(feature_order)]
    numeric = selected.apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
    return np.asarray(selected.notna().all(axis=1).to_numpy() & finite, dtype=bool)


def _mask_hash(
    evaluation_id: EvaluationId, feature_order: tuple[str, ...], retained: np.ndarray
) -> str:
    payload = {
        "evaluation_id": evaluation_id.value,
        "feature_order": feature_order,
        "retained": retained.astype(bool).tolist(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(canonical.encode()).hexdigest()


def build_evaluation_clock(
    source_rows: pd.DataFrame, plan: WalkForwardPlan, feature_spec: FeatureSpec
) -> EvaluationClock:
    """Build the complete-case clock for exactly one immutable evaluation feature spec."""

    timestamps = _validate_source_rows(source_rows, feature_spec.feature_order)
    if not plan.folds:
        raise ValueError("evaluation clock requires at least one planned fold")
    retained = _retained_mask(source_rows, feature_spec.feature_order)
    evidence: list[ClockFoldEvidence] = []
    for fold in plan.folds:
        train_indices, test_indices = _window_indices(timestamps, fold)
        evidence.append(
            ClockFoldEvidence(
                fold_id=fold.fold_id,
                retained_train_timestamps=tuple(
                    timestamps[index] for index in train_indices if retained[index]
                ),
                skipped_train_timestamps=tuple(
                    timestamps[index] for index in train_indices if not retained[index]
                ),
                retained_test_timestamps=tuple(
                    timestamps[index] for index in test_indices if retained[index]
                ),
                skipped_test_timestamps=tuple(
                    timestamps[index] for index in test_indices if not retained[index]
                ),
            )
        )
    return EvaluationClock(
        evaluation_id=feature_spec.evaluation_id,
        feature_order=feature_spec.feature_order,
        retained_mask_hash=_mask_hash(
            feature_spec.evaluation_id, feature_spec.feature_order, retained
        ),
        fold_evidence=tuple(evidence),
    )
