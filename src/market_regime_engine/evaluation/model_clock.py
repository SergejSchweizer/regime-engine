"""Pure complete-case feasibility checks for v4 model clocks."""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise
from typing import Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan
from market_regime_engine.feature_discovery.contracts import (
    MIN_FEATURE_VARIANCE,
    MIN_MODEL_TEST_OBSERVATIONS,
    MIN_MODEL_TRAIN_OBSERVATIONS,
    DiscoveryStatus,
    ModelClockFold,
    ModelClockPreflight,
)

_TIMESTAMP_COLUMN = "timestamp_m1"
SelectionScope = Literal["prototype", "prefix"]


def _utc(value: object, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != UTC.utcoffset(value)
    ):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return value


def _validate_feature_order(feature_order: tuple[str, ...]) -> None:
    if (
        not isinstance(feature_order, tuple)
        or not feature_order
        or any(not isinstance(name, str) or not name.strip() for name in feature_order)
        or len(set(feature_order)) != len(feature_order)
    ):
        raise ValueError("feature_order must be a non-empty duplicate-free tuple")


def _validate_thresholds(
    minimum_model_train_observations: int,
    minimum_model_test_observations: int,
    minimum_valid_fold_rate: float,
) -> None:
    if (
        not isinstance(minimum_model_train_observations, int)
        or isinstance(minimum_model_train_observations, bool)
        or minimum_model_train_observations < 1
        or not isinstance(minimum_model_test_observations, int)
        or isinstance(minimum_model_test_observations, bool)
        or minimum_model_test_observations < 1
    ):
        raise ValueError("model-row thresholds must be positive integers")
    if not np.isfinite(minimum_valid_fold_rate) or not 0.0 <= minimum_valid_fold_rate <= 1.0:
        raise ValueError("minimum_valid_fold_rate must be finite in [0,1]")


def _source_matrix(
    source_rows: pd.DataFrame, feature_order: tuple[str, ...]
) -> tuple[tuple[datetime, ...], np.ndarray, np.ndarray]:
    required = (_TIMESTAMP_COLUMN, *feature_order)
    missing = tuple(column for column in required if column not in source_rows.columns)
    if missing:
        raise ValueError(f"source rows are missing required columns: {missing}")
    timestamps = tuple(_utc(value, _TIMESTAMP_COLUMN) for value in source_rows[_TIMESTAMP_COLUMN])
    if any(current <= previous for previous, current in pairwise(timestamps)):
        raise ValueError("source timestamps must be strictly increasing and unique")
    try:
        values = source_rows.loc[:, list(feature_order)].to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("model-clock feature rows must be numeric") from error
    complete = source_rows.loc[:, list(feature_order)].notna().all(axis=1).to_numpy(dtype=bool)
    complete &= np.isfinite(values).all(axis=1)
    return timestamps, values, complete


def _window_indices(
    timestamps: tuple[datetime, ...],
    train_start: datetime,
    train_end: datetime,
    test_start: datetime,
    test_end: datetime,
    fold_id: str,
    train_source_observations: int,
    test_source_observations: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    train = tuple(
        index for index, timestamp in enumerate(timestamps) if train_start <= timestamp <= train_end
    )
    test = tuple(
        index for index, timestamp in enumerate(timestamps) if test_start <= timestamp <= test_end
    )
    if len(train) != train_source_observations or len(test) != test_source_observations:
        raise ValueError(f"{fold_id} source-row counts do not match the walk-forward plan")
    if (
        not train
        or not test
        or (
            timestamps[train[0]],
            timestamps[train[-1]],
            timestamps[test[0]],
            timestamps[test[-1]],
        )
        != (train_start, train_end, test_start, test_end)
    ):
        raise ValueError(f"{fold_id} source timestamps do not match the walk-forward plan")
    return train, test


def _first_train_variances(
    values: np.ndarray,
    complete: np.ndarray,
    train_indices: tuple[int, ...],
    feature_order: tuple[str, ...],
) -> tuple[int, tuple[tuple[str, float], ...]]:
    retained_indices = tuple(index for index in train_indices if complete[index])
    if not retained_indices:
        return 0, tuple((name, 0.0) for name in feature_order)
    retained = values[np.asarray(retained_indices, dtype=np.int64)]
    variances = np.var(retained, axis=0, ddof=0)
    return len(retained_indices), tuple(
        (name, float(variance)) for name, variance in zip(feature_order, variances, strict=True)
    )


def build_model_clock_preflight(
    source_rows: pd.DataFrame,
    feature_order: tuple[str, ...],
    plan: WalkForwardPlan,
    *,
    minimum_model_train_observations: int = MIN_MODEL_TRAIN_OBSERVATIONS,
    minimum_model_test_observations: int = MIN_MODEL_TEST_OBSERVATIONS,
    minimum_valid_fold_rate: float = 0.80,
) -> ModelClockPreflight:
    """Record common complete-case counts for every planned fold.

    The function performs no fitting, scaling, filling, feature dropping, or
    K-dependent work.  Non-finite/missing rows simply leave the common clock.
    Callers must use :func:`require_model_clock_eligible` to make the intended
    prototype-versus-prefix failure policy explicit.
    """

    if not isinstance(source_rows, pd.DataFrame):
        raise TypeError("source_rows must be a pandas DataFrame")
    if not isinstance(plan, WalkForwardPlan):
        raise TypeError("plan must be a WalkForwardPlan")
    _validate_feature_order(feature_order)
    _validate_thresholds(
        minimum_model_train_observations,
        minimum_model_test_observations,
        minimum_valid_fold_rate,
    )
    if not plan.folds:
        raise ValueError("model-clock preflight requires at least one planned fold")

    timestamps, values, complete = _source_matrix(source_rows, feature_order)
    fold_evidence: list[ModelClockFold] = []
    first_train_count: int | None = None
    first_train_variances: tuple[tuple[str, float], ...] | None = None
    for fold_index, fold in enumerate(plan.folds):
        train_indices, test_indices = _window_indices(
            timestamps,
            fold.train_start,
            fold.train_end,
            fold.test_start,
            fold.test_end,
            fold.fold_id,
            fold.train_source_observations,
            fold.test_source_observations,
        )
        train_count = int(sum(complete[index] for index in train_indices))
        test_count = int(sum(complete[index] for index in test_indices))
        if fold_index == 0:
            first_train_count, first_train_variances = _first_train_variances(
                values, complete, train_indices, feature_order
            )
        reasons: list[str] = []
        if train_count < minimum_model_train_observations:
            reasons.append(
                f"TRAIN complete observations {train_count} below "
                f"{minimum_model_train_observations}"
            )
        if test_count < minimum_model_test_observations:
            reasons.append(
                f"TEST complete observations {test_count} below {minimum_model_test_observations}"
            )
        invalid_reason = "; ".join(reasons) or None
        fold_evidence.append(
            ModelClockFold(
                fold_id=fold.fold_id,
                train_start=fold.train_start,
                train_end=fold.train_end,
                test_start=fold.test_start,
                test_end=fold.test_end,
                train_complete_observations=train_count,
                test_complete_observations=test_count,
                structurally_valid=invalid_reason is None,
                invalid_reason=invalid_reason,
            )
        )

    assert first_train_count is not None
    assert first_train_variances is not None
    valid_rate = sum(fold.structurally_valid for fold in fold_evidence) / len(fold_evidence)
    reasons = []
    if first_train_count < minimum_model_train_observations:
        reasons.append(
            f"first TRAIN complete observations {first_train_count} below "
            f"{minimum_model_train_observations}"
        )
    low_variance = tuple(
        name for name, variance in first_train_variances if variance <= MIN_FEATURE_VARIANCE
    )
    if low_variance:
        reasons.append(f"first TRAIN feature variance is not greater than 1e-12: {low_variance}")
    if valid_rate < minimum_valid_fold_rate:
        reasons.append(f"structural valid-fold rate {valid_rate} below {minimum_valid_fold_rate}")
    return ModelClockPreflight(
        feature_order=feature_order,
        plan_hash=plan.plan_hash,
        first_train_complete_observations=first_train_count,
        first_train_feature_variances=first_train_variances,
        folds=tuple(fold_evidence),
        structural_valid_fold_rate=valid_rate,
        status=DiscoveryStatus.VALID if not reasons else DiscoveryStatus.INVALID,
        invalid_reason="; ".join(reasons) or None,
    )


def require_model_clock_eligible(
    preflight: ModelClockPreflight,
    selection_scope: SelectionScope,
) -> None:
    """Make invalid-clock caller behavior explicit for prototype or prefix work."""

    if selection_scope not in ("prototype", "prefix"):
        raise ValueError("selection_scope must be 'prototype' or 'prefix'")
    if preflight.status is DiscoveryStatus.INVALID:
        action = "outer selection" if selection_scope == "prototype" else "this prefix"
        raise ValueError(
            f"{action} is ineligible because the model clock is invalid: {preflight.invalid_reason}"
        )


preflight_model_clock = build_model_clock_preflight
check_model_clock = build_model_clock_preflight


__all__ = [
    "SelectionScope",
    "build_model_clock_preflight",
    "check_model_clock",
    "preflight_model_clock",
    "require_model_clock_eligible",
]
