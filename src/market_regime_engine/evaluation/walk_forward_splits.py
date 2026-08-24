"""Deterministic expanding walk-forward source-row planner."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

from market_regime_engine.profiles.config import WalkForwardConfig

_MIN_TRAIN = 1260
_TEST = 63
_STEP = 63
_ALLOW_PARTIAL = False


def _utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return value


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold_index: int
    fold_id: str
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_source_observations: int
    test_source_observations: int

    def __post_init__(self) -> None:
        if self.fold_index < 1 or self.fold_id != f"fold_{self.fold_index:03d}":
            raise ValueError("fold identity must be deterministic and one-based")
        for field_name in ("train_start", "train_end", "test_start", "test_end"):
            _utc(getattr(self, field_name), field_name)
        if not self.train_start <= self.train_end < self.test_start <= self.test_end:
            raise ValueError("TRAIN and TEST bounds must be ordered and non-overlapping")
        if self.train_source_observations < _MIN_TRAIN:
            raise ValueError("TRAIN source observations are below the pinned minimum")
        if self.test_source_observations != _TEST:
            raise ValueError("TEST source observations must be exactly 63")


@dataclass(frozen=True, slots=True)
class WalkForwardPlan:
    folds: tuple[WalkForwardFold, ...]
    evaluation_cutoff: datetime | None
    plan_hash: str


def _assert_pinned(config: WalkForwardConfig) -> None:
    if (
        config.minimum_train_source_observations != _MIN_TRAIN
        or config.test_source_observations != _TEST
        or config.step_source_observations != _STEP
        or config.allow_partial_final_test is not _ALLOW_PARTIAL
    ):
        raise ValueError("walk-forward configuration differs from pinned Xetra plan")


def plan_walk_forward(
    timestamps: Sequence[datetime],
    config: WalkForwardConfig,
) -> WalkForwardPlan:
    """Plan complete expanding folds on the exact source-row observation sequence."""

    _assert_pinned(config)
    ordered = tuple(_utc(value, "source timestamp") for value in timestamps)
    if any(current <= previous for previous, current in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("source timestamps must be strictly increasing and unique")

    folds: list[WalkForwardFold] = []
    train_end_index = _MIN_TRAIN - 1
    test_start_index = train_end_index + 1
    fold_index = 1

    while test_start_index + _TEST <= len(ordered):
        test_end_index = test_start_index + _TEST - 1
        fold = WalkForwardFold(
            fold_index=fold_index,
            fold_id=f"fold_{fold_index:03d}",
            train_start=ordered[0],
            train_end=ordered[train_end_index],
            test_start=ordered[test_start_index],
            test_end=ordered[test_end_index],
            train_source_observations=train_end_index + 1,
            test_source_observations=_TEST,
        )
        folds.append(fold)
        fold_index += 1
        train_end_index += _STEP
        test_start_index += _STEP

    payload = {
        "allow_partial_final_test": _ALLOW_PARTIAL,
        "minimum_train_source_observations": _MIN_TRAIN,
        "step_source_observations": _STEP,
        "test_source_observations": _TEST,
        "folds": [
            {
                "fold_id": fold.fold_id,
                "fold_index": fold.fold_index,
                "test_end": fold.test_end.isoformat(),
                "test_start": fold.test_start.isoformat(),
                "train_end": fold.train_end.isoformat(),
                "train_start": fold.train_start.isoformat(),
            }
            for fold in folds
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    cutoff = folds[-1].test_end if folds else None
    return WalkForwardPlan(
        folds=tuple(folds),
        evaluation_cutoff=cutoff,
        plan_hash=sha256(canonical).hexdigest(),
    )
