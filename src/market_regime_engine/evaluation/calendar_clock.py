"""Canonical calendar-month walk-forward clock for the live evaluation cadence.

The clock is deliberately independent of feature fitting and model math.  It
only partitions an already ordered, timezone-aware source timestamp sequence
into expanding TRAIN windows and complete immediately-following calendar-month
TEST windows.  Local Berlin time is used for membership; original timestamp
objects are retained in every fold.
"""

from __future__ import annotations

import calendar
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
from itertools import pairwise
from zoneinfo import ZoneInfo

_DEFAULT_TIMEZONE = "Europe/Berlin"
_TIMESTAMP_FORMAT = "%Y-%m"
MIN_CALENDAR_MODEL_TEST_OBSERVATIONS = 1


def _aware(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value


def _month(value: datetime, timezone: ZoneInfo) -> str:
    local = value.astimezone(timezone)
    return local.strftime(_TIMESTAMP_FORMAT)


def _month_date(value: str) -> date:
    try:
        return datetime.strptime(value, _TIMESTAMP_FORMAT).date().replace(day=1)
    except ValueError as exc:
        raise ValueError(f"invalid calendar month: {value!r}") from exc


def _next_month(value: str) -> str:
    current = _month_date(value)
    if current.month == 12:
        return date(current.year + 1, 1, 1).strftime(_TIMESTAMP_FORMAT)
    return date(current.year, current.month + 1, 1).strftime(_TIMESTAMP_FORMAT)


@dataclass(frozen=True, slots=True)
class CalendarMonthFold:
    """One expanding TRAIN / immediately-following monthly TEST boundary."""

    fold_index: int
    fold_id: str
    train_through_month: str
    train_first_timestamp: datetime
    train_cutoff_timestamp: datetime
    test_calendar_month: str
    test_first_timestamp: datetime
    test_last_timestamp: datetime
    train_source_observations: int
    test_source_observations: int
    month_clock_hash: str

    def __post_init__(self) -> None:
        if self.fold_index < 1 or self.fold_id != f"fold_{self.fold_index:03d}":
            raise ValueError("calendar fold identity must be deterministic and one-based")
        _month_date(self.train_through_month)
        _month_date(self.test_calendar_month)
        if _next_month(self.train_through_month) != self.test_calendar_month:
            raise ValueError("TEST must be the immediately following calendar month")
        for value, name in (
            (self.train_first_timestamp, "train_first_timestamp"),
            (self.train_cutoff_timestamp, "train_cutoff_timestamp"),
            (self.test_first_timestamp, "test_first_timestamp"),
            (self.test_last_timestamp, "test_last_timestamp"),
        ):
            _aware(value, name)
        if not (
            self.train_first_timestamp
            <= self.train_cutoff_timestamp
            < self.test_first_timestamp
            <= self.test_last_timestamp
        ):
            raise ValueError("calendar fold timestamps are not ordered")
        if self.train_source_observations < 1 or self.test_source_observations < 1:
            raise ValueError("calendar fold source counts must be positive")
        if len(self.month_clock_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.month_clock_hash
        ):
            raise ValueError("month_clock_hash must be a lowercase SHA-256")

    @property
    def train_start(self) -> datetime:
        return self.train_first_timestamp

    @property
    def train_end(self) -> datetime:
        return self.train_cutoff_timestamp

    @property
    def test_start(self) -> datetime:
        return self.test_first_timestamp

    @property
    def test_end(self) -> datetime:
        return self.test_last_timestamp


@dataclass(frozen=True, slots=True)
class CalendarMonthPlan:
    """Deterministic monthly plan and its source cutoff."""

    timezone: str
    source_cutoff_timestamp: datetime
    folds: tuple[CalendarMonthFold, ...]
    plan_hash: str

    def __post_init__(self) -> None:
        if self.timezone != _DEFAULT_TIMEZONE:
            raise ValueError("calendar clock timezone is pinned to Europe/Berlin")
        _aware(self.source_cutoff_timestamp, "source_cutoff_timestamp")
        if len(self.plan_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.plan_hash
        ):
            raise ValueError("plan_hash must be a lowercase SHA-256")
        expected_ids = tuple(f"fold_{index:03d}" for index in range(1, len(self.folds) + 1))
        if tuple(fold.fold_id for fold in self.folds) != expected_ids:
            raise ValueError("calendar folds must preserve canonical order")
        if len({fold.test_calendar_month for fold in self.folds}) != len(self.folds):
            raise ValueError("calendar TEST months must be unique")


def _fold_hash(
    *,
    timezone: str,
    fold_index: int,
    train_month: str,
    train_first: datetime,
    train_cutoff: datetime,
    test_month: str,
    test_first: datetime,
    test_last: datetime,
    train_count: int,
    test_count: int,
) -> str:
    payload = {
        "fold_index": fold_index,
        "test_calendar_month": test_month,
        "test_first_timestamp": test_first.isoformat(),
        "test_last_timestamp": test_last.isoformat(),
        "test_source_observations": test_count,
        "timezone": timezone,
        "train_first_timestamp": train_first.isoformat(),
        "train_cutoff_timestamp": train_cutoff.isoformat(),
        "train_source_observations": train_count,
        "train_through_month": train_month,
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def plan_calendar_month(
    timestamps: Sequence[datetime],
    *,
    minimum_train_source_observations: int = 1260,
    evaluation_cutoff: datetime | None = None,
    timezone: str = _DEFAULT_TIMEZONE,
) -> CalendarMonthPlan:
    """Build expanding folds whose TEST is the next complete local month.

    A month is closed only when the effective source cutoff is on or after its
    local final calendar day.  The planner never synthesizes rows and never
    jumps across an absent immediate-following month.
    """

    if timezone != _DEFAULT_TIMEZONE:
        raise ValueError("calendar clock timezone is pinned to Europe/Berlin")
    if (
        not isinstance(minimum_train_source_observations, int)
        or isinstance(minimum_train_source_observations, bool)
        or minimum_train_source_observations < 1
    ):
        raise ValueError("minimum_train_source_observations must be a positive integer")
    if not timestamps:
        raise ValueError("calendar clock requires at least one source timestamp")
    ordered = tuple(_aware(value, "source timestamp") for value in timestamps)
    if any(current <= previous for previous, current in pairwise(ordered)):
        raise ValueError("source timestamps must be strictly increasing and unique")
    cutoff = _aware(evaluation_cutoff, "evaluation_cutoff") if evaluation_cutoff else ordered[-1]
    if cutoff < ordered[0] or cutoff > ordered[-1]:
        raise ValueError("evaluation_cutoff must lie within the source timestamp range")
    effective = tuple(value for value in ordered if value <= cutoff)
    if not effective:
        raise ValueError("evaluation_cutoff leaves no source timestamps")

    zone = ZoneInfo(timezone)
    by_month: dict[str, tuple[datetime, ...]] = {}
    for value in effective:
        key = _month(value, zone)
        by_month[key] = (*by_month.get(key, ()), value)
    months = tuple(sorted(by_month))
    closed_cutoff_local = cutoff.astimezone(zone)

    def is_closed(month: str) -> bool:
        first = _month_date(month)
        last_day = calendar.monthrange(first.year, first.month)[1]
        return closed_cutoff_local.date() >= date(first.year, first.month, last_day)

    folds: list[CalendarMonthFold] = []
    for train_month in months:
        test_month = _next_month(train_month)
        if test_month not in by_month or not is_closed(test_month):
            continue
        train_values = tuple(
            value for month in months if month <= train_month for value in by_month[month]
        )
        test_values = by_month[test_month]
        if len(train_values) < minimum_train_source_observations:
            continue
        index = len(folds) + 1
        train_cutoff = train_values[-1]
        fold_hash = _fold_hash(
            timezone=timezone,
            fold_index=index,
            train_month=train_month,
            train_first=train_values[0],
            train_cutoff=train_cutoff,
            test_month=test_month,
            test_first=test_values[0],
            test_last=test_values[-1],
            train_count=len(train_values),
            test_count=len(test_values),
        )
        folds.append(
            CalendarMonthFold(
                fold_index=index,
                fold_id=f"fold_{index:03d}",
                train_through_month=train_month,
                train_first_timestamp=train_values[0],
                train_cutoff_timestamp=train_cutoff,
                test_calendar_month=test_month,
                test_first_timestamp=test_values[0],
                test_last_timestamp=test_values[-1],
                train_source_observations=len(train_values),
                test_source_observations=len(test_values),
                month_clock_hash=fold_hash,
            )
        )

    plan_payload = {
        "evaluation_cutoff": cutoff.isoformat(),
        "folds": [
            {
                "fold_id": fold.fold_id,
                "month_clock_hash": fold.month_clock_hash,
                "test_calendar_month": fold.test_calendar_month,
                "test_first_timestamp": fold.test_first_timestamp.isoformat(),
                "test_last_timestamp": fold.test_last_timestamp.isoformat(),
                "train_cutoff_timestamp": fold.train_cutoff_timestamp.isoformat(),
                "train_first_timestamp": fold.train_first_timestamp.isoformat(),
                "train_source_observations": fold.train_source_observations,
                "test_source_observations": fold.test_source_observations,
                "train_through_month": fold.train_through_month,
            }
            for fold in folds
        ],
        "minimum_train_source_observations": minimum_train_source_observations,
        "timezone": timezone,
    }
    plan_hash = sha256(
        json.dumps(plan_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return CalendarMonthPlan(timezone, cutoff, tuple(folds), plan_hash)


__all__ = [
    "MIN_CALENDAR_MODEL_TEST_OBSERVATIONS",
    "CalendarMonthFold",
    "CalendarMonthPlan",
    "plan_calendar_month",
]
