from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from market_regime_engine.evaluation.calendar_clock import plan_calendar_month
from market_regime_engine.evaluation.model_clock import build_calendar_model_clock_preflight


def _daily(start: date, end: date) -> tuple[datetime, ...]:
    count = (end - start).days + 1
    return tuple(
        datetime.combine(start + timedelta(days=index), datetime.min.time(), tzinfo=UTC)
        for index in range(count)
    )


def test_month_clock_uses_real_month_lengths_and_exact_source_cutoffs() -> None:
    timestamps = _daily(date(2020, 1, 1), date(2020, 5, 31))
    plan = plan_calendar_month(timestamps, minimum_train_source_observations=3)

    assert tuple(fold.test_calendar_month for fold in plan.folds) == (
        "2020-02",
        "2020-03",
        "2020-04",
        "2020-05",
    )
    assert tuple(fold.test_source_observations for fold in plan.folds) == (29, 31, 30, 31)
    assert plan.folds[0].train_through_month == "2020-01"
    assert plan.folds[0].train_cutoff_timestamp == timestamps[30]
    assert plan.folds[-1].test_last_timestamp == timestamps[-1]
    assert all(len(fold.month_clock_hash) == 64 for fold in plan.folds)
    assert len(plan.plan_hash) == 64


def test_month_clock_excludes_partial_final_month() -> None:
    timestamps = _daily(date(2020, 1, 1), date(2020, 5, 15))
    plan = plan_calendar_month(timestamps, minimum_train_source_observations=3)

    assert tuple(fold.test_calendar_month for fold in plan.folds) == (
        "2020-02",
        "2020-03",
        "2020-04",
    )


def test_month_clock_does_not_jump_over_absent_immediate_month() -> None:
    timestamps = (
        *_daily(date(2020, 1, 1), date(2020, 1, 3)),
        *_daily(date(2020, 3, 1), date(2020, 3, 31)),
    )
    plan = plan_calendar_month(timestamps, minimum_train_source_observations=1)

    assert tuple(fold.test_calendar_month for fold in plan.folds) == ()


def test_month_membership_uses_berlin_time_and_preserves_original_timezone() -> None:
    timestamps = (
        datetime(2024, 2, 29, 23, 30, tzinfo=UTC),
        datetime(2024, 3, 31, 22, 30, tzinfo=UTC),
        datetime(2024, 4, 30, 21, 30, tzinfo=UTC),
    )
    plan = plan_calendar_month(timestamps, minimum_train_source_observations=1)

    assert tuple(fold.test_calendar_month for fold in plan.folds) == ("2024-04",)
    assert plan.folds[0].test_first_timestamp == timestamps[1]
    assert plan.folds[0].test_first_timestamp.tzinfo is UTC
    assert plan.source_cutoff_timestamp.tzinfo is UTC


def test_future_month_rows_do_not_change_earlier_fold_membership() -> None:
    base = _daily(date(2020, 1, 1), date(2020, 4, 30))
    extended = (*base, *_daily(date(2020, 5, 1), date(2020, 5, 31)))
    first = plan_calendar_month(base, minimum_train_source_observations=3).folds[0]
    extended_first = plan_calendar_month(extended, minimum_train_source_observations=3).folds[0]

    assert first == extended_first


def test_monthly_model_clock_preflight_preserves_month_evidence_and_gates() -> None:
    timestamps = _daily(date(2020, 1, 1), date(2020, 5, 31))
    rows = pd.DataFrame(
        {
            "timestamp_m1": timestamps,
            "f0": tuple(float(index) for index in range(len(timestamps))),
            "f1": tuple(float(index + 1) for index in range(len(timestamps))),
        }
    )
    plan = plan_calendar_month(timestamps, minimum_train_source_observations=3)
    preflight = build_calendar_model_clock_preflight(
        rows,
        ("f0", "f1"),
        plan,
        minimum_model_train_observations=3,
        minimum_model_test_observations=1,
        minimum_valid_fold_rate=1.0,
    )

    assert preflight.status.value == "valid"
    assert preflight.structural_valid_fold_rate == 1.0
    assert preflight.folds[0].train_through_month == "2020-01"
    assert preflight.folds[0].test_calendar_month == "2020-02"
    assert preflight.folds[0].train_cutoff_timestamp == timestamps[30]
    assert preflight.folds[0].month_clock_hash == plan.folds[0].month_clock_hash


@pytest.mark.parametrize(
    "bad",
    (
        (),
        (datetime(2024, 1, 1),),
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)),
    ),
)
def test_month_clock_fails_closed_for_empty_naive_or_duplicate_source(
    bad: tuple[datetime, ...],
) -> None:
    with pytest.raises(ValueError):
        plan_calendar_month(bad)
