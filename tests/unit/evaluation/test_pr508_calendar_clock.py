from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from market_regime_engine.evaluation.calendar_clock import plan_calendar_month


def _daily(start: date, end: date) -> tuple[datetime, ...]:
    return tuple(
        datetime.combine(start + timedelta(days=index), datetime.min.time(), tzinfo=UTC)
        for index in range((end - start).days + 1)
    )


def _serialized_plan(timestamps: tuple[datetime, ...]) -> bytes:
    return pickle.dumps(
        plan_calendar_month(timestamps, minimum_train_source_observations=1),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def test_golden_month_lengths_include_leap_day_and_december_to_january_rollover() -> None:
    timestamps = _daily(date(2020, 11, 1), date(2021, 3, 31))
    plan = plan_calendar_month(timestamps, minimum_train_source_observations=1)

    assert tuple(fold.test_calendar_month for fold in plan.folds) == (
        "2020-12",
        "2021-01",
        "2021-02",
        "2021-03",
    )
    assert tuple(fold.test_source_observations for fold in plan.folds) == (31, 31, 28, 31)
    assert plan.folds[0].train_through_month == "2020-11"
    assert plan.folds[1].train_through_month == "2020-12"
    assert plan.folds[2].train_through_month == "2021-01"


def test_local_month_membership_is_unique_across_cet_and_cest_transitions() -> None:
    timestamps = (
        datetime(2024, 3, 31, 0, 30, tzinfo=UTC),
        datetime(2024, 3, 31, 1, 30, tzinfo=UTC),
        datetime(2024, 9, 30, 21, 30, tzinfo=UTC),
        datetime(2024, 10, 27, 0, 30, tzinfo=UTC),
        datetime(2024, 10, 27, 1, 30, tzinfo=UTC),
        datetime(2024, 10, 31, 23, 0, tzinfo=UTC),
    )
    local_months = tuple(
        value.astimezone(ZoneInfo("Europe/Berlin")).strftime("%Y-%m") for value in timestamps
    )

    assert local_months == ("2024-03", "2024-03", "2024-09", "2024-10", "2024-10", "2024-11")
    assert len(local_months) == len(set((index, month) for index, month in enumerate(local_months)))
    plan = plan_calendar_month(timestamps, minimum_train_source_observations=1)
    assert tuple(fold.test_calendar_month for fold in plan.folds) == ("2024-10",)
    assert plan.folds[0].test_first_timestamp == timestamps[3]


def test_january_refit_then_february_and_march_oos_months_are_causal() -> None:
    timestamps = _daily(date(2024, 1, 1), date(2024, 3, 31))
    plan = plan_calendar_month(timestamps, minimum_train_source_observations=1)

    january = plan.folds[0]
    february = plan.folds[1]
    assert january.train_through_month == "2024-01"
    assert january.test_calendar_month == "2024-02"
    assert january.train_cutoff_timestamp < january.test_first_timestamp
    assert february.train_through_month == "2024-02"
    assert february.test_calendar_month == "2024-03"
    assert february.train_cutoff_timestamp == january.test_last_timestamp
    assert january.test_last_timestamp < february.test_first_timestamp


def test_mid_september_cutoff_never_emits_partial_september() -> None:
    timestamps = _daily(date(2024, 1, 1), date(2024, 9, 15))
    plan = plan_calendar_month(
        timestamps,
        minimum_train_source_observations=1,
        evaluation_cutoff=timestamps[-1],
    )

    assert plan.folds[-1].test_calendar_month == "2024-08"
    assert "2024-09" not in tuple(fold.test_calendar_month for fold in plan.folds)


def test_calendar_clock_rejects_fixed_block_and_trading_day_approximations() -> None:
    timestamps = _daily(date(2024, 1, 1), date(2024, 4, 30))
    plan = plan_calendar_month(timestamps, minimum_train_source_observations=1)

    assert tuple(fold.test_source_observations for fold in plan.folds) == (29, 31, 30)
    assert all(fold.test_source_observations != 63 for fold in plan.folds)
    assert all(fold.test_source_observations != 21 for fold in plan.folds)
    assert tuple(fold.test_calendar_month for fold in plan.folds) != (
        "2024-02",
        "2024-03",
        "2024-04",
        "2024-05",
    )


def test_duplicate_source_rows_fail_instead_of_creating_overlapping_test_folds() -> None:
    timestamps = _daily(date(2024, 1, 1), date(2024, 3, 31))
    duplicate = (*timestamps[:60], timestamps[59], *timestamps[60:])

    with pytest.raises(ValueError, match="strictly increasing and unique"):
        plan_calendar_month(duplicate, minimum_train_source_observations=1)


def test_serial_plan_construction_is_byte_identical_and_future_rows_do_not_change_first_fold() -> (
    None
):
    timestamps = _daily(date(2024, 1, 1), date(2024, 5, 31))
    plan = plan_calendar_month(timestamps, minimum_train_source_observations=1)
    repeated = plan_calendar_month(timestamps, minimum_train_source_observations=1)
    extended = plan_calendar_month(
        (*timestamps, *_daily(date(2024, 6, 1), date(2024, 6, 30))),
        minimum_train_source_observations=1,
    )

    assert pickle.dumps(plan, protocol=pickle.HIGHEST_PROTOCOL) == pickle.dumps(
        repeated, protocol=pickle.HIGHEST_PROTOCOL
    )
    with ProcessPoolExecutor(max_workers=1) as executor:
        assert executor.submit(_serialized_plan, timestamps).result() == pickle.dumps(
            plan, protocol=pickle.HIGHEST_PROTOCOL
        )
    assert plan.folds[0] == extended.folds[0]
