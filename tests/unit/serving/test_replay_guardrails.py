from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from market_regime_engine.serving.replay_admission import ReplayAdmission
from market_regime_engine.serving.replay_limits import ReplayGuardrailError, ReplayLimits

START = datetime(2020, 1, 1, tzinfo=UTC)


def test_pinned_defaults_and_inclusive_interval_validation() -> None:
    limits = ReplayLimits()
    assert limits.max_rows == 10_000
    assert limits.max_internal_rows == 15_000
    assert limits.max_range_days == 14_610
    assert limits.timeout_seconds == 60
    assert limits.max_response_bytes == 26_214_400
    assert limits.max_concurrency_per_worker == 1
    limits.validate_interval(START, START)
    limits.validate_interval(START, START + timedelta(days=14_610))

    with pytest.raises(ReplayGuardrailError) as inverted:
        limits.validate_interval(START + timedelta(seconds=1), START)
    assert (inverted.value.status_code, inverted.value.error_code) == (
        400,
        "invalid_replay_interval",
    )
    with pytest.raises(ReplayGuardrailError) as too_long:
        limits.validate_interval(START, START + timedelta(days=14_610, seconds=1))
    assert too_long.value.status_code == 413
    with pytest.raises(ReplayGuardrailError) as naive:
        limits.validate_interval(START.replace(tzinfo=None), START)
    assert naive.value.status_code == 400


def test_preflight_and_final_size_limits_are_exact_413() -> None:
    limits = ReplayLimits()
    limits.validate_estimates(
        response_rows=10_000,
        internal_rows=15_000,
        estimated_response_bytes=26_214_400,
    )
    cases = (
        ({"response_rows": 10_001, "internal_rows": 1, "estimated_response_bytes": 1}, "replay_row_limit"),
        ({"response_rows": 1, "internal_rows": 15_001, "estimated_response_bytes": 1}, "replay_internal_row_limit"),
        ({"response_rows": 1, "internal_rows": 1, "estimated_response_bytes": 26_214_401}, "replay_response_too_large"),
    )
    for values, code in cases:
        with pytest.raises(ReplayGuardrailError) as exc_info:
            limits.validate_estimates(**values)
        assert exc_info.value.status_code == 413
        assert exc_info.value.error_code == code
    with pytest.raises(ReplayGuardrailError) as final_size:
        limits.validate_serialized_size(26_214_401)
    assert final_size.value.status_code == 413


def test_process_capacity_is_nonblocking_and_released_after_work_stops() -> None:
    admission = ReplayAdmission(ReplayLimits())
    with admission.admit():
        with pytest.raises(ReplayGuardrailError) as capacity:
            with admission.admit():
                raise AssertionError("unreachable")
        assert capacity.value.status_code == 503
        assert capacity.value.error_code == "replay_capacity_exhausted"
    with admission.admit():
        pass


def test_cooperative_monotonic_deadline_returns_504_after_unwind() -> None:
    now = [100.0]
    admission = ReplayAdmission(ReplayLimits(timeout_seconds=60), clock=lambda: now[0])
    with pytest.raises(ReplayGuardrailError) as timed_out:
        with admission.admit() as permit:
            permit.check_deadline()
            now[0] = 160.0
            permit.check_deadline()
            now[0] = 160.0001
            permit.check_deadline()
    assert timed_out.value.status_code == 504
    assert timed_out.value.error_code == "replay_timeout"
    with admission.admit():
        pass


def test_environment_override_validation() -> None:
    limits = ReplayLimits.from_env(
        {
            "REGIME_REPLAY_MAX_ROWS": "9",
            "REGIME_REPLAY_TIMEOUT_SECONDS": "2.5",
        }
    )
    assert limits.max_rows == 9
    assert limits.timeout_seconds == 2.5
    with pytest.raises(ValueError, match="integer"):
        ReplayLimits.from_env({"REGIME_REPLAY_MAX_ROWS": "bad"})
    with pytest.raises(ValueError, match="numeric"):
        ReplayLimits.from_env({"REGIME_REPLAY_TIMEOUT_SECONDS": "bad"})
