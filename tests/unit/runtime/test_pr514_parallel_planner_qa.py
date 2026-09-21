from __future__ import annotations

import time

import pytest

import market_regime_engine.runtime.parallel as parallel_module
from market_regime_engine.runtime.parallel import FoldParallelExecutor, ParallelExecutionPlan


def delayed_square(value: int) -> int:
    time.sleep((8 - value) * 0.001)
    return value * value


def fail_on_three(value: int) -> int:
    if value == 3:
        raise RuntimeError("worker failure")
    return value


def interrupt_worker(_: int) -> int:
    raise KeyboardInterrupt


def test_worker_count_matrix_has_no_hidden_cap_and_respects_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parallel_module, "available_cpu_count", lambda: 86)

    def fake_worker_count(requested: int | None = None, *, task_count: int | None = None) -> int:
        assert task_count is not None
        return min(requested or 86, 86, task_count)

    monkeypatch.setattr(parallel_module, "cpu_worker_count", fake_worker_count)
    assert [
        ParallelExecutionPlan.create(100, requested_workers=requested).worker_count
        for requested in (1, 8, 32, 64)
    ] == [1, 8, 32, 64]
    assert ParallelExecutionPlan.create(100).worker_count == 86


def test_process_completion_order_cannot_change_canonical_results() -> None:
    plan = ParallelExecutionPlan.create(8, requested_workers=2)
    with FoldParallelExecutor[int, int](plan, max_pending=2) as executor:
        result = executor.map_ordered(delayed_square, range(8))
    assert result == tuple(value * value for value in range(8))


def test_worker_failure_propagates_without_partial_result() -> None:
    plan = ParallelExecutionPlan.create(8, requested_workers=2)
    with (
        FoldParallelExecutor[int, int](plan, max_pending=2) as executor,
        pytest.raises(RuntimeError, match="worker failure"),
    ):
        executor.map_ordered(fail_on_three, range(8))


def test_keyboard_interrupt_cancels_pending_tasks_and_propagates() -> None:
    plan = ParallelExecutionPlan.create(8, requested_workers=2)
    with (
        FoldParallelExecutor[int, int](plan, max_pending=2) as executor,
        pytest.raises(KeyboardInterrupt),
    ):
        executor.map_ordered(interrupt_worker, range(8))


def test_serial_mode_is_identical_to_process_task_order() -> None:
    plan = ParallelExecutionPlan.create(8, requested_workers=1)
    with FoldParallelExecutor[int, int](plan) as executor:
        assert executor.map_ordered(delayed_square, range(8)) == tuple(
            value * value for value in range(8)
        )
