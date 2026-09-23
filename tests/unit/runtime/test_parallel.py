from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from market_regime_engine.runtime.parallel import (
    FoldParallelExecutor,
    ParallelExecutionPlan,
    ReadOnlyMatrix,
)
from market_regime_engine.runtime.processes import cpu_process_pool


def square(value: int) -> int:
    return value * value


def child_pool_attempt(_: int) -> str:
    try:
        with cpu_process_pool(1):
            return "accepted"
    except RuntimeError as error:
        return str(error)


def worker_native_environment(_: int) -> tuple[str | None, str | None, str | None, str | None]:
    return tuple(
        os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    )  # type: ignore[return-value]


def test_auto_plan_uses_available_budget_and_task_count() -> None:
    plan = ParallelExecutionPlan.create(3)
    assert plan.available_cpu_budget >= 1
    assert plan.worker_count == min(plan.available_cpu_budget, 3)
    assert plan.runtime_metadata()["effective_worker_count"] == plan.worker_count

    limited = ParallelExecutionPlan.create(86, memory_worker_limit=4)
    assert limited.worker_count <= 4
    assert limited.shared_matrix_identity is None


def test_serial_executor_is_reference_mode_and_deterministic() -> None:
    plan = ParallelExecutionPlan.create(4, requested_workers=1)
    with FoldParallelExecutor[int, int](plan) as executor:
        assert executor.map_ordered(square, (3, 1, 2, 0)) == (9, 1, 4, 0)


def test_process_executor_is_bounded_ordered_and_caps_native_threads() -> None:
    plan = ParallelExecutionPlan.create(8, requested_workers=2)
    with FoldParallelExecutor[int, int](plan, max_pending=1) as executor:
        assert executor.map_ordered(square, range(8)) == tuple(value * value for value in range(8))
    environment_plan = ParallelExecutionPlan.create(1, requested_workers=2)
    with FoldParallelExecutor[int, tuple[str | None, ...]](environment_plan) as executor:
        environment = executor.map_ordered(worker_native_environment, (0,))[0]
    assert environment == ("1", "1", "1", "1")


def test_process_workers_cannot_create_child_pools() -> None:
    plan = ParallelExecutionPlan.create(2, requested_workers=2)
    with FoldParallelExecutor[int, str](plan) as executor:
        messages = executor.map_ordered(child_pool_attempt, (0, 1))
    assert messages == ("process-pool workers may not create child process pools",) * 2


def test_read_only_matrix_is_one_shared_identity_and_cleans_up(tmp_path: Path) -> None:
    source = np.ascontiguousarray(np.arange(12, dtype=np.float64).reshape(3, 4))
    with ReadOnlyMatrix.create(source, tmp_path) as shared:
        path = shared.path
        assert shared.array.flags.writeable is False
        assert np.array_equal(shared.array, source)
        with pytest.raises(ValueError, match="read-only"):
            shared.array[0, 0] = 99.0
        assert len(shared.identity) == 64
    assert not path.exists()
