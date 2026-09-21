from __future__ import annotations

import time

import pytest

from market_regime_engine.evaluations.task_frontier import (
    FrontierTask,
    SharedTaskFrontier,
)


def _task(task_id: str, seed: int) -> FrontierTask[tuple[str, ...]]:
    return FrontierTask(
        task_id,
        2 + seed % 4,
        "fold-001",
        ("feature_a", "feature_b"),
        seed,
        "a" * 64,
        "matrix-001",
        (0, 1, 2),
        (3, 4),
        (task_id,),
    )


def _worker(task: FrontierTask[tuple[str, ...]]) -> tuple[str, ...]:
    time.sleep((4 - task.seed) * 0.005)
    return task.payload


def _failing_worker(_task: FrontierTask[tuple[str, ...]]) -> tuple[str, ...]:
    raise RuntimeError("fit backend failure")


def test_frontier_reuses_one_pool_and_assembles_canonical_order() -> None:
    tasks = tuple(_task(f"task-{seed}", seed) for seed in (3, 1, 2))
    with SharedTaskFrontier[tuple[str, ...], tuple[str, ...]](max_workers=2) as frontier:
        first = frontier.map(tasks, _worker)
        second = frontier.map(tuple(reversed(tasks)), _worker)

    assert tuple(task.task_id for task, _ in first.values) == (
        "task-1",
        "task-2",
        "task-3",
    )
    assert first.values == second.values
    assert first.metrics.submitted_count == 3
    assert first.metrics.completed_count == 3
    assert first.metrics.max_queue_depth == 3
    assert first.metrics.runnable_tasks == 3


def test_frontier_rejects_duplicate_ids_and_worker_failure() -> None:
    tasks = (_task("task-1", 1), _task("task-1", 2))
    with SharedTaskFrontier[tuple[str, ...], tuple[str, ...]](max_workers=1) as frontier:
        with pytest.raises(ValueError, match="unique"):
            frontier.map(tasks, _worker)

        with pytest.raises(RuntimeError, match="fit backend failure"):
            frontier.map((_task("task-2", 1),), _failing_worker)
