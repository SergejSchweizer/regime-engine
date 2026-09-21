"""One persistent process frontier for independent CPU-bound HMM fit tasks."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import Future, as_completed
from dataclasses import dataclass
from time import monotonic
from typing import Any

from market_regime_engine.evaluations.process_parallel import cpu_process_pool, is_pickleable
from market_regime_engine.runtime.cpu import cpu_worker_count


@dataclass(frozen=True, slots=True)
class FrontierTask[T]:
    """Immutable identity for one independent HMM fit.

    Data transfer is intentionally represented by row/column indices and a
    shared matrix identity.  A worker must resolve the slice from the shared
    read-only source rather than receive a copied fold matrix.
    """

    task_id: str
    state_count: int
    fold_id: str
    candidate_subset: tuple[str, ...]
    seed: int
    profile_hash: str
    matrix_identity: str
    row_indices: tuple[int, ...]
    column_indices: tuple[int, ...]
    payload: T

    def __post_init__(self) -> None:
        if not self.task_id or self.task_id.strip() != self.task_id:
            raise ValueError("frontier task_id must be non-empty and trimmed")
        if self.state_count not in (2, 3, 4, 5):
            raise ValueError("frontier state_count must be 2, 3, 4, or 5")
        if not self.fold_id or self.fold_id.strip() != self.fold_id:
            raise ValueError("frontier fold_id must be non-empty and trimmed")
        if not self.candidate_subset or len(set(self.candidate_subset)) != len(
            self.candidate_subset
        ):
            raise ValueError("frontier candidate_subset must be non-empty and unique")
        if self.seed < 0:
            raise ValueError("frontier seed must be non-negative")
        if len(self.profile_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.profile_hash
        ):
            raise ValueError("frontier profile_hash must be a lowercase SHA-256")
        if not self.matrix_identity or self.matrix_identity.strip() != self.matrix_identity:
            raise ValueError("frontier matrix_identity must be non-empty and trimmed")
        if not self.row_indices or any(index < 0 for index in self.row_indices):
            raise ValueError("frontier row_indices must be non-empty and non-negative")
        if not self.column_indices or any(index < 0 for index in self.column_indices):
            raise ValueError("frontier column_indices must be non-empty and non-negative")

    @property
    def canonical_key(self) -> tuple[object, ...]:
        return (
            self.state_count,
            self.fold_id,
            self.candidate_subset,
            self.seed,
            self.task_id,
        )


@dataclass(frozen=True, slots=True)
class FrontierMetrics:
    submitted_count: int
    completed_count: int
    max_queue_depth: int
    runnable_tasks: int
    elapsed_seconds: float
    worker_utilization_proxy: float


@dataclass(frozen=True, slots=True)
class FrontierResult[T, R]:
    values: tuple[tuple[FrontierTask[T], R], ...]
    metrics: FrontierMetrics


def _run_frontier_task[T, R](worker: Callable[[FrontierTask[T]], R], task: FrontierTask[T]) -> R:
    return worker(task)


class SharedTaskFrontier[T, R]:
    """Persistent bounded process pool with canonical result assembly."""

    def __init__(self, max_workers: int | None = None) -> None:
        self._requested_workers = max_workers
        self._executor: Any = None
        self._context: Any = None
        self._worker_limit = 0

    @property
    def worker_limit(self) -> int:
        if self._worker_limit < 1:
            raise RuntimeError("frontier is not active")
        return self._worker_limit

    def __enter__(self) -> SharedTaskFrontier[T, R]:
        if os.environ.get("REGIME_CPU_PROCESS_WORKER") == "1":
            raise RuntimeError("frontier workers may not create child process pools")
        self._worker_limit = cpu_worker_count(self._requested_workers)
        self._context = cpu_process_pool(self._worker_limit)
        self._executor = self._context.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._context is not None:
            self._context.__exit__(exc_type, exc, traceback)
        self._executor = None
        self._worker_limit = 0

    def map(
        self,
        tasks: Iterable[FrontierTask[T]],
        worker: Callable[[FrontierTask[T]], R],
    ) -> FrontierResult[T, R]:
        if self._executor is None:
            raise RuntimeError("frontier must be entered before map")
        if not is_pickleable(worker):
            raise TypeError("frontier worker must be pickleable")
        ordered = tuple(sorted(tasks, key=lambda task: task.canonical_key))
        if not ordered:
            raise ValueError("frontier requires at least one task")
        if len({task.task_id for task in ordered}) != len(ordered):
            raise ValueError("frontier task IDs must be unique")
        started = monotonic()
        futures: dict[Future[R], FrontierTask[T]] = {
            self._executor.submit(_run_frontier_task, worker, task): task for task in ordered
        }
        completed: dict[str, R] = {}
        for future in as_completed(futures):
            task = futures[future]
            # Unexpected worker exceptions intentionally propagate immediately.
            completed[task.task_id] = future.result()
        elapsed = monotonic() - started
        metrics = FrontierMetrics(
            submitted_count=len(ordered),
            completed_count=len(completed),
            max_queue_depth=len(ordered),
            runnable_tasks=len(ordered),
            elapsed_seconds=elapsed,
            worker_utilization_proxy=(
                min(1.0, len(ordered) / self.worker_limit) if elapsed >= 0.0 else 0.0
            ),
        )
        return FrontierResult(
            tuple((task, completed[task.task_id]) for task in ordered),
            metrics,
        )


__all__ = [
    "FrontierMetrics",
    "FrontierResult",
    "FrontierTask",
    "SharedTaskFrontier",
]
