"""Shared, GIL-independent execution primitives for fold-local CPU stages."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import AbstractContextManager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import mkstemp
from types import TracebackType
from typing import TypeVar, cast

import numpy as np

from market_regime_engine.runtime.cpu import available_cpu_count, cpu_worker_count

T = TypeVar("T")
R = TypeVar("R")
_WORKER_MARKER = "REGIME_CPU_PROCESS_WORKER"


@dataclass(frozen=True, slots=True)
class ParallelExecutionPlan:
    """Immutable effective resource decision for one runnable fold stage."""

    available_cpu_budget: int
    runnable_task_count: int
    requested_workers: int | None
    memory_worker_limit: int | None
    worker_count: int
    native_thread_count: int = 1
    shared_matrix_identity: str | None = None

    @classmethod
    def create(
        cls,
        task_count: int,
        *,
        requested_workers: int | None = None,
        memory_worker_limit: int | None = None,
        shared_matrix_identity: str | None = None,
    ) -> ParallelExecutionPlan:
        if task_count < 1:
            raise ValueError("task_count must be at least 1")
        if memory_worker_limit is not None and memory_worker_limit < 1:
            raise ValueError("memory_worker_limit must be at least 1")
        available = available_cpu_count()
        worker_count = cpu_worker_count(requested_workers, task_count=task_count)
        if memory_worker_limit is not None:
            worker_count = min(worker_count, memory_worker_limit)
        return cls(
            available_cpu_budget=available,
            runnable_task_count=task_count,
            requested_workers=requested_workers,
            memory_worker_limit=memory_worker_limit,
            worker_count=worker_count,
            shared_matrix_identity=shared_matrix_identity,
        )

    @property
    def serial(self) -> bool:
        return self.worker_count == 1

    @property
    def mode(self) -> str:
        return "serial" if self.serial else "process"

    def runtime_metadata(self) -> dict[str, int | str | None]:
        return {
            "available_cpu_budget": self.available_cpu_budget,
            "runnable_task_count": self.runnable_task_count,
            "requested_workers": self.requested_workers,
            "memory_worker_limit": self.memory_worker_limit,
            "effective_worker_count": self.worker_count,
            "native_thread_count_per_worker": self.native_thread_count,
            "parallel_mode": self.mode,
            "shared_matrix_identity": self.shared_matrix_identity,
        }


class ReadOnlyMatrix:
    """One immutable, file-backed numeric matrix shared by process workers."""

    def __init__(
        self, path: Path, shape: tuple[int, ...], dtype: np.dtype[np.generic], identity: str
    ):
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self.identity = identity
        self._matrix: np.memmap | None = None

    @classmethod
    def create(cls, matrix: np.ndarray, directory: str | Path) -> ReadOnlyMatrix:
        if matrix.ndim != 2 or not matrix.flags.c_contiguous:
            raise ValueError("shared matrix must be a contiguous two-dimensional array")
        root = Path(directory)
        root.mkdir(mode=0o750, parents=True, exist_ok=True)
        descriptor, raw_path = mkstemp(prefix="regime-fold-", suffix=".matrix", dir=root)
        os.close(descriptor)
        path = Path(raw_path)
        identity = sha256(memoryview(matrix).cast("B")).hexdigest()
        mapped = np.memmap(path, dtype=matrix.dtype, mode="w+", shape=matrix.shape)
        mapped[:] = matrix
        mapped.flush()
        del mapped
        return cls(path, tuple(matrix.shape), matrix.dtype, identity)

    def __enter__(self) -> ReadOnlyMatrix:
        mapped = np.memmap(self.path, dtype=self.dtype, mode="r", shape=self.shape)
        mapped.flags.writeable = False
        self._matrix = mapped
        return self

    def __exit__(self, *_args: object) -> None:
        self._matrix = None
        self.path.unlink(missing_ok=True)

    @property
    def array(self) -> np.memmap:
        if self._matrix is None:
            raise RuntimeError("shared matrix is not open")
        return self._matrix


class FoldParallelExecutor[T, R]:
    """Reuse one bounded process pool and restore canonical result ordering."""

    def __init__(self, plan: ParallelExecutionPlan, *, max_pending: int | None = None) -> None:
        if max_pending is not None and max_pending < 1:
            raise ValueError("max_pending must be at least 1")
        self.plan = plan
        self._max_pending = max_pending or max(1, plan.worker_count * 2)
        self._pool_context: AbstractContextManager[ProcessPoolExecutor] | None = None
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> FoldParallelExecutor[T, R]:
        if not self.plan.serial:
            from market_regime_engine.runtime.processes import cpu_process_pool

            self._pool_context = cpu_process_pool(self.plan.worker_count)
            self._pool = self._pool_context.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._pool_context is not None:
            self._pool_context.__exit__(exc_type, exc, traceback)
            self._pool_context = None
            self._pool = None

    def map_ordered(self, function: Callable[[T], R], tasks: Iterable[T]) -> tuple[R, ...]:
        ordered_tasks = tuple(tasks)
        if len(ordered_tasks) != self.plan.runnable_task_count:
            raise ValueError("task count differs from the execution plan")
        if self.plan.serial:
            return tuple(function(task) for task in ordered_tasks)
        if self._pool is None:
            raise RuntimeError("FoldParallelExecutor must be entered before map_ordered")

        results: list[R | None] = [None] * len(ordered_tasks)
        pending: dict[Future[R], int] = {}
        pool = self._pool
        assert pool is not None
        iterator = iter(enumerate(ordered_tasks))

        def fill() -> None:
            while len(pending) < self._max_pending:
                try:
                    index, task = next(iterator)
                except StopIteration:
                    return
                pending[pool.submit(function, task)] = index

        fill()
        try:
            while pending:
                completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in completed:
                    index = pending.pop(future)
                    results[index] = future.result()
                fill()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
        return tuple(cast(R, result) for result in results)


__all__ = ["FoldParallelExecutor", "ParallelExecutionPlan", "ReadOnlyMatrix"]
