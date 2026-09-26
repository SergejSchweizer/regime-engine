"""GIL-independent process-pool primitives for CPU-bound runtime stages."""

from __future__ import annotations

import multiprocessing
import os
import pickle
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from multiprocessing.context import BaseContext
from typing import cast

from market_regime_engine.runtime.cpu import cpu_worker_count

_NATIVE_THREAD_LIMITER: object | None = None


def is_pickleable(value: object) -> bool:
    try:
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except AttributeError, OSError, pickle.PickleError, TypeError:
        return False
    return True


def _initialize_process_worker(
    initializer: Callable[..., None] | None,
    initargs: tuple[object, ...],
    cpu_affinity: tuple[int, ...] | None,
) -> None:
    global _NATIVE_THREAD_LIMITER
    os.environ["REGIME_CPU_PROCESS_WORKER"] = "1"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
    except ImportError:
        _NATIVE_THREAD_LIMITER = None
    else:
        _NATIVE_THREAD_LIMITER = threadpool_limits(limits=1)
        _NATIVE_THREAD_LIMITER.__enter__()  # type: ignore[union-attr]
    if cpu_affinity is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpu_affinity)
    if initializer is not None:
        initializer(*initargs)


@contextmanager
def cpu_process_pool(
    max_workers: int | None = None,
    *,
    initializer: Callable[..., None] | None = None,
    initargs: tuple[object, ...] = (),
    cpu_affinity: tuple[int, ...] | None = None,
) -> Iterator[ProcessPoolExecutor]:
    """Create a bounded process pool, independent of the Python GIL."""
    if os.environ.get("REGIME_CPU_PROCESS_WORKER") == "1":
        raise RuntimeError("process-pool workers may not create child process pools")
    worker_limit = cpu_worker_count(max_workers)
    # Never fork a live application process.  The evaluation parent has
    # already initialized MLflow, HTTP and test-runner threads; forking after
    # that point can copy a locked mutex into every child and leave the parent
    # waiting forever for futures whose workers are idle in futex.  Spawn
    # starts a clean interpreter and keeps the pool GIL-independent without
    # inheriting those locks.
    context: BaseContext = multiprocessing.get_context("spawn")
    with warnings.catch_warnings():
        if context.get_start_method() == "fork":
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                module=r"multiprocessing\.popen_fork",
            )
        pool_initializer = cast(Callable[[], object], _initialize_process_worker)
        executor = ProcessPoolExecutor(
            max_workers=worker_limit,
            mp_context=context,
            initializer=pool_initializer,
            initargs=cast(tuple[()], (initializer, initargs, cpu_affinity)),
        )
        try:
            yield executor
        except BaseException:
            terminate = getattr(executor, "terminate_workers", None)
            if terminate is not None:
                terminate()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)


__all__ = ["cpu_process_pool", "is_pickleable"]
