"""Small process-pool helpers for CPU-bound v4 evaluation orchestration."""

from __future__ import annotations

import multiprocessing
import os
import threading
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from multiprocessing.context import BaseContext
from typing import cast

from market_regime_engine.runtime.cpu import cpu_worker_count

_NATIVE_THREAD_LIMITER: object | None = None


def _limit_native_numerical_threads() -> None:
    """Keep one BLAS/OpenMP numerical lane per process-pool worker."""

    global _NATIVE_THREAD_LIMITER
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
        # The environment cap is enough when numerical libraries initialise
        # after worker startup; minimal extension installs remain supported.
        _NATIVE_THREAD_LIMITER = None
    else:
        limiter = threadpool_limits(limits=1)
        limiter.__enter__()
        # Keep the context alive for the worker lifetime, including forked
        # workers whose numerical libraries were already imported.
        _NATIVE_THREAD_LIMITER = limiter


def _initialize_process_worker(
    initializer: Callable[..., None] | None,
    initargs: tuple[object, ...],
    cpu_affinity: tuple[int, ...] | None,
) -> None:
    _limit_native_numerical_threads()
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
    """Create a GIL-independent pool with a safe start method.

    ``fork`` is fastest when called by the main thread and preserves the
    immutable runtime configuration.  A pool created from a worker thread
    uses ``spawn`` so it cannot inherit a partially-held interpreter lock.
    """

    worker_limit = cpu_worker_count(max_workers)
    methods = multiprocessing.get_all_start_methods()
    context: BaseContext
    if "fork" in methods and threading.current_thread() is threading.main_thread():
        context = multiprocessing.get_context("fork")
    else:
        context = multiprocessing.get_context("spawn")
    with warnings.catch_warnings():
        if context.get_start_method() == "fork":
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"This process .* is multi-threaded, use .*fork\(\) may lead "
                    r"to deadlocks"
                ),
                category=DeprecationWarning,
                module=r"multiprocessing\.popen_fork",
            )
        pool_initializer = cast(Callable[[], object], _initialize_process_worker)
        pool_initargs = (initializer, initargs, cpu_affinity)
        with ProcessPoolExecutor(
            max_workers=worker_limit,
            mp_context=context,
            initializer=pool_initializer,
            initargs=cast(tuple[()], pool_initargs),
        ) as executor:
            yield executor
