"""Small process-pool helpers for CPU-bound v4 evaluation orchestration."""

from __future__ import annotations

import multiprocessing
import threading
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from multiprocessing.context import BaseContext
from typing import cast

from market_regime_engine.runtime.cpu import cpu_worker_count


@contextmanager
def cpu_process_pool(
    max_workers: int | None = None,
    *,
    initializer: Callable[..., None] | None = None,
    initargs: tuple[object, ...] = (),
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
        with ProcessPoolExecutor(
            max_workers=worker_limit,
            mp_context=context,
            initializer=initializer,
            initargs=cast(tuple[()], initargs),
        ) as executor:
            yield executor
