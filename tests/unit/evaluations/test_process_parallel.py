from __future__ import annotations

import os

from market_regime_engine.evaluations.process_parallel import cpu_process_pool


def _native_thread_probe(_value: int) -> tuple[dict[str, str | None], tuple[int, ...]]:
    import numpy
    from threadpoolctl import threadpool_info

    del numpy
    counts = tuple(
        int(item["num_threads"])
        for item in threadpool_info()
        if item.get("num_threads") is not None
    )
    return (
        {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        counts,
    )


def test_process_pool_caps_native_numerical_threads() -> None:
    with cpu_process_pool(2) as executor:
        values = [executor.submit(_native_thread_probe, index).result() for index in range(2)]

    for environment, thread_counts in values:
        assert set(environment.values()) == {"1"}
        assert all(count <= 1 for count in thread_counts)
