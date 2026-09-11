#!/usr/bin/env python3
"""Benchmark GIL-independent process scaling for CPU-bound orchestration.

This intentionally uses a pure-Python CPU kernel: if this benchmark scales,
the measured parallelism comes from separate interpreters rather than from
threads accidentally sharing the GIL.  It reports wall time, throughput,
speedup, child CPU utilization, and peak child RSS as JSON.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from concurrent.futures import Future

from market_regime_engine.evaluations.process_parallel import cpu_process_pool
from market_regime_engine.runtime.cpu import available_cpu_count, cpu_topology


def _cpu_kernel(iterations: int) -> int:
    value = 0x12345678
    for index in range(iterations):
        value = ((value ^ (index + 0x9E3779B9)) * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
    return value


def _run(workers: int, *, tasks: int, iterations: int) -> dict[str, float | int]:
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    with cpu_process_pool(workers) as executor:
        futures: list[Future[int]] = [
            executor.submit(_cpu_kernel, iterations) for _ in range(tasks)
        ]
        results = [future.result() for future in futures]
    wall_seconds = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu_seconds = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    return {
        "workers": workers,
        "tasks": tasks,
        "iterations_per_task": iterations,
        "wall_seconds": wall_seconds,
        "throughput_tasks_per_second": tasks / wall_seconds,
        "child_cpu_seconds": child_cpu_seconds,
        "cpu_utilization_of_available_percent": 100.0
        * child_cpu_seconds
        / (wall_seconds * available_cpu_count()),
        "worker_saturation_percent": 100.0 * child_cpu_seconds / (wall_seconds * workers),
        "peak_child_rss_mib": after.ru_maxrss / 1024.0,
        "result_checksum": sum(results) & 0xFFFFFFFF,
    }


def main() -> None:
    topology = cpu_topology()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", default="1,2,4,8,16,32,physical,logical")
    parser.add_argument("--tasks", type=int, default=max(32, topology.logical_cpu_count))
    parser.add_argument("--iterations", type=int, default=500_000)
    args = parser.parse_args()
    if args.tasks < 1 or args.iterations < 1:
        parser.error("--tasks and --iterations must be positive")

    requested: list[int] = []
    for value in args.workers.split(","):
        if value == "physical":
            requested.append(topology.physical_core_count)
        elif value == "logical":
            requested.append(topology.logical_cpu_count)
        else:
            requested.append(int(value))
    unique_workers = tuple(dict.fromkeys(requested))
    baseline: dict[str, float | int] | None = None
    records: list[dict[str, float | int]] = []
    for workers in unique_workers:
        record = _run(workers, tasks=args.tasks, iterations=args.iterations)
        if baseline is None:
            baseline = record
        record["speedup_vs_first"] = float(baseline["wall_seconds"]) / float(record["wall_seconds"])
        records.append(record)
    print(
        json.dumps(
            {
                "available_logical_cpus": topology.logical_cpu_count,
                "available_physical_cores": topology.physical_core_count,
                "numa_nodes": topology.numa_node_count,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
