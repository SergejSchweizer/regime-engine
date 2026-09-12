"""Low-overhead, opt-in performance telemetry for complete evaluations."""

from __future__ import annotations

import json
import os
import resource
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from market_regime_engine.runtime.cpu import available_cpu_count, cpu_topology


def _rss_mib() -> float | None:
    try:
        value = next(
            line.split()[1]
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith("VmRSS:")
        )
    except OSError, StopIteration, ValueError:
        return None
    return float(value) / 1024.0


@dataclass(frozen=True, slots=True)
class PerformanceStage:
    name: str
    wall_seconds: float
    process_cpu_seconds: float
    child_cpu_seconds: float
    parent_rss_mib: float | None
    child_peak_rss_mib: float | None
    task_count: int | None
    worker_count: int | None

    def as_dict(self) -> dict[str, Any]:
        # rusage counters can differ by a few nanoseconds across process
        # reaping boundaries.  A negative child delta is measurement noise,
        # never useful CPU time.
        child_cpu_seconds = max(0.0, self.child_cpu_seconds)
        cpu_seconds = self.process_cpu_seconds + child_cpu_seconds
        capacity = available_cpu_count()
        return {
            "name": self.name,
            "wall_seconds": self.wall_seconds,
            "process_cpu_seconds": self.process_cpu_seconds,
            "child_cpu_seconds": child_cpu_seconds,
            "total_cpu_seconds": cpu_seconds,
            "effective_cpu_utilization_percent": (
                100.0 * cpu_seconds / (self.wall_seconds * capacity)
                if self.wall_seconds > 0
                else 0.0
            ),
            "parent_rss_mib": self.parent_rss_mib,
            "child_peak_rss_mib": self.child_peak_rss_mib,
            "available_logical_cpus": capacity,
            "task_count": self.task_count,
            "worker_count": self.worker_count,
        }


class PerformanceRecorder:
    """Collect stage telemetry and atomically write one JSON report on finish."""

    def __init__(self, path: str | Path | None) -> None:
        self._path = None if path is None else Path(path).expanduser()
        self._started = time.perf_counter()
        self._stages: list[PerformanceStage] = []
        self._metadata: dict[str, Any] = {}
        self._status = "RUNNING"
        self._error: dict[str, str] | None = None

    @classmethod
    def from_environment(cls) -> PerformanceRecorder:
        return cls(os.environ.get("REGIME_PERFORMANCE_REPORT_PATH"))

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        task_count: int | None = None,
        worker_count: int | None = None,
    ) -> Iterator[None]:
        started = time.perf_counter()
        process_before = resource.getrusage(resource.RUSAGE_SELF)
        children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        try:
            yield
        finally:
            process_after = resource.getrusage(resource.RUSAGE_SELF)
            children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
            self._stages.append(
                PerformanceStage(
                    name=name,
                    wall_seconds=time.perf_counter() - started,
                    process_cpu_seconds=(
                        process_after.ru_utime
                        + process_after.ru_stime
                        - process_before.ru_utime
                        - process_before.ru_stime
                    ),
                    child_cpu_seconds=(
                        children_after.ru_utime
                        + children_after.ru_stime
                        - children_before.ru_utime
                        - children_before.ru_stime
                    ),
                    parent_rss_mib=_rss_mib(),
                    child_peak_rss_mib=children_after.ru_maxrss / 1024.0,
                    task_count=task_count,
                    worker_count=worker_count,
                )
            )

    def update_metadata(self, values: Mapping[str, Any]) -> None:
        self._metadata.update(values)

    def finish(
        self,
        *,
        status: str,
        metadata: Mapping[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        if metadata:
            self.update_metadata(metadata)
        self._status = status
        if error is not None:
            self._error = {"type": type(error).__name__, "message": str(error)}
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        topology = cpu_topology()
        report: dict[str, Any] = {
            "schema_version": 2,
            "status": self._status,
            "available_logical_cpus": available_cpu_count(),
            "affinity_logical_cpus": topology.logical_cpu_count,
            "physical_core_count": topology.physical_core_count,
            "numa_nodes": [
                {"node": node_id, "logical_cpu_count": len(cpus)}
                for node_id, cpus in topology.numa_nodes
            ],
            "native_thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
                if os.environ.get(name) is not None
            },
            "wall_seconds": time.perf_counter() - self._started,
            "peak_parent_rss_mib": max(
                (stage.parent_rss_mib or 0.0 for stage in self._stages), default=0.0
            ),
            "peak_child_rss_mib": max(
                (stage.child_peak_rss_mib or 0.0 for stage in self._stages), default=0.0
            ),
            "metadata": self._metadata,
            "stages": [stage.as_dict() for stage in self._stages],
        }
        if self._error is not None:
            report["error"] = self._error
        temporary = self._path.with_name(f".{self._path.name}.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self._path)


__all__ = ["PerformanceRecorder", "PerformanceStage"]
