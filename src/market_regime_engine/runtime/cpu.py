"""Affinity- and cgroup-aware CPU topology and worker sizing.

CPU-bound work must size process pools from the CPUs actually available to the
process.  ``os.cpu_count()`` describes the host, not necessarily the container
or the scheduler allocation, so it is deliberately not used here.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

_WORKER_ENV = "REGIME_CPU_WORKERS"


@dataclass(frozen=True, slots=True)
class CpuTopology:
    """CPUs visible to this process, including physical-core and NUMA data."""

    allowed_cpus: tuple[int, ...]
    physical_cores: tuple[tuple[int, int], ...]
    numa_nodes: tuple[tuple[int, tuple[int, ...]], ...]

    @property
    def logical_cpu_count(self) -> int:
        return len(self.allowed_cpus)

    @property
    def physical_core_count(self) -> int:
        return len(self.physical_cores)

    @property
    def numa_node_count(self) -> int:
        return len(self.numa_nodes)


def _allowed_cpus() -> tuple[int, ...]:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        return tuple(sorted(getter(0)))
    count = os.cpu_count() or 1
    return tuple(range(count))


def _parse_cpu_list(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for part in value.strip().split(","):
        if not part:
            continue
        bounds = part.split("-", 1)
        start = int(bounds[0])
        stop = int(bounds[-1])
        result.extend(range(start, stop + 1))
    return tuple(result)


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except OSError, ValueError:
        return None


def _physical_core_key(cpu: int) -> tuple[int, int]:
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    package = _read_int(topology / "physical_package_id")
    core = _read_int(topology / "core_id")
    return (package if package is not None else 0, core if core is not None else cpu)


def _cgroup_cpu_limit() -> int | None:
    """Return the integer CPU concurrency permitted by the active cgroup."""

    cgroup_mount = Path("/sys/fs/cgroup")
    roots = [cgroup_mount]
    try:
        membership = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        membership = []
    for line in membership:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _hierarchy, controllers, relative_path = fields
        if controllers == "":
            roots.append(cgroup_mount / relative_path.lstrip("/"))
        elif "cpu" in controllers.split(","):
            roots.append(cgroup_mount / "cpu" / relative_path.lstrip("/"))

    for root in dict.fromkeys(roots):
        v2_quota_path = root / "cpu.max"
        try:
            raw = v2_quota_path.read_text().strip().split()
        except OSError:
            raw = []
        if raw and raw[0] != "max":
            quota = int(raw[0])
            period = int(raw[1]) if len(raw) > 1 else 100_000
            if quota > 0 and period > 0:
                return max(1, math.floor(quota / period))

        quota_path = root / "cpu.cfs_quota_us"
        legacy_quota = _read_int(quota_path)
        legacy_period = _read_int(root / "cpu.cfs_period_us")
        if (
            legacy_quota is not None
            and legacy_period is not None
            and legacy_quota > 0
            and legacy_period > 0
        ):
            return max(1, math.floor(legacy_quota / legacy_period))
    return None


def cpu_topology() -> CpuTopology:
    """Discover the current affinity mask, physical cores, and NUMA nodes."""

    allowed = _allowed_cpus()
    physical = tuple(sorted({_physical_core_key(cpu) for cpu in allowed}))
    nodes: list[tuple[int, tuple[int, ...]]] = []
    for node_path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        node_name = node_path.name.removeprefix("node")
        try:
            node_id = int(node_name)
            node_cpus = tuple(
                sorted(set(_parse_cpu_list((node_path / "cpulist").read_text())) & set(allowed))
            )
        except OSError, ValueError:
            continue
        if node_cpus:
            nodes.append((node_id, node_cpus))
    return CpuTopology(allowed, physical, tuple(nodes))


def available_cpu_count() -> int:
    """Return the maximum useful logical worker count for this process."""

    topology_count = max(1, cpu_topology().logical_cpu_count)
    quota = _cgroup_cpu_limit()
    return max(1, min(topology_count, quota)) if quota is not None else topology_count


def cpu_worker_count(requested: int | None = None, *, task_count: int | None = None) -> int:
    """Resolve a process worker count without exceeding affinity or cgroup limits.

    ``REGIME_CPU_WORKERS`` is an optional deployment override.  It is useful
    for benchmarking and for sharing a host with other workloads; absent an
    override, all CPUs available to this process are used.
    """

    if requested is not None:
        value = requested
    else:
        configured = os.environ.get(_WORKER_ENV)
        value = int(configured) if configured else available_cpu_count()
    if value < 1:
        raise ValueError("max_workers must be at least 1")
    value = min(value, available_cpu_count())
    if task_count is not None:
        if task_count < 1:
            raise ValueError("task_count must be at least 1")
        value = min(value, task_count)
    return value
