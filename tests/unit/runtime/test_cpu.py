from __future__ import annotations

import os

import pytest

from market_regime_engine.runtime.cpu import (
    cpu_topology,
    cpu_worker_count,
    nested_worker_limits,
)


def test_topology_is_affinity_aware() -> None:
    topology = cpu_topology()
    assert topology.allowed_cpus
    assert topology.logical_cpu_count == len(topology.allowed_cpus)
    assert topology.physical_core_count >= 1
    assert all(set(cpus) <= set(topology.allowed_cpus) for _node, cpus in topology.numa_nodes)


def test_worker_count_respects_task_count_and_explicit_limit() -> None:
    assert cpu_worker_count(2, task_count=1) == 1
    assert cpu_worker_count(2, task_count=2) == 2
    with pytest.raises(ValueError, match="at least 1"):
        cpu_worker_count(0)


def test_worker_count_uses_deployment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGIME_CPU_WORKERS", "3")
    assert cpu_worker_count(task_count=99) == min(3, len(os.sched_getaffinity(0)))


@pytest.mark.parametrize(
    ("total_budget", "outer_workers", "expected"),
    (
        (86, 12, (7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)),
        (86, 24, (3,) * 14 + (2,) * 10),
        (24, 24, (1,) * 24),
        (86, 1, (86,)),
    ),
)
def test_nested_worker_limits_partition_outer_and_child_budget(
    total_budget: int,
    outer_workers: int,
    expected: tuple[int, ...],
) -> None:
    limits = nested_worker_limits(total_budget, outer_workers)
    assert limits == expected
    if outer_workers == 1:
        assert limits[0] == total_budget
    else:
        assert outer_workers + sum(value for value in limits if value > 1) == total_budget


def test_nested_worker_limits_rejects_impossible_outer_count() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        nested_worker_limits(2, 3)
