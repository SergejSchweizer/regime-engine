from __future__ import annotations

from collections import OrderedDict
from time import sleep

import numpy as np
import pytest

from market_regime_engine.feature_discovery.feature_roles import (
    CORE_FEATURES,
    TEMPORAL_KEY,
    build_feature_role_contract,
)
from market_regime_engine.feature_discovery.global_reduction import (
    prune_global_correlated_features,
)
from market_regime_engine.runtime.parallel import FoldParallelExecutor, ParallelExecutionPlan


def _delayed_identity(task: tuple[int, float]) -> int:
    index, delay = task
    sleep(delay)
    return index


def _failing_tile(_task: int) -> int:
    raise RuntimeError("injected correlation tile failure")


def test_worker_counts_and_auto_preserve_global_mapping_and_hash() -> None:
    rows = np.arange(90, dtype=np.float64)
    names = (*CORE_FEATURES, *(f"family_pc_vix_{index}" for index in range(1, 9)))
    values = OrderedDict(
        (
            (name, tuple(np.sin(rows * (index + 1) / 17.0) + index / 100.0))
            for index, name in enumerate(names)
        )
    )
    values[names[1]] = tuple(rows)
    values[names[2]] = tuple(-rows)
    values[names[3]] = tuple(rows + 0.01)
    contract = build_feature_role_contract((TEMPORAL_KEY, *CORE_FEATURES))

    results = tuple(
        prune_global_correlated_features(values, contract, max_workers=worker_count)
        for worker_count in (1, 8, 32, 64, None)
    )

    assert all(result == results[0] for result in results)
    assert len(results[0].result_hash) == 64


def test_independent_reference_matches_every_reported_qualifying_edge() -> None:
    rows = np.arange(90, dtype=np.float64)
    base = tuple(rows)
    inverse = tuple(-rows)
    near = tuple(rows + 0.1)
    unstable = tuple(np.where(rows < 30, rows, np.sin(rows)))
    names = ("vix_log_level", "us_10y_log_level", "family_pc_vix_1", "family_pc_vix_2")
    values = OrderedDict(
        ((names[0], base), (names[1], inverse), (names[2], near), (names[3], unstable))
    )
    contract = build_feature_role_contract((TEMPORAL_KEY, names[0], names[1]))
    result = prune_global_correlated_features(values, contract, max_workers=8)

    def qualifies(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
        full = abs(float(np.corrcoef(left, right)[0, 1]))
        thirds = tuple(
            abs(float(np.corrcoef(left[start : start + 30], right[start : start + 30])[0, 1]))
            for start in (0, 30, 60)
        )
        return full >= 0.95 and min(thirds) >= 0.90

    qualifying = {
        frozenset((left, right))
        for left_index, left in enumerate(names)
        for right in names[left_index + 1 :]
        if qualifies(values[left], values[right])
    }
    assert qualifying == {
        frozenset((names[0], names[1])),
        frozenset((names[0], names[2])),
        frozenset((names[1], names[2])),
    }
    assert {frozenset((item.leader, item.removed)) for item in result.evidence} == {
        frozenset((names[0], names[1])),
        frozenset((names[0], names[2])),
    }
    assert result.representatives == (names[0], names[3])


def test_reordered_tile_completion_preserves_ordered_results() -> None:
    plan = ParallelExecutionPlan.create(4, requested_workers=4)
    tasks = ((0, 0.04), (1, 0.03), (2, 0.02), (3, 0.01))
    with FoldParallelExecutor[tuple[int, float], int](plan, max_pending=2) as executor:
        assert executor.map_ordered(_delayed_identity, tasks) == (0, 1, 2, 3)


def test_injected_tile_failure_aborts_without_returning_partial_mapping() -> None:
    plan = ParallelExecutionPlan.create(2, requested_workers=2)
    with (
        FoldParallelExecutor[int, int](plan, max_pending=2) as executor,
        pytest.raises(RuntimeError, match="injected correlation tile failure"),
    ):
        executor.map_ordered(_failing_tile, (0, 1))
