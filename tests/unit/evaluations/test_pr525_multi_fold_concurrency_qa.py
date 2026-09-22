from __future__ import annotations

import time
from concurrent.futures import as_completed
from typing import Any

import pytest

import market_regime_engine.evaluations.global_regime_v4 as global_v4
from market_regime_engine.evaluations.process_parallel import cpu_process_pool
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.runtime.cpu import (
    available_cpu_count,
    cpu_worker_count,
    nested_worker_limits,
)
from tests.unit.evaluations.test_global_regime_v4 import (
    _catalog,
    _pickleable_outer_callback,
    _pickleable_teacher_callback,
    _rows,
)


def _delayed_fold_marker(task: tuple[int, float]) -> tuple[int, float, float]:
    fold_index, delay = task
    started = time.monotonic()
    time.sleep(delay)
    return fold_index, started, time.monotonic()


def test_eight_outer_fold_markers_overlap_and_are_reassembled_canonically() -> None:
    tasks = tuple((index, 0.02 + (7 - index) * 0.005) for index in range(8))
    with cpu_process_pool(max_workers=min(8, available_cpu_count())) as executor:
        futures = tuple(executor.submit(_delayed_fold_marker, task) for task in tasks)
        completed_order = [future.result() for future in as_completed(futures)]

    canonical = tuple(sorted(completed_order))
    assert tuple(item[0] for item in canonical) == tuple(range(8))
    assert {item[0] for item in completed_order} == set(range(8))
    assert (
        max(
            sum(
                other_start <= end and other_end >= start for _, other_start, other_end in canonical
            )
            for _, start, end in canonical
        )
        >= 2
    )


def test_worker_count_matrix_is_deterministic_and_bounded_by_eight_folds() -> None:
    expected = min(8, available_cpu_count())
    assert cpu_worker_count(1, task_count=8) == 1
    assert cpu_worker_count(8, task_count=8) == expected
    assert cpu_worker_count(32, task_count=8) == expected
    assert cpu_worker_count(64, task_count=8) == expected
    assert cpu_worker_count(None, task_count=8) == expected


def test_single_fold_keeps_the_direct_full_budget_path() -> None:
    total = min(86, available_cpu_count())
    assert nested_worker_limits(total, 1) == (total,)


def test_worker_matrix_has_identical_controller_results_and_typed_invalidity_is_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog()
    rows = _rows()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    plan = global_v4.plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    process_counts: list[int] = []

    def invalid_fold(*args: Any, **kwargs: Any):
        del kwargs
        fold = args[-1]
        configuration = global_v4._fallback_configuration(catalog, "build-1", fold, "invalid")
        return global_v4._invalid_outer_fold(fold, configuration, "typed statistical invalidity")

    class InlinePool:
        def __init__(self, max_workers: int, **_: object) -> None:
            process_counts.append(max_workers)

        def __enter__(self) -> InlinePool:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(self, function: object, fold: object):
            class Future:
                def result(self) -> object:
                    return function(fold)  # type: ignore[operator]

            return Future()

    monkeypatch.setattr(global_v4, "cpu_process_pool", InlinePool)
    monkeypatch.setattr(global_v4, "_evaluate_outer_fold", invalid_fold)
    monkeypatch.setattr(global_v4, "_evaluate_outer_fold_process", invalid_fold)

    results = []
    for requested in (1, 8, 32, 64, None):
        results.append(
            global_v4.evaluate_global_regime_v4(
                rows,
                catalog=catalog,
                profile=profile,
                outer_runner=_pickleable_outer_callback,
                teacher_refitter=_pickleable_teacher_callback,
                max_workers=requested,
            )
        )

    assert tuple(result.result_hash for result in results) == (results[0].result_hash,) * 5
    assert all(result.valid_fold_count == 0 for result in results)
    assert len(plan.folds) >= 5
    assert process_counts


def test_unexpected_outer_fold_failure_aborts_before_result_is_assembled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog()
    rows = _rows()
    profile = load_profile("configs/profiles/xetra_v4.yaml")

    def crashing_fold(fold):
        if fold.fold_index == 2:
            raise RuntimeError("unexpected software failure")
        configuration = global_v4._fallback_configuration(catalog, "build-1", fold, "failure")
        return global_v4._invalid_outer_fold(fold, configuration, "synthetic")

    class InlinePool:
        def __init__(self, max_workers: int, **_: object) -> None:
            del max_workers

        def __enter__(self) -> InlinePool:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(self, function: object, fold: object):
            class Future:
                def result(self) -> object:
                    return function(fold)  # type: ignore[operator]

            return Future()

    monkeypatch.setattr(global_v4, "cpu_process_pool", InlinePool)
    monkeypatch.setattr(global_v4, "_evaluate_outer_fold_process", crashing_fold)
    with pytest.raises(RuntimeError, match="unexpected software failure"):
        global_v4.evaluate_global_regime_v4(
            rows,
            catalog=catalog,
            profile=profile,
            outer_runner=_pickleable_outer_callback,
            teacher_refitter=_pickleable_teacher_callback,
            max_workers=8,
        )
