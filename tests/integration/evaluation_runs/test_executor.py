from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from threading import Lock
from time import monotonic, sleep
from typing import NoReturn

import pytest

from market_regime_engine.evaluation_runs.contracts import EvaluationRunIdentity
from market_regime_engine.evaluation_runs.executor import (
    DomainInvalid,
    ResumableEvaluationExecutor,
)
from market_regime_engine.evaluation_runs.graph import (
    EvaluationWorkGraph,
    WorkUnitNode,
    make_work_unit,
)
from market_regime_engine.evaluation_runs.store import SQLiteEvaluationRunStore

pytestmark = pytest.mark.integration


def identity() -> EvaluationRunIdentity:
    return EvaluationRunIdentity(
        "global_regime_v4",
        "xetra",
        4,
        "a" * 64,
        1,
        "b" * 64,
        "c" * 64,
        datetime(2026, 1, 1, tzinfo=UTC),
        "d" * 40,
        "e" * 64,
        "3.14.7",
    )


def _parallel_compute(node: WorkUnitNode, parents: tuple[bytes, ...]) -> bytes:
    del parents
    sleep(0.2)
    return node.identity.key.encode("utf-8")


def _process_executor_worker(root: str) -> tuple[str, int, int]:
    run = identity()
    root_node = make_work_unit(run, unit_type="outer", coordinates=(("fold", "001"),))
    child_node = make_work_unit(
        run,
        unit_type="seed",
        coordinates=(("fold", "001"), ("seed", "11")),
    )
    graph = EvaluationWorkGraph.from_nodes(
        run,
        (WorkUnitNode(root_node), WorkUnitNode(child_node, (root_node.key,))),
    )

    def compute(node: WorkUnitNode, parents: tuple[bytes, ...]) -> bytes:
        if node.identity.key == root_node.key:
            sleep(0.15)
            return b"root"
        assert parents == (b"root",)
        return b"child"

    result = ResumableEvaluationExecutor(SQLiteEvaluationRunStore(root)).execute(
        run, graph, compute
    )
    return result.root_evidence_hash, result.computed_unit_count, result.reused_unit_count


def test_executor_resumes_and_completed_run_is_zero_compute(tmp_path: Path) -> None:
    run = identity()
    root = make_work_unit(run, unit_type="outer", coordinates=(("fold", "001"),))
    store = SQLiteEvaluationRunStore(tmp_path)
    executor = ResumableEvaluationExecutor(store)
    calls: list[str] = []

    def compute(node: WorkUnitNode, parents: tuple[bytes, ...]) -> bytes:
        calls.append(node.identity.key)
        if node.identity.key == root.key:
            return b"root"
        assert parents == (b"root",)
        return b"child"

    child = make_work_unit(
        run,
        unit_type="seed",
        coordinates=(("fold", "001"), ("seed", "11")),
        parent_payload_hashes=(sha256(b"root").hexdigest(),),
    )
    graph = EvaluationWorkGraph.from_nodes(
        run,
        (WorkUnitNode(root), WorkUnitNode(child, (root.key,))),
    )
    first = executor.execute(run, graph, compute)
    assert calls == [root.key, child.key]
    assert first.computed_unit_count == 2
    calls.clear()
    second = executor.execute(run, graph, lambda *_: (_ for _ in ()).throw(AssertionError()))
    assert second.root_evidence_hash == first.root_evidence_hash
    assert second.computed_unit_count == 0
    assert second.reused_unit_count == 2
    assert calls == []


def test_independent_pickleable_units_use_process_workers(tmp_path: Path) -> None:
    run = identity()
    nodes = tuple(
        WorkUnitNode(
            make_work_unit(
                run,
                unit_type="independent",
                coordinates=(("index", f"{index:03d}"),),
            )
        )
        for index in range(4)
    )
    graph = EvaluationWorkGraph.from_nodes(run, nodes)
    store = SQLiteEvaluationRunStore(tmp_path)
    started = monotonic()
    result = ResumableEvaluationExecutor(store, max_workers=4).execute(
        run, graph, _parallel_compute
    )
    elapsed = monotonic() - started

    assert result.computed_unit_count == 4
    assert result.reused_unit_count == 0
    assert elapsed < 0.65
    assert all(
        store.load_completed_work_unit(run, node.identity) == node.identity.key.encode("utf-8")
        for node in nodes
    )


def test_executor_caches_domain_invalid_without_retry(tmp_path: Path) -> None:
    run = identity()
    node = make_work_unit(run, unit_type="candidate", coordinates=(("id", "bad"),))
    graph = EvaluationWorkGraph.from_nodes(run, (WorkUnitNode(node),))
    executor = ResumableEvaluationExecutor(SQLiteEvaluationRunStore(tmp_path))
    calls = 0

    def invalid(*_args: object) -> NoReturn:
        nonlocal calls
        calls += 1
        raise DomainInvalid("insufficient observations")

    executor.execute(run, graph, invalid)
    executor.execute(run, graph, lambda *_: (_ for _ in ()).throw(AssertionError()))
    assert calls == 1


def test_concurrent_duplicate_executors_share_one_terminal_payload(tmp_path: Path) -> None:
    run = identity()
    root = make_work_unit(run, unit_type="outer", coordinates=(("fold", "001"),))
    child = make_work_unit(
        run,
        unit_type="seed",
        coordinates=(("fold", "001"), ("seed", "11")),
    )
    graph = EvaluationWorkGraph.from_nodes(
        run,
        (WorkUnitNode(root), WorkUnitNode(child, (root.key,))),
    )
    store = SQLiteEvaluationRunStore(tmp_path)
    executor_a = ResumableEvaluationExecutor(store)
    executor_b = ResumableEvaluationExecutor(store)
    calls: list[str] = []
    calls_lock = Lock()

    def compute(node: WorkUnitNode, parents: tuple[bytes, ...]) -> bytes:
        with calls_lock:
            calls.append(node.identity.key)
        if node.identity.key == root.key:
            sleep(0.15)
            return b"root"
        assert parents == (b"root",)
        return b"child"

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = tuple(
            pool.map(
                lambda executor: executor.execute(run, graph, compute),
                (executor_a, executor_b),
            )
        )

    assert first.root_evidence_hash == second.root_evidence_hash
    assert sorted(calls) == sorted((root.key, child.key))
    assert first.computed_unit_count + second.computed_unit_count == 2
    assert first.reused_unit_count + second.reused_unit_count == 2


def test_two_spawned_executors_share_one_terminal_payload(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        first, second = tuple(pool.map(_process_executor_worker, (str(tmp_path),) * 2))

    assert first[0] == second[0]
    assert first[1] + second[1] == 2
    assert first[2] + second[2] == 2


def test_large_dag_interruption_restarts_only_unfinished_units(tmp_path: Path) -> None:
    run = identity()
    nodes = tuple(
        WorkUnitNode(
            make_work_unit(
                run,
                unit_type="inner_fold",
                coordinates=(("index", f"{index:03d}"),),
            )
        )
        for index in range(120)
    )
    graph = EvaluationWorkGraph.from_nodes(run, nodes)
    store = SQLiteEvaluationRunStore(tmp_path)
    executor = ResumableEvaluationExecutor(store)
    calls: list[str] = []

    def interrupted_compute(node: WorkUnitNode, parents: tuple[bytes, ...]) -> bytes:
        del parents
        calls.append(node.identity.key)
        if len(calls) == 47:
            raise RuntimeError("forced process interruption")
        return node.identity.key.encode("utf-8")

    with pytest.raises(RuntimeError, match="forced process interruption"):
        executor.execute(run, graph, interrupted_compute)

    first_attempt = tuple(calls)
    result = executor.execute(
        run,
        graph,
        lambda node, parents: node.identity.key.encode("utf-8"),
    )
    assert result.computed_unit_count == 74
    assert result.reused_unit_count == 46
    assert len(first_attempt) == 47
    assert len(set(first_attempt)) == 47
    assert store.state(run).status == "COMPLETE"


def test_multiple_interruption_points_match_uninterrupted_root(tmp_path: Path) -> None:
    run = identity()
    nodes = tuple(
        WorkUnitNode(
            make_work_unit(
                run,
                unit_type="inner_fold",
                coordinates=(("index", f"{index:03d}"),),
            )
        )
        for index in range(120)
    )
    graph = EvaluationWorkGraph.from_nodes(run, nodes)

    baseline_store = SQLiteEvaluationRunStore(tmp_path / "baseline")
    baseline = ResumableEvaluationExecutor(baseline_store).execute(
        run,
        graph,
        lambda node, parents: node.identity.key.encode("utf-8"),
    )
    for interruption in (1, 17, 59, 119):
        store = SQLiteEvaluationRunStore(tmp_path / f"interruption-{interruption}")
        calls = 0

        def interrupted(
            node: WorkUnitNode,
            parents: tuple[bytes, ...],
            interruption_point: int = interruption,
        ) -> bytes:
            nonlocal calls
            del parents
            calls += 1
            if calls == interruption_point:
                raise RuntimeError("forced interruption")
            return node.identity.key.encode("utf-8")

        with pytest.raises(RuntimeError, match="forced interruption"):
            ResumableEvaluationExecutor(store).execute(run, graph, interrupted)
        resumed = ResumableEvaluationExecutor(store).execute(
            run,
            graph,
            lambda node, parents: node.identity.key.encode("utf-8"),
        )
        assert resumed.root_evidence_hash == baseline.root_evidence_hash
        assert resumed.reused_unit_count == interruption - 1
        assert resumed.computed_unit_count == 120 - (interruption - 1)
