from __future__ import annotations

import pickle
from datetime import UTC, datetime
from hashlib import sha256
from types import SimpleNamespace
from typing import NoReturn

import pytest

import market_regime_engine.evaluation_runs.executor as executor_module
import market_regime_engine.evaluation_runs.hmm_units as hmm_units
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
from market_regime_engine.evaluation_runs.hmm_units import HMMSeedCheckpoint, SeedFitOutcome
from market_regime_engine.evaluation_runs.store import SQLiteEvaluationRunStore, WorkUnitStatus
from market_regime_engine.training.multistart import StartDiagnostic


def _identity() -> EvaluationRunIdentity:
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


def _process_safe_payload(node: WorkUnitNode, parents: tuple[bytes, ...]) -> bytes:
    return f"{node.identity.unit_type}:{len(parents)}".encode()


def _process_domain_invalid(node: WorkUnitNode, parents: tuple[bytes, ...]) -> bytes:
    del parents
    if node.identity.unit_type == "invalid":
        raise DomainInvalid("domain rejected")
    return b"valid"


def _process_technical_failure(node: WorkUnitNode, parents: tuple[bytes, ...]) -> bytes:
    del node, parents
    raise RuntimeError("worker failed")


def test_executor_reuses_completed_graph_and_validates_configuration(tmp_path) -> None:
    run = _identity()
    root = make_work_unit(run, unit_type="outer", coordinates=(("fold", "001"),))
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
    store = SQLiteEvaluationRunStore(tmp_path)
    executor = ResumableEvaluationExecutor(store, max_workers=1)
    calls: list[str] = []

    def compute(node: WorkUnitNode, parents: tuple[bytes, ...]) -> bytes:
        calls.append(node.identity.key)
        return b"root" if node.identity.key == root.key else b"child"

    first = executor.execute(run, graph, compute)
    second = executor.execute(run, graph, lambda *_: (_ for _ in ()).throw(AssertionError()))
    assert first.computed_unit_count == 2
    assert second.reused_unit_count == 2
    assert first.root_evidence_hash == second.root_evidence_hash
    assert calls == [root.key, child.key]

    with pytest.raises(ValueError, match="max_workers"):
        ResumableEvaluationExecutor(store, max_workers=0)
    with pytest.raises(ValueError, match="does not belong"):
        executor.execute(
            EvaluationRunIdentity(
                "other",
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
            ),
            graph,
            compute,
        )


def test_executor_rejects_unpickleable_callbacks_and_cyclic_graphs(tmp_path) -> None:
    run = _identity()
    store = SQLiteEvaluationRunStore(tmp_path)
    executor = ResumableEvaluationExecutor(store)
    node = make_work_unit(run, unit_type="single")
    graph = EvaluationWorkGraph.from_nodes(run, (WorkUnitNode(node),))
    assert not executor._process_safe(lambda *_: b"payload")
    with pytest.raises(ValueError, match="wait_timeout_seconds"):
        executor._load_or_claim(run, WorkUnitNode(node), wait_timeout_seconds=0.0)

    first = make_work_unit(run, unit_type="first")
    second = make_work_unit(run, unit_type="second")
    with pytest.raises(ValueError, match="dependency cycle"):
        EvaluationWorkGraph.from_nodes(
            run,
            (WorkUnitNode(first, (second.key,)), WorkUnitNode(second, (first.key,))),
        )
    assert graph.run_key == run.key


def test_executor_claim_wait_handles_concurrent_commit_and_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    run = _identity()
    node = WorkUnitNode(make_work_unit(run, unit_type="waiting"))

    class FakeStore:
        def __init__(self, states: list[object | None], claims: list[bool]) -> None:
            self.states = iter(states)
            self.claims = iter(claims)

        def load_completed_work_unit(self, *_args: object) -> None:
            return None

        def claim_work_unit(self, *_args: object) -> bool:
            return next(self.claims)

        def work_unit_state(self, *_args: object) -> object | None:
            return next(self.states)

    monkeypatch.setattr(executor_module, "sleep", lambda _seconds: None)
    monkeypatch.setattr(executor_module, "monotonic", iter((0.0, 0.0)).__next__)
    cached, claimed = ResumableEvaluationExecutor(
        FakeStore([None], [True])  # type: ignore[arg-type]
    )._load_or_claim(run, node)
    assert cached is None and claimed

    monkeypatch.setattr(executor_module, "monotonic", iter((0.0, 2.0)).__next__)
    with pytest.raises(RuntimeError, match="remains claimed"):
        ResumableEvaluationExecutor(
            FakeStore([SimpleNamespace(status=WorkUnitStatus.PENDING)] * 2, [False, False])  # type: ignore[arg-type]
        )._load_or_claim(run, node, wait_timeout_seconds=1.0)


def test_executor_completed_run_requires_durable_payload(tmp_path) -> None:
    run = _identity()
    node = make_work_unit(run, unit_type="completed")
    graph = SimpleNamespace(run_key=run.key, nodes=(WorkUnitNode(node),))

    class CompletedStore(SQLiteEvaluationRunStore):
        def open_run(self, _identity: EvaluationRunIdentity) -> object:
            return SimpleNamespace(status="COMPLETE", root_identity_hash="root")

        def load_completed_run(self, _identity: EvaluationRunIdentity) -> None:
            return None

    with pytest.raises(ValueError, match="durable payload"):
        ResumableEvaluationExecutor(CompletedStore(tmp_path)).execute(
            run, graph, _process_safe_payload
        )


def test_executor_caches_domain_invalid_and_releases_technical_failure(tmp_path) -> None:
    run = _identity()
    node = make_work_unit(run, unit_type="candidate", coordinates=(("id", "bad"),))
    graph = EvaluationWorkGraph.from_nodes(run, (WorkUnitNode(node),))
    store = SQLiteEvaluationRunStore(tmp_path)
    executor = ResumableEvaluationExecutor(store)
    calls = 0

    def invalid(*_args: object) -> NoReturn:
        nonlocal calls
        calls += 1
        raise DomainInvalid("insufficient observations")

    executor.execute(run, graph, invalid)
    executor.execute(run, graph, lambda *_: (_ for _ in ()).throw(AssertionError()))
    assert calls == 1

    technical_node = make_work_unit(run, unit_type="technical")
    technical_graph = EvaluationWorkGraph.from_nodes(run, (WorkUnitNode(technical_node),))
    technical_store = SQLiteEvaluationRunStore(tmp_path / "technical")
    technical_executor = ResumableEvaluationExecutor(technical_store)
    with pytest.raises(RuntimeError, match="temporary"):
        technical_executor.execute(
            run,
            technical_graph,
            lambda *_: (_ for _ in ()).throw(RuntimeError("temporary")),
        )
    state = technical_store.work_unit_state(run, technical_node)
    assert state is not None and state.status is WorkUnitStatus.PENDING
    result = technical_executor.execute(run, technical_graph, lambda *_: b"recovered")
    assert result.computed_unit_count == 1


def test_executor_rejects_invalid_callback_payload_and_parent_hash(tmp_path) -> None:
    run = _identity()
    root = make_work_unit(run, unit_type="root")
    graph = EvaluationWorkGraph.from_nodes(run, (WorkUnitNode(root),))
    store = SQLiteEvaluationRunStore(tmp_path)
    executor = ResumableEvaluationExecutor(store)
    with pytest.raises(ValueError, match="non-empty bytes"):
        executor.execute(run, graph, lambda *_: b"")
    assert store.work_unit_state(run, root).status is WorkUnitStatus.PENDING

    child = make_work_unit(
        run,
        unit_type="child",
        parent_payload_hashes=("f" * 64,),
    )
    invalid_graph = EvaluationWorkGraph.from_nodes(
        run,
        (WorkUnitNode(root), WorkUnitNode(child, (root.key,))),
    )
    with pytest.raises(ValueError, match="parent payload hash mismatch"):
        executor.execute(run, invalid_graph, lambda *_: b"root")


def test_hmm_seed_checkpoint_roundtrip_is_idempotent_and_rejects_conflicts(tmp_path) -> None:
    run = _identity()
    store = SQLiteEvaluationRunStore(tmp_path)
    checkpoint = HMMSeedCheckpoint(
        store=store,
        run_identity=run,
        candidate_id="candidate",
        fold_id="fold",
        state_count=2,
    )
    outcome = SeedFitOutcome(
        diagnostic=StartDiagnostic(
            seed=11,
            success=False,
            converged=False,
            iterations=None,
            train_log_likelihood=None,
            artifact=None,
            failure_reason="synthetic failure",
        ),
        result=None,
    )

    store.open_run(run)
    assert checkpoint.load(11) is None
    checkpoint.save(11, outcome)
    assert checkpoint.load(11) == outcome
    checkpoint.save(11, outcome)
    with pytest.raises(ValueError, match="different bytes"):
        checkpoint.save(
            11,
            SeedFitOutcome(
                diagnostic=StartDiagnostic(
                    seed=11,
                    success=False,
                    converged=False,
                    iterations=None,
                    train_log_likelihood=None,
                    artifact=None,
                    failure_reason="different failure",
                ),
                result=None,
            ),
        )

    incompatible_store = SQLiteEvaluationRunStore(tmp_path / "incompatible")
    incompatible_store.open_run(run)
    incompatible_checkpoint = HMMSeedCheckpoint(
        store=incompatible_store,
        run_identity=run,
        candidate_id="candidate",
        fold_id="fold",
        state_count=2,
    )
    incompatible_unit = incompatible_checkpoint.unit(11)
    assert incompatible_store.claim_work_unit(run, incompatible_unit)
    incompatible_store.complete_work_unit(run, incompatible_unit, pickle.dumps("wrong"))
    with pytest.raises(ValueError, match="incompatible"):
        incompatible_checkpoint.load(11)


def test_hmm_seed_checkpoint_times_out_behind_live_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClaimedStore:
        def load_completed_work_unit(self, identity, unit):
            del identity, unit
            return None

        def claim_work_unit(self, identity, unit):
            del identity, unit
            return False

        def work_unit_state(self, identity, unit):
            del identity, unit
            return SimpleNamespace(status=hmm_units.WorkUnitStatus.RUNNING)

    values = iter((0.0, 31.0))
    monkeypatch.setattr(hmm_units, "monotonic", lambda: next(values))
    checkpoint = HMMSeedCheckpoint(
        store=ClaimedStore(),
        run_identity=_identity(),
        candidate_id="candidate",
        fold_id="fold",
        state_count=2,
    )
    outcome = SeedFitOutcome(
        diagnostic=StartDiagnostic(
            seed=11,
            success=False,
            converged=False,
            iterations=None,
            train_log_likelihood=None,
            artifact=None,
            failure_reason="timeout fixture",
        ),
        result=None,
    )
    with pytest.raises(RuntimeError, match="remains claimed"):
        checkpoint.save(11, outcome)


def test_executor_uses_process_workers_for_independent_units(tmp_path) -> None:
    run = _identity()
    nodes = tuple(
        WorkUnitNode(make_work_unit(run, unit_type="independent", coordinates=(("index", str(i)),)))
        for i in range(3)
    )
    result = ResumableEvaluationExecutor(SQLiteEvaluationRunStore(tmp_path), max_workers=2).execute(
        run,
        EvaluationWorkGraph.from_nodes(run, nodes),
        _process_safe_payload,
    )
    assert result.computed_unit_count == 3
    assert result.reused_unit_count == 0


def test_executor_process_workers_persist_domain_invalid_siblings(tmp_path) -> None:
    run = _identity()
    store = SQLiteEvaluationRunStore(tmp_path)
    nodes = tuple(
        WorkUnitNode(make_work_unit(run, unit_type=unit_type)) for unit_type in ("invalid", "valid")
    )
    result = ResumableEvaluationExecutor(store, max_workers=2).execute(
        run,
        EvaluationWorkGraph.from_nodes(run, nodes),
        _process_domain_invalid,
    )
    assert result.computed_unit_count == 2
    invalid_state = store.work_unit_state(run, nodes[0].identity)
    assert invalid_state is not None and invalid_state.status is WorkUnitStatus.DOMAIN_INVALID
    assert b"DOMAIN_INVALID" in (store.load_completed_work_unit(run, nodes[0].identity) or b"")


def test_executor_process_worker_failure_releases_claims(tmp_path) -> None:
    run = _identity()
    node = WorkUnitNode(make_work_unit(run, unit_type="technical"))
    store = SQLiteEvaluationRunStore(tmp_path)
    with pytest.raises(RuntimeError, match="worker failed"):
        ResumableEvaluationExecutor(store, max_workers=2).execute(
            run,
            EvaluationWorkGraph.from_nodes(run, (node,)),
            _process_technical_failure,
        )
    state = store.work_unit_state(run, node.identity)
    assert state is not None and state.status is WorkUnitStatus.PENDING
