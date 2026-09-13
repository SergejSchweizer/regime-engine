from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from market_regime_engine.evaluation_runs.contracts import (
    EvaluationRunIdentity,
    WorkUnitIdentity,
)
from market_regime_engine.evaluation_runs.executor import ResumableEvaluationExecutor
from market_regime_engine.evaluation_runs.graph import (
    EvaluationWorkGraph,
    WorkUnitNode,
    make_work_unit,
)
from market_regime_engine.evaluation_runs.store import SQLiteEvaluationRunStore, WorkUnitStatus

pytestmark = pytest.mark.integration

_DAG_LAYERS = 5
_DAG_WIDTH = 24
_DAG_SIZE = _DAG_LAYERS * _DAG_WIDTH
_INTERRUPTION_SEED = 243


def _run_identity(*, plan_hash: str = "b" * 64) -> EvaluationRunIdentity:
    return EvaluationRunIdentity(
        "global_regime_v4",
        "xetra",
        4,
        "a" * 64,
        1,
        plan_hash,
        "c" * 64,
        datetime(2026, 1, 1, tzinfo=UTC),
        "d" * 40,
        "e" * 64,
        "3.14.7",
    )


def _graph(identity: EvaluationRunIdentity) -> EvaluationWorkGraph:
    nodes: list[WorkUnitNode] = []
    for layer in range(_DAG_LAYERS):
        for index in range(_DAG_WIDTH):
            unit = make_work_unit(
                identity,
                unit_type="acceptance_dag",
                coordinates=(
                    ("layer", f"{layer:02d}"),
                    ("node", f"{index:02d}"),
                ),
                unit_parameters=(("contract", "pr243"),),
            )
            if layer == 0:
                parents: tuple[str, ...] = ()
            else:
                parents = tuple(
                    sorted(
                        (
                            make_work_unit(
                                identity,
                                unit_type="acceptance_dag",
                                coordinates=(
                                    ("layer", f"{layer - 1:02d}"),
                                    ("node", f"{index:02d}"),
                                ),
                                unit_parameters=(("contract", "pr243"),),
                            ).key,
                            make_work_unit(
                                identity,
                                unit_type="acceptance_dag",
                                coordinates=(
                                    ("layer", f"{layer - 1:02d}"),
                                    ("node", f"{(index + 1) % _DAG_WIDTH:02d}"),
                                ),
                                unit_parameters=(("contract", "pr243"),),
                            ).key,
                        )
                    )
                )
            nodes.append(WorkUnitNode(unit, parents))
    graph = EvaluationWorkGraph.from_nodes(identity, nodes)
    assert len(graph.nodes) == _DAG_SIZE
    return graph


def _stable_payload(node: WorkUnitNode, parents: tuple[bytes, ...]) -> bytes:
    node_bytes = node.identity.key.encode("utf-8")
    digest = sha256(b"\0".join((node_bytes, *parents))).hexdigest().encode("ascii")
    return node_bytes + b":" + digest


@dataclass(frozen=True, slots=True)
class _InterruptAt:
    """Pickleable process callback that fails at one deterministic DAG unit."""

    target_key: str

    def __call__(self, node: WorkUnitNode, parents: tuple[bytes, ...]) -> bytes:
        if node.identity.key == self.target_key:
            raise RuntimeError(f"seeded interruption at {self.target_key}")
        return _stable_payload(node, parents)


def _expected_payloads(graph: EvaluationWorkGraph) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for node in graph.nodes:
        payloads[node.identity.key] = _stable_payload(
            node, tuple(payloads[parent] for parent in node.parent_keys)
        )
    return payloads


def _effective_unit(
    identity: EvaluationRunIdentity,
    node: WorkUnitNode,
    payloads: dict[str, bytes],
) -> WorkUnitIdentity:
    return make_work_unit(
        identity,
        unit_type=node.identity.unit_type,
        coordinates=node.identity.coordinates,
        parent_payload_hashes=tuple(
            sha256(payloads[parent]).hexdigest() for parent in node.parent_keys
        ),
        unit_parameters=node.identity.unit_parameters,
    )


def test_seeded_restarts_of_large_dag_preserve_terminal_and_root_identity(
    tmp_path: Path,
) -> None:
    """Exercise PR-243 with 120 units, process workers, and five restarts."""

    run = _run_identity()
    graph = _graph(run)
    expected = _expected_payloads(graph)

    baseline_store = SQLiteEvaluationRunStore(tmp_path / "baseline")
    baseline = ResumableEvaluationExecutor(baseline_store, max_workers=_DAG_WIDTH).execute(
        run, graph, _stable_payload
    )
    assert baseline.payload

    interrupted_store = SQLiteEvaluationRunStore(tmp_path / "interrupted")
    executor = ResumableEvaluationExecutor(interrupted_store, max_workers=_DAG_WIDTH)
    by_layer = {
        layer: tuple(
            node
            for node in graph.nodes
            if dict(node.identity.coordinates)["layer"] == f"{layer:02d}"
        )
        for layer in range(_DAG_LAYERS)
    }
    randomizer = random.Random(_INTERRUPTION_SEED)
    targets = tuple(randomizer.choice(by_layer[layer]).identity.key for layer in range(_DAG_LAYERS))
    assert len(set(targets)) == _DAG_LAYERS

    for target in targets:
        with pytest.raises(RuntimeError, match="seeded interruption"):
            executor.execute(run, graph, _InterruptAt(target))
        target_node = next(node for node in graph.nodes if node.identity.key == target)
        state = interrupted_store.work_unit_state(run, _effective_unit(run, target_node, expected))
        assert state is not None
        assert state.status is WorkUnitStatus.PENDING

        # A changed dependency hash must not silently reuse the old structural
        # key.  Do this while the run is still resumable so the executor checks
        # the dependency contract instead of taking the completed-run fast path.
        if target == targets[0]:
            mutation_node = by_layer[1][0]
            actual_hashes = tuple(
                sha256(expected[parent]).hexdigest() for parent in mutation_node.parent_keys
            )
            changed_hash = sha256(b"mutated-parent-payload").hexdigest()
            if changed_hash == actual_hashes[0]:
                changed_hash = sha256(b"mutated-parent-payload-2").hexdigest()
            mutated_hashes = (changed_hash, *actual_hashes[1:])
            mutated_identity = make_work_unit(
                run,
                unit_type=mutation_node.identity.unit_type,
                coordinates=mutation_node.identity.coordinates,
                parent_payload_hashes=mutated_hashes,
                unit_parameters=mutation_node.identity.unit_parameters,
            )
            mutated_nodes = list(graph.nodes)
            mutation_index = graph.nodes.index(mutation_node)
            mutated_nodes[mutation_index] = WorkUnitNode(
                mutated_identity, mutation_node.parent_keys
            )
            mutated_graph = EvaluationWorkGraph.from_nodes(run, mutated_nodes)
            with pytest.raises(ValueError, match="parent payload hash mismatch"):
                executor.execute(run, mutated_graph, _stable_payload)

    resumed = executor.execute(run, graph, _stable_payload)
    assert resumed.payload == baseline.payload
    assert resumed.root_evidence_hash == baseline.root_evidence_hash
    assert interrupted_store.load_completed_run(run) == baseline.payload
    assert interrupted_store.state(run).work_unit_count == _DAG_SIZE

    target_set = set(targets)
    for node in graph.nodes:
        unit = _effective_unit(run, node, expected)
        state = interrupted_store.work_unit_state(run, unit)
        payload = interrupted_store.load_completed_work_unit(run, unit)
        assert state is not None
        assert state.status is WorkUnitStatus.COMPLETE
        assert payload == expected[node.identity.key]
        assert state.attempt_count == (2 if node.identity.key in target_set else 1)

        # Repeating the identical terminal write is idempotent; a different
        # payload cannot create a second terminal result for the same unit.
        interrupted_store.complete_work_unit(run, unit, payload)
        assert interrupted_store.work_unit_state(run, unit) == state
        with pytest.raises(ValueError, match="immutable"):
            interrupted_store.complete_work_unit(run, unit, payload + b"-mutated")

    changed_run = replace(run, evaluation_plan_hash="f" * 64)
    changed_graph = _graph(changed_run)
    changed = ResumableEvaluationExecutor(interrupted_store, max_workers=_DAG_WIDTH).execute(
        changed_run, changed_graph, _stable_payload
    )
    assert changed.computed_unit_count == _DAG_SIZE
    assert changed.reused_unit_count == 0
    assert changed.payload != resumed.payload
    assert interrupted_store.state(changed_run).work_unit_count == _DAG_SIZE
