"""Crash-safe deterministic executor for evaluation work graphs."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, replace
from hashlib import sha256
from time import monotonic, sleep

from market_regime_engine.evaluation_runs.contracts import (
    EvaluationRunIdentity,
    canonical_json,
)
from market_regime_engine.evaluation_runs.graph import EvaluationWorkGraph, WorkUnitNode
from market_regime_engine.evaluation_runs.store import (
    SQLiteEvaluationRunStore,
    WorkUnitStatus,
)
from market_regime_engine.evaluations.process_parallel import cpu_process_pool


class DomainInvalid(Exception):
    """Raised by a compute callback for a deterministic terminal invalid result."""


@dataclass(frozen=True, slots=True)
class EvaluationExecutionResult:
    root_evidence_hash: str
    payload: bytes
    computed_unit_count: int
    reused_unit_count: int


ComputeCallback = Callable[[WorkUnitNode, tuple[bytes, ...]], bytes]


class ResumableEvaluationExecutor:
    """Execute only missing/reclaimable units and finalize one canonical root."""

    def __init__(
        self,
        store: SQLiteEvaluationRunStore,
        *,
        max_workers: int | None = None,
    ) -> None:
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be positive when provided")
        self._store = store
        self._max_workers = max_workers

    @staticmethod
    def _process_safe(compute: ComputeCallback) -> bool:
        """Return whether ``compute`` can be sent to a worker process."""

        try:
            pickle.dumps(compute)
        except AttributeError, OSError, pickle.PicklingError, TypeError:
            return False
        return True

    @staticmethod
    def _complete_result(
        store: SQLiteEvaluationRunStore,
        identity: EvaluationRunIdentity,
        node: WorkUnitNode,
        result: object,
    ) -> bytes:
        if not isinstance(result, bytes) or not result:
            store.fail_work_unit(
                identity, node.identity, "compute callback returned invalid payload"
            )
            raise ValueError("compute callback must return non-empty bytes")
        store.complete_work_unit(identity, node.identity, result)
        return result

    def _load_or_claim(
        self,
        identity: EvaluationRunIdentity,
        node: WorkUnitNode,
        *,
        wait_timeout_seconds: float = 30.0,
    ) -> tuple[bytes | None, bool]:
        """Reuse a terminal unit or claim it, waiting behind a live owner."""

        if wait_timeout_seconds <= 0.0:
            raise ValueError("wait_timeout_seconds must be positive")
        unit = node.identity
        started = monotonic()
        while True:
            cached = self._store.load_completed_work_unit(identity, unit)
            if cached is not None:
                return cached, False
            if self._store.claim_work_unit(identity, unit):
                return None, True
            state = self._store.work_unit_state(identity, unit)
            if state is None:
                # A concurrent transaction may have committed between the
                # failed claim and this read. Retry through the CAS path.
                continue
            if state.status in {WorkUnitStatus.COMPLETE, WorkUnitStatus.DOMAIN_INVALID}:
                continue
            if monotonic() - started >= wait_timeout_seconds:
                raise RuntimeError(f"work unit remains claimed: {unit.key}")
            sleep(0.01)

    def execute(
        self,
        identity: EvaluationRunIdentity,
        graph: EvaluationWorkGraph,
        compute: ComputeCallback,
    ) -> EvaluationExecutionResult:
        if graph.run_key != identity.key:
            raise ValueError("work graph does not belong to evaluation identity")
        state = self._store.open_run(identity)
        if state.status == "COMPLETE":
            payload = self._store.load_completed_run(identity)
            if payload is None:
                raise ValueError("completed run has no durable payload")
            return EvaluationExecutionResult(
                root_evidence_hash=state.root_identity_hash or "",
                payload=payload,
                computed_unit_count=0,
                reused_unit_count=len(graph.nodes),
            )

        payloads: dict[str, bytes] = {}
        computed = 0
        reused = 0
        remaining = {node.identity.key: node for node in graph.nodes}
        process_safe = self._process_safe(compute)
        while remaining:
            ready = tuple(
                node
                for node in graph.nodes
                if node.identity.key in remaining
                and all(parent in payloads for parent in node.parent_keys)
            )
            if not ready:
                raise ValueError("work graph parent payload is unavailable")
            # A non-pickleable callback must preserve the historical
            # one-unit-at-a-time semantics, including interruption boundaries.
            if not process_safe:
                ready = ready[:1]

            claimed: list[tuple[WorkUnitNode, tuple[bytes, ...]]] = []
            for node in ready:
                unit = node.identity
                parent_payloads = tuple(payloads[parent] for parent in node.parent_keys)
                parent_hashes = tuple(sha256(payload).hexdigest() for payload in parent_payloads)
                if unit.parent_payload_hashes and unit.parent_payload_hashes != parent_hashes:
                    raise ValueError(f"parent payload hash mismatch for {unit.key}")
                if parent_hashes != unit.parent_payload_hashes:
                    unit = replace(unit, parent_payload_hashes=parent_hashes)
                execution_node = replace(node, identity=unit)
                cached, acquired = self._load_or_claim(identity, execution_node)
                remaining.pop(node.identity.key)
                if cached is not None:
                    payloads[unit.key] = cached
                    # Keep the structural alias available to children when a
                    # caller omitted parent hashes from the graph identity.
                    payloads[node.identity.key] = cached
                    reused += 1
                else:
                    if not acquired:
                        raise RuntimeError(f"work unit claim was not acquired: {unit.key}")
                    claimed.append((execution_node, parent_payloads))

            if not claimed:
                continue

            results: list[tuple[WorkUnitNode, object | None, Exception | None]] = []
            use_processes = process_safe and len(claimed) > 1
            if use_processes:
                worker_limit = min(len(claimed), self._max_workers or len(claimed))
                with cpu_process_pool(worker_limit) as process_pool:
                    futures: list[tuple[WorkUnitNode, Future[bytes]]] = [
                        (node, process_pool.submit(compute, node, parents))
                        for node, parents in claimed
                    ]
                    for node, future in futures:
                        try:
                            results.append((node, future.result(), None))
                        except Exception as exc:
                            results.append((node, None, exc))
            else:
                for node, parents in claimed:
                    try:
                        results.append((node, compute(node, parents), None))
                    except Exception as exc:
                        results.append((node, None, exc))

            first_error: Exception | None = None
            for node, result, error in results:
                if error is not None:
                    if isinstance(error, DomainInvalid):
                        invalid_payload = canonical_json(
                            {"status": "DOMAIN_INVALID", "reason": str(error)}
                        )
                        self._store.complete_work_unit(
                            identity,
                            node.identity,
                            invalid_payload,
                            domain_invalid=True,
                        )
                        payloads[node.identity.key] = invalid_payload
                        computed += 1
                    else:
                        self._store.fail_work_unit(
                            identity, node.identity, "technical compute failure"
                        )
                        if first_error is None:
                            first_error = error
                    continue
                payloads[node.identity.key] = self._complete_result(
                    self._store, identity, node, result
                )
                computed += 1
            if first_error is not None:
                raise first_error

        root_payload = canonical_json(
            {
                "evaluation_run_key": identity.key,
                "units": [
                    {
                        "work_unit_key": key,
                        "payload_sha256": sha256(payloads[key]).hexdigest(),
                    }
                    for key in sorted(payloads)
                ],
            }
        )
        root_hash = sha256(root_payload).hexdigest()
        self._store.complete_run(identity, root_hash, root_payload)
        return EvaluationExecutionResult(root_hash, root_payload, computed, reused)


__all__ = [
    "DomainInvalid",
    "EvaluationExecutionResult",
    "ResumableEvaluationExecutor",
]
