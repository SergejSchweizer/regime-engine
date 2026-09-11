"""Crash-safe deterministic executor for evaluation work graphs."""

from __future__ import annotations

from collections.abc import Callable
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

    def __init__(self, store: SQLiteEvaluationRunStore) -> None:
        self._store = store

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
        for node in graph.nodes:
            unit = node.identity
            parent_payloads = tuple(payloads[parent] for parent in node.parent_keys)
            parent_hashes = tuple(sha256(payload).hexdigest() for payload in parent_payloads)
            if unit.parent_payload_hashes and unit.parent_payload_hashes != parent_hashes:
                raise ValueError(f"parent payload hash mismatch for {unit.key}")
            if parent_hashes != unit.parent_payload_hashes:
                unit = replace(unit, parent_payload_hashes=parent_hashes)
            execution_node = replace(node, identity=unit)
            cached, claimed = self._load_or_claim(identity, execution_node)
            if cached is not None:
                payloads[unit.key] = cached
                reused += 1
                continue
            if not all(parent in payloads for parent in node.parent_keys):
                raise ValueError(f"work graph parent payload is unavailable for {unit.key}")
            if not claimed:
                raise RuntimeError(f"work unit claim was not acquired: {unit.key}")
            try:
                result = compute(execution_node, parent_payloads)
            except DomainInvalid as exc:
                invalid_payload = canonical_json({"status": "DOMAIN_INVALID", "reason": str(exc)})
                self._store.complete_work_unit(
                    identity,
                    unit,
                    invalid_payload,
                    domain_invalid=True,
                )
                payloads[unit.key] = invalid_payload
                computed += 1
                continue
            except Exception:
                self._store.fail_work_unit(identity, unit, "technical compute failure")
                raise
            if not isinstance(result, bytes) or not result:
                self._store.fail_work_unit(
                    identity,
                    unit,
                    "compute callback returned invalid payload",
                )
                raise ValueError("compute callback must return non-empty bytes")
            self._store.complete_work_unit(identity, unit, result)
            payloads[unit.key] = result
            computed += 1

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
