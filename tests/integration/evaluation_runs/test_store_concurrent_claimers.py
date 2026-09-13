"""Focused process-level acceptance tests for SQLite work-unit claims."""

from __future__ import annotations

import multiprocessing
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from queue import Empty
from typing import Any

import pytest

from market_regime_engine.evaluation_runs.contracts import (
    EvaluationRunIdentity,
    WorkUnitIdentity,
)
from market_regime_engine.evaluation_runs.store import (
    SQLiteEvaluationRunStore,
    WorkUnitStatus,
)

pytestmark = pytest.mark.integration

_CLAIMER_COUNT = 12
_PROCESS_TIMEOUT_SECONDS = 20
_RUN_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_CANONICAL_PAYLOAD = b"one-canonical-terminal-payload"
_STALE_PAYLOAD = b"payload-from-expired-owner"


def _identity() -> EvaluationRunIdentity:
    return EvaluationRunIdentity(
        "global_regime_v4",
        "xetra",
        4,
        "a" * 64,
        1,
        "b" * 64,
        "c" * 64,
        _RUN_NOW,
        "d" * 40,
        "e" * 64,
        "3.14.7",
    )


def _unit(identity: EvaluationRunIdentity) -> WorkUnitIdentity:
    return WorkUnitIdentity(
        identity.key,
        "candidate_seed",
        (("candidate_id", "gaussian_hmm_k2_full"), ("seed", "11")),
    )


def _claim_and_complete(
    root: str,
    start_gate: Any,
    result_queue: Any,
    worker_id: int,
) -> None:
    """Race one independently spawned process through claim and completion."""

    try:
        identity = _identity()
        store = SQLiteEvaluationRunStore(root)
        start_gate.wait(timeout=_PROCESS_TIMEOUT_SECONDS)
        claimed = store.claim_work_unit(
            identity,
            _unit(identity),
            lease_seconds=60,
            owner=f"claimer-{worker_id}",
        )
        if claimed:
            store.complete_work_unit(
                identity,
                _unit(identity),
                _CANONICAL_PAYLOAD,
                owner=f"claimer-{worker_id}",
            )
        result_queue.put({"worker_id": worker_id, "claimed": claimed, "error": None})
    except BaseException as exc:  # return child failures to the parent assertion
        result_queue.put(
            {
                "worker_id": worker_id,
                "claimed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def _run_concurrent_claimers(root: Path, count: int = _CLAIMER_COUNT) -> list[dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    start_gate = context.Barrier(count + 1)
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_claim_and_complete,
            args=(str(root), start_gate, result_queue, worker_id),
        )
        for worker_id in range(count)
    ]
    for process in processes:
        process.start()

    try:
        start_gate.wait(timeout=_PROCESS_TIMEOUT_SECONDS)
        for process in processes:
            process.join(timeout=_PROCESS_TIMEOUT_SECONDS)
        stuck = [process.pid for process in processes if process.is_alive()]
        if stuck:
            pytest.fail(f"spawned claimers did not terminate: {stuck}")

        results: list[dict[str, Any]] = []
        for _ in processes:
            try:
                results.append(result_queue.get(timeout=_PROCESS_TIMEOUT_SECONDS))
            except Empty as exc:
                pytest.fail("a spawned claimer returned no result")
                raise AssertionError from exc
        assert [process.exitcode for process in processes] == [0] * count
        assert not [result for result in results if result["error"]], results
        return results
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()


def _expire_lease(store: SQLiteEvaluationRunStore, identity: EvaluationRunIdentity) -> None:
    unit = _unit(identity)
    expired = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    with store._connect() as connection:
        updated = connection.execute(
            """
            UPDATE work_units
            SET lease_expires_at=?
            WHERE run_key=? AND work_unit_key=? AND status='RUNNING'
            """,
            (expired, identity.key, unit.key),
        ).rowcount
    assert updated == 1


def _claim_and_hold(
    root: str,
    claimed_event: Any,
) -> None:
    """Claim a unit and remain alive until the parent kills this stale owner."""

    identity = _identity()
    store = SQLiteEvaluationRunStore(root)
    claimed = store.claim_work_unit(
        identity,
        _unit(identity),
        lease_seconds=3_600,
        owner="stale-owner",
    )
    assert claimed
    claimed_event.set()
    # The parent terminates this process after making its lease stale.
    claimed_event.wait(timeout=_PROCESS_TIMEOUT_SECONDS)


def _stale_owner_waits_then_completes(
    root: str,
    stale_claimed_event: Any,
    reclaimer_claimed_event: Any,
    result_queue: Any,
) -> None:
    """Attempt a late write from the old owner after a new claim succeeds."""

    try:
        identity = _identity()
        store = SQLiteEvaluationRunStore(root)
        claimed = store.claim_work_unit(
            identity,
            _unit(identity),
            lease_seconds=3_600,
            owner="stale-owner",
        )
        assert claimed
        stale_claimed_event.set()
        assert reclaimer_claimed_event.wait(timeout=_PROCESS_TIMEOUT_SECONDS)
        try:
            store.complete_work_unit(
                identity,
                _unit(identity),
                _STALE_PAYLOAD,
                owner="stale-owner",
            )
        except ValueError as exc:
            result_queue.put({"role": "stale", "status": "rejected", "error": str(exc)})
        else:
            result_queue.put({"role": "stale", "status": "completed", "error": None})
    except BaseException as exc:
        result_queue.put(
            {"role": "stale", "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        )


def _reclaimer_claims_then_completes(
    root: str,
    start_event: Any,
    claimed_event: Any,
    release_event: Any,
    result_queue: Any,
) -> None:
    """Hold the reclaimed claim until the stale-owner write has been tested."""

    try:
        identity = _identity()
        store = SQLiteEvaluationRunStore(root)
        assert start_event.wait(timeout=_PROCESS_TIMEOUT_SECONDS)
        claimed = store.claim_work_unit(
            identity,
            _unit(identity),
            lease_seconds=60,
            owner="reclaimer",
        )
        assert claimed
        claimed_event.set()
        assert release_event.wait(timeout=_PROCESS_TIMEOUT_SECONDS)
        try:
            store.complete_work_unit(
                identity,
                _unit(identity),
                _CANONICAL_PAYLOAD,
                owner="reclaimer",
            )
        except ValueError as exc:
            result_queue.put({"role": "reclaimer", "status": "rejected", "error": str(exc)})
        else:
            result_queue.put({"role": "reclaimer", "status": "completed", "error": None})
    except BaseException as exc:
        result_queue.put(
            {
                "role": "reclaimer",
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def test_many_spawned_claimers_have_one_owner_and_one_terminal_payload(tmp_path: Path) -> None:
    """Concurrent claimers must converge on one durable, immutable result."""

    identity = _identity()
    store = SQLiteEvaluationRunStore(tmp_path)
    store.open_run(identity)
    store.enable_write_ahead_logging()

    results = _run_concurrent_claimers(tmp_path)

    assert sum(bool(result["claimed"]) for result in results) == 1
    state = store.work_unit_state(identity, _unit(identity))
    assert state is not None
    assert state.status is WorkUnitStatus.COMPLETE
    assert state.attempt_count == 1
    assert state.payload_hash == sha256(_CANONICAL_PAYLOAD).hexdigest()
    assert store.load_completed_work_unit(identity, _unit(identity)) == _CANONICAL_PAYLOAD
    with store._connect() as connection:
        row = connection.execute(
            "SELECT status, payload, payload_hash, lease_owner, lease_expires_at "
            "FROM work_units WHERE run_key=? AND work_unit_key=?",
            (identity.key, _unit(identity).key),
        ).fetchone()
    assert row is not None
    assert tuple(row) == (
        WorkUnitStatus.COMPLETE.value,
        _CANONICAL_PAYLOAD,
        sha256(_CANONICAL_PAYLOAD).hexdigest(),
        None,
        None,
    )
    with pytest.raises(ValueError, match="immutable"):
        store.complete_work_unit(identity, _unit(identity), b"contradictory-payload")


def test_many_spawned_reclaimers_safely_reclaim_one_stale_lease(tmp_path: Path) -> None:
    """After a dead owner lease expires, one reclaimer may publish the result."""

    context = multiprocessing.get_context("spawn")
    identity = _identity()
    store = SQLiteEvaluationRunStore(tmp_path)
    store.open_run(identity)
    store.enable_write_ahead_logging()
    claimed_event = context.Event()
    stale_owner = context.Process(
        target=_claim_and_hold,
        args=(str(tmp_path), claimed_event),
    )
    stale_owner.start()
    assert claimed_event.wait(timeout=_PROCESS_TIMEOUT_SECONDS)
    stale_owner.terminate()
    stale_owner.join(timeout=_PROCESS_TIMEOUT_SECONDS)
    assert stale_owner.exitcode is not None
    _expire_lease(store, identity)

    results = _run_concurrent_claimers(tmp_path)

    assert sum(bool(result["claimed"]) for result in results) == 1
    state = store.work_unit_state(identity, _unit(identity))
    assert state is not None
    assert state.status is WorkUnitStatus.COMPLETE
    assert state.attempt_count == 2
    assert store.load_completed_work_unit(identity, _unit(identity)) == _CANONICAL_PAYLOAD


def test_expired_owner_cannot_publish_after_reclaimer_claim(tmp_path: Path) -> None:
    """A stale owner must not write a contradictory payload after reclamation."""

    context = multiprocessing.get_context("spawn")
    identity = _identity()
    store = SQLiteEvaluationRunStore(tmp_path)
    store.open_run(identity)
    store.enable_write_ahead_logging()
    stale_claimed_event = context.Event()
    reclaimer_start_event = context.Event()
    reclaimer_claimed_event = context.Event()
    release_reclaimer_event = context.Event()
    result_queue = context.Queue()
    stale_owner = context.Process(
        target=_stale_owner_waits_then_completes,
        args=(str(tmp_path), stale_claimed_event, reclaimer_claimed_event, result_queue),
    )
    reclaimer = context.Process(
        target=_reclaimer_claims_then_completes,
        args=(
            str(tmp_path),
            reclaimer_start_event,
            reclaimer_claimed_event,
            release_reclaimer_event,
            result_queue,
        ),
    )
    stale_owner.start()
    try:
        assert stale_claimed_event.wait(timeout=_PROCESS_TIMEOUT_SECONDS)
        _expire_lease(store, identity)
        reclaimer.start()
        reclaimer_start_event.set()
        assert reclaimer_claimed_event.wait(timeout=_PROCESS_TIMEOUT_SECONDS)
        stale_result = result_queue.get(timeout=_PROCESS_TIMEOUT_SECONDS)
        release_reclaimer_event.set()
        reclaimer_result = result_queue.get(timeout=_PROCESS_TIMEOUT_SECONDS)
        stale_owner.join(timeout=_PROCESS_TIMEOUT_SECONDS)
        reclaimer.join(timeout=_PROCESS_TIMEOUT_SECONDS)
    finally:
        for process in (stale_owner, reclaimer):
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()

    assert stale_owner.exitcode == 0
    assert reclaimer.exitcode == 0
    assert stale_result["role"] == "stale"
    assert stale_result["status"] == "rejected", (
        "stale owner was allowed to publish after reclamation; "
        "complete_work_unit must verify the lease owner: "
        f"{stale_result}"
    )
    assert stale_result["error"]
    assert reclaimer_result["role"] == "reclaimer"
    assert reclaimer_result["status"] == "completed", reclaimer_result
    assert store.load_completed_work_unit(identity, _unit(identity)) == _CANONICAL_PAYLOAD
