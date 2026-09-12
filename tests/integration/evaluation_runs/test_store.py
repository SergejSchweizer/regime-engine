from __future__ import annotations

import multiprocessing
import os
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest

from market_regime_engine.evaluation_runs.contracts import (
    EvaluationRunIdentity,
    WorkUnitIdentity,
)
from market_regime_engine.evaluation_runs.store import (
    SQLiteEvaluationRunStore,
)

pytestmark = pytest.mark.integration
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def identity() -> EvaluationRunIdentity:
    return EvaluationRunIdentity(
        "global_regime_v4",
        "xetra",
        4,
        "a" * 64,
        1,
        "b" * 64,
        "c" * 64,
        NOW,
        "d" * 40,
        "e" * 64,
        "3.14.7",
    )


def unit(run: EvaluationRunIdentity, seed: str = "11") -> WorkUnitIdentity:
    return WorkUnitIdentity(
        run.key,
        "candidate_seed",
        (("candidate_id", "gaussian_hmm_k2_full"), ("seed", seed)),
    )


def _crash_after_uncommitted_payload(root: str, mode: str) -> None:
    run = identity()
    work_unit = unit(run, "71")
    store = SQLiteEvaluationRunStore(root)
    store.open_run(run)
    assert store.claim_work_unit(run, work_unit, lease_seconds=3_600)
    payload = b"payload-written-before-ledger-commit"
    if mode == "before_commit":
        with store._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE work_units SET payload=?, payload_hash=? "
                "WHERE run_key=? AND work_unit_key=?",
                (payload, sha256(payload).hexdigest(), run.key, work_unit.key),
            )
            os._exit(17)
    store.complete_work_unit(run, work_unit, payload)
    os._exit(0)


def test_sqlite_claim_completion_idempotency_and_terminal_invalid(tmp_path: Path) -> None:
    run = identity()
    store = SQLiteEvaluationRunStore(tmp_path)
    store.open_run(run)
    first = unit(run)
    assert store.claim_work_unit(run, first, owner="worker-a")
    assert not store.claim_work_unit(run, first, owner="worker-b")
    payload = b"canonical-unit-payload"
    store.complete_work_unit(run, first, payload)
    assert store.load_completed_work_unit(run, first) == payload
    assert not store.claim_work_unit(run, first)
    with pytest.raises(ValueError, match="immutable"):
        store.complete_work_unit(run, first, b"different")

    invalid = unit(run, "23")
    assert store.claim_work_unit(run, invalid)
    store.complete_work_unit(run, invalid, b"domain-invalid", domain_invalid=True)
    assert store.load_completed_work_unit(run, invalid) == b"domain-invalid"
    assert not store.claim_work_unit(run, invalid)

    final = b"root-evidence"
    store.complete_run(run, "f" * 64, final)
    assert store.load_completed_run(run) == final
    assert store.state(run).status == "COMPLETE"
    assert not store.claim_work_unit(run, unit(run, "37"))
    assert store.load_identity(run.key) == run


def test_sqlite_reclaims_expired_lease_and_retries_technical_failure(tmp_path: Path) -> None:
    run = identity()
    store = SQLiteEvaluationRunStore(tmp_path)
    store.open_run(run)
    pending = unit(run, "53")
    assert store.claim_work_unit(run, pending, lease_seconds=1, owner="worker-a")
    with store._connect() as connection:  # test the persisted lease boundary directly
        connection.execute(
            "UPDATE work_units SET lease_expires_at=? WHERE run_key=? AND work_unit_key=?",
            ((NOW - timedelta(days=1)).isoformat(), run.key, pending.key),
        )
    assert store.claim_work_unit(run, pending, owner="worker-b")
    store.fail_work_unit(run, pending, "temporary database outage")
    assert store.claim_work_unit(run, pending, owner="worker-c")


def test_process_exit_before_ledger_commit_never_creates_false_complete(
    tmp_path: Path,
) -> None:
    run = identity()
    work_unit = unit(run, "71")
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_crash_after_uncommitted_payload,
        args=(str(tmp_path), "before_commit"),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 17

    store = SQLiteEvaluationRunStore(tmp_path)
    state = store.work_unit_state(run, work_unit)
    assert state is not None
    assert state.status.value == "RUNNING"
    assert store.load_completed_work_unit(run, work_unit) is None

    with store._connect() as connection:
        connection.execute(
            "UPDATE work_units SET lease_expires_at=? WHERE run_key=? AND work_unit_key=?",
            ((NOW - timedelta(days=1)).isoformat(), run.key, work_unit.key),
        )
    assert store.claim_work_unit(run, work_unit, owner="recovery-worker")
    store.complete_work_unit(run, work_unit, b"recomputed-after-crash")
    assert store.load_completed_work_unit(run, work_unit) == b"recomputed-after-crash"


def test_process_exit_after_ledger_commit_reuses_terminal_payload(tmp_path: Path) -> None:
    run = identity()
    work_unit = unit(run, "71")
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_crash_after_uncommitted_payload,
        args=(str(tmp_path), "after_commit"),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 0

    store = SQLiteEvaluationRunStore(tmp_path)
    assert store.load_completed_work_unit(run, work_unit) == b"payload-written-before-ledger-commit"
