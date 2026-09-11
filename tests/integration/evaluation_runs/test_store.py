from __future__ import annotations

from datetime import UTC, datetime, timedelta
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
