from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from market_regime_engine.evaluation_runs.contracts import (
    EvaluationRunIdentity,
    WorkUnitIdentity,
)
from market_regime_engine.evaluation_runs.store import (
    SQLiteEvaluationRunStore,
    WorkUnitStatus,
)


def _identity() -> EvaluationRunIdentity:
    return EvaluationRunIdentity(
        evaluation_id="global_regime_v4",
        profile_id="xetra",
        profile_config_version=4,
        profile_hash="a" * 64,
        evaluation_contract_version=1,
        evaluation_plan_hash="b" * 64,
        dataset_snapshot_key="c" * 64,
        evaluation_cutoff=datetime(2026, 1, 1, tzinfo=UTC),
        repository_commit_sha="d" * 40,
        uv_lock_sha256="e" * 64,
        python_version="3.14.7",
    )


def _unit(identity: EvaluationRunIdentity, name: str = "stage") -> WorkUnitIdentity:
    return WorkUnitIdentity(
        evaluation_run_key=identity.key,
        unit_type=name,
        coordinates=(("fold", "001"),),
    )


def test_store_claim_complete_resume_and_run_finalization(tmp_path) -> None:
    identity = _identity()
    store = SQLiteEvaluationRunStore(tmp_path / "runs")
    store.enable_write_ahead_logging()
    assert store.open_run(identity).status == "RUNNING"
    assert store.load_identity(identity.key) == identity
    unit = _unit(identity)
    assert store.work_unit_state(identity, unit) is None
    assert store.claim_work_unit(identity, unit, owner="worker-a")
    assert not store.claim_work_unit(identity, unit, owner="worker-b")
    state = store.work_unit_state(identity, unit)
    assert state is not None
    assert state.status is WorkUnitStatus.RUNNING
    assert state.attempt_count == 1
    with pytest.raises(ValueError, match="lease owner"):
        store.complete_work_unit(identity, unit, b"payload", owner="worker-b")
    store.complete_work_unit(identity, unit, b"payload", owner="worker-a")
    assert store.load_completed_work_unit(identity, unit) == b"payload"
    assert store.load_completed_work_unit_payloads(identity, "stage/") == ((unit.key, b"payload"),)
    # Identical terminal writes are idempotent; a changed payload is not.
    store.complete_work_unit(identity, unit, b"payload", owner="worker-a")
    with pytest.raises(ValueError, match="immutable"):
        store.complete_work_unit(identity, unit, b"different", owner="worker-a")
    store.complete_run(identity, "f" * 64, b"final")
    assert store.load_completed_run(identity) == b"final"
    assert store.state(identity).status == "COMPLETE"
    assert not store.claim_work_unit(identity, _unit(identity, "other"))
    store.complete_run(identity, "f" * 64, b"final")


def test_store_releases_technical_failures_and_terminalizes_domain_units(tmp_path) -> None:
    identity = _identity()
    store = SQLiteEvaluationRunStore(tmp_path / "runs")
    store.open_run(identity)
    retry = _unit(identity, "retry")
    assert store.claim_work_unit(identity, retry, owner="worker")
    with pytest.raises(ValueError, match="technical failure"):
        store.fail_work_unit(identity, retry, " ", owner="worker")
    store.fail_work_unit(identity, retry, "temporary failure", owner="worker")
    assert store.work_unit_state(identity, retry).status is WorkUnitStatus.PENDING
    assert store.claim_work_unit(identity, retry, owner="worker")
    store.complete_work_unit(identity, retry, b"retry-result", owner="worker")

    stage = WorkUnitIdentity(
        identity.key,
        "v4_stage",
        coordinates=(("scope", "fold_001"), ("stage", "quality")),
    )
    assert store.claim_work_unit(identity, stage, owner="worker")
    with pytest.raises(RuntimeError, match="remains claimed"):
        store.terminalize_pending_stage_units_for_invalid_outer_fold(
            identity, "fold_001", "invalid fold"
        )
    store.fail_work_unit(identity, stage, "release", owner="worker")
    assert (
        store.terminalize_pending_stage_units_for_invalid_outer_fold(
            identity, "fold_001", "invalid fold"
        )
        == 1
    )
    assert store.work_unit_state(identity, stage).status is WorkUnitStatus.DOMAIN_INVALID
    assert store.load_completed_work_unit(identity, stage) is not None


def test_store_rejects_invalid_inputs_and_missing_ledgers(tmp_path) -> None:
    identity = _identity()
    store = SQLiteEvaluationRunStore(tmp_path / "runs")
    unit = _unit(identity)
    with pytest.raises(ValueError, match="opened before claiming"):
        store.claim_work_unit(identity, unit)
    with pytest.raises(ValueError, match="positive"):
        store.claim_work_unit(identity, unit, lease_seconds=0)
    with pytest.raises(ValueError, match="non-empty bytes"):
        store.complete_work_unit(identity, unit, b"")
    with pytest.raises(ValueError, match="key_prefix"):
        store.load_completed_work_unit_payloads(identity, " ")
    with pytest.raises(ValueError, match="missing"):
        store.load_completed_run(identity)
    with pytest.raises(ValueError, match="fold_id"):
        store.terminalize_pending_stage_units_for_invalid_outer_fold(identity, "bad/fold", "reason")

    # The database remains inspectable and has the expected durable schema.
    with sqlite3.connect(store._database) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
    assert {"runs", "work_units"} <= tables
