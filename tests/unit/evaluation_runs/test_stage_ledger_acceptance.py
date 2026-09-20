"""Focused acceptance tests for the fine-grained stage ledger contract."""

from __future__ import annotations

import pickle
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from market_regime_engine.evaluation.errors import RecoverableEvaluationInvalidity
from market_regime_engine.evaluation_runs.contracts import EvaluationRunIdentity
from market_regime_engine.evaluation_runs.stages import StageCheckpoint
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


def _checkpoint(tmp_path: Path) -> tuple[StageCheckpoint, SQLiteEvaluationRunStore]:
    identity = _identity()
    store = SQLiteEvaluationRunStore(tmp_path / "runs")
    store.open_run(identity)
    return StageCheckpoint(identity, store, "outer_fold_001"), store


def test_every_stage_identity_carries_scope_stage_parents_and_parameters(
    tmp_path: Path,
) -> None:
    checkpoint, _store = _checkpoint(tmp_path)
    parent_payloads = (b"parent-a", b"parent-b")
    parameters = (("state_count", "3"), ("candidate", "gaussian"))

    unit = checkpoint._unit("prefix_search", parameters, parent_payloads)

    assert unit.unit_type == "v4_stage"
    assert dict(unit.coordinates) == {
        "scope": "outer_fold_001",
        "stage": "prefix_search",
    }
    assert unit.parent_payload_hashes == tuple(
        sha256(payload).hexdigest() for payload in parent_payloads
    )
    assert unit.unit_parameters == tuple(sorted(parameters))

    # Each contract component participates in the deterministic identity hash.
    assert (
        unit.work_unit_input_hash
        != checkpoint._unit("other_stage", parameters, parent_payloads).work_unit_input_hash
    )
    assert (
        unit.work_unit_input_hash
        != checkpoint._unit(
            "prefix_search", (("state_count", "4"), ("candidate", "gaussian")), parent_payloads
        ).work_unit_input_hash
    )
    assert (
        unit.work_unit_input_hash
        != checkpoint._unit(
            "prefix_search", parameters, (b"different-parent", b"parent-b")
        ).work_unit_input_hash
    )


def test_domain_invalid_terminal_payload_is_immutable_and_not_recomputed(tmp_path: Path) -> None:
    checkpoint, store = _checkpoint(tmp_path)
    calls = 0

    def invalid() -> object:
        nonlocal calls
        calls += 1
        raise RecoverableEvaluationInvalidity("no eligible prefix")

    with pytest.raises(ValueError, match="no eligible prefix"):
        checkpoint.run("prefix_search", invalid)

    unit = checkpoint._unit("prefix_search", (), ())
    state = store.work_unit_state(checkpoint.identity, unit)
    assert state is not None
    assert state.status is WorkUnitStatus.DOMAIN_INVALID
    original_payload = store.load_completed_work_unit(checkpoint.identity, unit)
    assert original_payload is not None

    with pytest.raises(ValueError, match="no eligible prefix"):
        checkpoint.run("prefix_search", invalid)
    assert calls == 1
    assert store.load_completed_work_unit(checkpoint.identity, unit) == original_payload

    # A terminal unit cannot be overwritten with another payload or status.
    with pytest.raises(ValueError, match="terminal work unit is immutable"):
        store.complete_work_unit(
            checkpoint.identity,
            unit,
            b"different-terminal-payload",
            domain_invalid=True,
        )


def test_technical_failure_returns_stage_to_retryable_pending(tmp_path: Path) -> None:
    checkpoint, store = _checkpoint(tmp_path)
    attempts = 0

    def flaky() -> dict[str, int]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary infrastructure failure")
        return {"value": 7}

    with pytest.raises(RuntimeError, match="temporary infrastructure failure"):
        checkpoint.run("distance", flaky)

    unit = checkpoint._unit("distance", (), ())
    failed_state = store.work_unit_state(checkpoint.identity, unit)
    assert failed_state is not None
    assert failed_state.status is WorkUnitStatus.PENDING
    assert failed_state.payload_hash is None
    assert store.load_completed_work_unit(checkpoint.identity, unit) is None

    assert checkpoint.run("distance", flaky) == {"value": 7}
    retried_state = store.work_unit_state(checkpoint.identity, unit)
    assert retried_state is not None
    assert retried_state.status is WorkUnitStatus.COMPLETE
    assert retried_state.attempt_count == 2


def test_completed_stage_reuses_exact_cached_payload(tmp_path: Path) -> None:
    checkpoint, store = _checkpoint(tmp_path)
    value = {
        "ordered": [3, 1, 4],
        "nested": {"alpha": "immutable", "beta": ("x", "y")},
    }
    expected_payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    calls = 0

    def compute() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return value

    assert checkpoint.run("quality", compute) == value
    unit = checkpoint._unit("quality", (), ())
    assert store.load_completed_work_unit(checkpoint.identity, unit) == expected_payload
    assert checkpoint.run("quality", compute) == value
    assert calls == 1
