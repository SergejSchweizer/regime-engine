from __future__ import annotations

from datetime import UTC, datetime

import pytest

from market_regime_engine.evaluation_runs.contracts import EvaluationRunIdentity
from market_regime_engine.evaluation_runs.stages import StageCheckpoint
from market_regime_engine.evaluation_runs.store import (
    SQLiteEvaluationRunStore,
    WorkUnitStatus,
)


def test_stage_checkpoint_reuses_completed_payload(tmp_path) -> None:
    identity = EvaluationRunIdentity(
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
    store = SQLiteEvaluationRunStore(tmp_path / "runs")
    store.open_run(identity)
    checkpoint = StageCheckpoint(identity, store, "fold_001")
    calls = 0

    def compute() -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"value": 42}

    assert checkpoint.run("quality", compute) == {"value": 42}
    assert checkpoint.run("quality", compute) == {"value": 42}
    assert calls == 1


def test_stage_checkpoint_persists_and_rethrows_domain_invalid(tmp_path) -> None:
    identity = EvaluationRunIdentity(
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
    store = SQLiteEvaluationRunStore(tmp_path / "runs")
    store.open_run(identity)
    checkpoint = StageCheckpoint(identity, store, "fold_001")
    calls = 0

    def compute() -> dict[str, int]:
        nonlocal calls
        calls += 1
        raise ValueError("no eligible prefix")

    with pytest.raises(ValueError, match="no eligible prefix"):
        checkpoint.run("prefix_search", compute)

    unit = checkpoint._unit("prefix_search", (), ())
    state = store.work_unit_state(identity, unit)
    assert state is not None
    assert state.status is WorkUnitStatus.DOMAIN_INVALID

    with pytest.raises(ValueError, match="no eligible prefix"):
        checkpoint.run("prefix_search", compute)
    assert calls == 1


def test_stage_store_repairs_only_pending_fold_scope(tmp_path) -> None:
    identity = EvaluationRunIdentity(
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
    store = SQLiteEvaluationRunStore(tmp_path / "runs")
    store.open_run(identity)
    checkpoint = StageCheckpoint(identity, store, "fold_001")
    unit = checkpoint._unit("teacher_reference", (), ())
    assert store.claim_work_unit(identity, unit)

    with pytest.raises(RuntimeError, match="remains claimed"):
        store.terminalize_pending_stage_units_for_invalid_outer_fold(
            identity,
            "fold_001",
            "TRAIN-only selection failed",
        )
    # A claimed unit is never mutated by the repair path.
    assert store.work_unit_state(identity, unit).status is WorkUnitStatus.RUNNING

    store.fail_work_unit(identity, unit, "test lease release")
    assert (
        store.terminalize_pending_stage_units_for_invalid_outer_fold(
            identity,
            "fold_001",
            "TRAIN-only selection failed",
        )
        == 1
    )
    assert store.work_unit_state(identity, unit).status is WorkUnitStatus.DOMAIN_INVALID
