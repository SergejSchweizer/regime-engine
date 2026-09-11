from __future__ import annotations

from datetime import UTC, datetime

from market_regime_engine.evaluation_runs.contracts import EvaluationRunIdentity
from market_regime_engine.evaluation_runs.stages import StageCheckpoint
from market_regime_engine.evaluation_runs.store import SQLiteEvaluationRunStore


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
