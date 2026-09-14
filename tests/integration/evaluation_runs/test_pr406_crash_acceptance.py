"""Hermetic PR-406 crash-boundary acceptance evidence."""

from __future__ import annotations

import multiprocessing
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from market_regime_engine.commands.v4_backend import _configured_state_root
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

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _identity() -> EvaluationRunIdentity:
    return EvaluationRunIdentity(
        "global_regime_v4",
        "xetra",
        4,
        "a" * 64,
        1,
        "b" * 64,
        "c" * 64,
        _NOW,
        "d" * 40,
        "e" * 64,
        "3.14.7",
    )


def _node(identity: EvaluationRunIdentity) -> WorkUnitIdentity:
    return make_work_unit(
        identity,
        unit_type="filesystem_crash_boundary",
        coordinates=(("unit", "000"),),
    )


def _graph(identity: EvaluationRunIdentity) -> EvaluationWorkGraph:
    return EvaluationWorkGraph.from_nodes(identity, (WorkUnitNode(_node(identity)),))


def _kill_after_filesystem_write(root: str) -> None:
    """Leave a filesystem artifact, then kill before executor completion."""

    identity = _identity()
    artifact = Path(root) / "partial-artifact.bin"
    artifact.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    store = SQLiteEvaluationRunStore(root)
    store.open_run(identity)

    def compute(_node: WorkUnitNode, _parents: tuple[bytes, ...]) -> bytes:
        with artifact.open("wb") as handle:
            handle.write(b"written-before-ledger-commit")
            handle.flush()
            os.fsync(handle.fileno())
        os._exit(23)

    ResumableEvaluationExecutor(store, max_workers=1).execute(
        identity,
        _graph(identity),
        compute,
    )


def _stable_compute(node: WorkUnitNode, _parents: tuple[bytes, ...]) -> bytes:
    return b"canonical-payload:" + node.identity.key.encode("utf-8")


def _canonical_payload(unit: WorkUnitIdentity) -> bytes:
    return b"canonical-payload:" + unit.key.encode("utf-8")


def _expire_claim(root: Path, identity: EvaluationRunIdentity, unit: WorkUnitIdentity) -> None:
    store = SQLiteEvaluationRunStore(root)
    with store._connect() as connection:
        connection.execute(
            "UPDATE work_units SET lease_expires_at=? WHERE run_key=? AND work_unit_key=?",
            ("2000-01-01T00:00:00+00:00", identity.key, unit.key),
        )


def test_configured_deployment_volume_restarts_after_process_kill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured external state root remains recoverable after SIGKILL-like exit."""

    repository = tmp_path / "checkout"
    repository.mkdir()
    volume = tmp_path / "mounted-deployment-volume"
    monkeypatch.setenv("REGIME_ENGINE_STATE_ROOT", str(volume))
    state_root = _configured_state_root(repository)
    ledger_root = state_root / "evaluation-runs"
    identity = _identity()
    unit = _node(identity)

    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_kill_after_filesystem_write, args=(str(ledger_root),))
    process.start()
    process.join(timeout=20)
    assert process.exitcode == 23

    assert (ledger_root / "partial-artifact.bin").is_file()
    store = SQLiteEvaluationRunStore(ledger_root)
    state = store.work_unit_state(identity, unit)
    assert state is not None
    assert state.status is WorkUnitStatus.RUNNING
    assert store.load_completed_work_unit(identity, unit) is None

    _expire_claim(ledger_root, identity, unit)
    resumed = ResumableEvaluationExecutor(store, max_workers=1).execute(
        identity,
        _graph(identity),
        _stable_compute,
    )
    baseline = ResumableEvaluationExecutor(SQLiteEvaluationRunStore(tmp_path / "baseline")).execute(
        identity, _graph(identity), _stable_compute
    )

    assert resumed.payload == baseline.payload
    assert resumed.root_evidence_hash == baseline.root_evidence_hash
    assert resumed.computed_unit_count == 1
    assert resumed.reused_unit_count == 0
    final_state = store.work_unit_state(identity, unit)
    assert final_state is not None
    assert final_state.status is WorkUnitStatus.COMPLETE
    assert store.load_completed_work_unit(identity, unit) == _canonical_payload(unit)


def test_executor_filesystem_crash_boundary_never_fabricates_completion(
    tmp_path: Path,
) -> None:
    """A file written before a killed callback cannot become ledger evidence."""

    identity = _identity()
    unit = _node(identity)
    root = tmp_path / "filesystem-boundary"
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_kill_after_filesystem_write, args=(str(root),))
    process.start()
    process.join(timeout=20)
    assert process.exitcode == 23

    store = SQLiteEvaluationRunStore(root)
    state = store.work_unit_state(identity, unit)
    assert state is not None
    assert state.status is WorkUnitStatus.RUNNING
    assert store.load_completed_work_unit(identity, unit) is None
    assert (root / "partial-artifact.bin").read_bytes() == b"written-before-ledger-commit"

    _expire_claim(root, identity, unit)
    resumed = ResumableEvaluationExecutor(store, max_workers=1).execute(
        identity,
        _graph(identity),
        _stable_compute,
    )
    assert resumed.computed_unit_count == 1
    assert resumed.reused_unit_count == 0
    assert store.load_completed_work_unit(identity, unit) == _canonical_payload(unit)
