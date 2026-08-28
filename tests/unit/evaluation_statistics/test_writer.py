from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

import market_regime_engine.evaluation_statistics.writer as writer_module
from market_regime_engine.evaluation_statistics import (
    RunStatistics,
    RunType,
    StatisticsWriter,
    Status,
)
from market_regime_engine.evaluations.contracts import EvaluationId


def statistics(
    evaluation_id: EvaluationId, status: Status, run_type: RunType = RunType.CANDIDATE
) -> RunStatistics:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return RunStatistics(
        evaluation_id=evaluation_id,
        mlflow_run_id="run-123",
        run_type=run_type,
        run_name="candidate",
        status=status,
        started_at=now,
        ended_at=None if status is Status.RUNNING else now,
        evidence={"lineage": {"clock_hash": "a" * 64}},
    )


@pytest.mark.parametrize("evaluation_id", tuple(EvaluationId))
@pytest.mark.parametrize("run_type", tuple(RunType))
def test_writer_creates_and_finalizes_one_immutable_dossier(
    tmp_path, evaluation_id: EvaluationId, run_type: RunType
) -> None:
    writer = StatisticsWriter(tmp_path)
    directory = writer.start(statistics(evaluation_id, Status.RUNNING, run_type))
    digest = writer.finalize(statistics(evaluation_id, Status.FINISHED, run_type))

    assert directory == tmp_path / "evaluations" / evaluation_id.value / "run-123"
    assert len(digest) == 64
    assert digest in (directory / "statistics.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        writer.finalize(statistics(evaluation_id, Status.FINISHED, run_type))


def test_statistics_rejects_nonfinite_or_forbidden_evidence() -> None:
    with pytest.raises(ValueError, match="finite"):
        RunStatistics(
            EvaluationId.MEDOID_MULTIVARIATE,
            "run",
            RunType.PARENT,
            "parent",
            Status.RUNNING,
            datetime(2026, 1, 1, tzinfo=UTC),
            evidence={"aggregate": {"score": float("nan")}},
        )
    with pytest.raises(ValueError, match="forbidden"):
        RunStatistics(
            EvaluationId.MEDOID_MULTIVARIATE,
            "run",
            RunType.PARENT,
            "parent",
            Status.RUNNING,
            datetime(2026, 1, 1, tzinfo=UTC),
            evidence={"identity": {"password": "forbidden"}},
        )
    with pytest.raises(ValueError, match="unknown"):
        RunStatistics(
            EvaluationId.MEDOID_MULTIVARIATE,
            "run",
            RunType.PARENT,
            "parent",
            Status.RUNNING,
            datetime(2026, 1, 1, tzinfo=UTC),
            evidence={"unknown": {}},
        )


def test_failed_run_retains_safe_final_dossier(tmp_path) -> None:
    writer = StatisticsWriter(tmp_path)
    writer.start(statistics(EvaluationId.MEDOID_MULTIVARIATE, Status.RUNNING))
    failed = RunStatistics(
        EvaluationId.MEDOID_MULTIVARIATE,
        "run-123",
        RunType.CANDIDATE,
        "candidate",
        Status.FAILED,
        datetime(2026, 1, 1, tzinfo=UTC),
        ended_at=datetime(2026, 1, 1, tzinfo=UTC),
        evidence={"failure": {"code": "FIT_FAILED", "reason": "convergence failure"}},
    )
    writer.finalize(failed)
    assert '"status":"FAILED"' in (
        tmp_path / "evaluations" / "medoid_multivariate" / "run-123" / "statistics.json"
    ).read_text(encoding="utf-8")


def test_statistics_contract_fails_closed_for_invalid_lifecycle_values() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    common = (
        EvaluationId.MEDOID_MULTIVARIATE,
        "run",
        RunType.PARENT,
        "parent",
        Status.RUNNING,
        now,
    )
    with pytest.raises(ValueError, match="schema version"):
        RunStatistics(*common, schema_version=2)
    with pytest.raises(ValueError, match="mlflow_run_id"):
        RunStatistics(EvaluationId.MEDOID_MULTIVARIATE, " ", *common[2:])
    with pytest.raises(ValueError, match="parent_run_id"):
        RunStatistics(*common, parent_run_id=" ")
    with pytest.raises(ValueError, match="timezone-aware"):
        RunStatistics(*common[:-1], datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="must not precede"):
        RunStatistics(*common, ended_at=datetime(2025, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="RUNNING"):
        RunStatistics(*common, ended_at=now)
    with pytest.raises(ValueError, match="final statistics"):
        RunStatistics(*common[:4], Status.FINISHED, now)
    with pytest.raises(ValueError, match="failure code"):
        RunStatistics(*common[:4], Status.FAILED, now, ended_at=now)
    with pytest.raises(ValueError, match="mapping keys"):
        RunStatistics(*common, evidence={"aggregate": {1: "invalid"}})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="JSON primitives"):
        RunStatistics(*common, evidence={"aggregate": {"bad": {1}}})


def test_writer_fails_closed_and_cleans_atomic_temporary_files(tmp_path, monkeypatch) -> None:
    writer = StatisticsWriter(tmp_path)
    running = statistics(EvaluationId.MEDOID_MULTIVARIATE, Status.RUNNING)
    finished = statistics(EvaluationId.MEDOID_MULTIVARIATE, Status.FINISHED)
    with pytest.raises(ValueError, match="initial statistics"):
        writer.start(finished)
    with pytest.raises(FileNotFoundError, match="initialized"):
        writer.finalize(finished)
    directory = writer.start(running)
    with pytest.raises(FileExistsError, match="immutable"):
        writer.start(running)
    with pytest.raises(ValueError, match="FINISHED or FAILED"):
        writer.finalize(running)

    target = directory / "atomic.json"
    monkeypatch.setattr(writer_module.os, "replace", lambda *_: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        StatisticsWriter._atomic_write(target, b"content")
    assert not tuple(directory.glob(".atomic.json.*"))


def test_preflight_reports_an_unwritable_root(tmp_path, monkeypatch) -> None:
    def fail_write(_: Path, __: bytes) -> int:
        raise OSError("read-only")

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    with pytest.raises(OSError, match="not writable"):
        StatisticsWriter(tmp_path).preflight()
