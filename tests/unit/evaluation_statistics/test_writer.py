from __future__ import annotations

from datetime import UTC, datetime

import pytest

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
