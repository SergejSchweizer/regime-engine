from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

import market_regime_engine.mlflow_support.evaluation_tracking as module
from market_regime_engine.evaluation_statistics.contracts import RunStatistics, RunType, Status
from market_regime_engine.evaluation_statistics.writer import StatisticsWriter
from market_regime_engine.evaluations.contracts import DELTA1_FEATURES, EvaluationId
from market_regime_engine.mlflow_support.evaluation_tracking import track_statistics_run


class RecordingPort:
    def __init__(self) -> None:
        self.artifacts: list[tuple[str, str, str]] = []
        self.finished: list[str] = []
        self.failed: list[str] = []

    def start_run(self, *, run_name: str, parent_run_id: str | None = None) -> str:
        del run_name, parent_run_id
        return "run-1"

    def log_params(self, run_id: str, params: dict[str, str]) -> None:
        assert run_id == "run-1" and len(params["statistics_sha256"]) == 64

    def log_metric_points(self, run_id: str, points: tuple[object, ...]) -> None:
        del run_id, points

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str) -> None:
        self.artifacts.append((run_id, local_path, artifact_path))

    def end_run(self, run_id: str) -> None:
        self.finished.append(run_id)

    def fail_run(self, run_id: str) -> None:
        self.failed.append(run_id)


def _statistics() -> RunStatistics:
    return RunStatistics(
        EvaluationId.MEDOID_MULTIVARIATE,
        "placeholder",
        RunType.PARENT,
        "parent",
        Status.RUNNING,
        datetime(2026, 1, 1, tzinfo=UTC),
        evidence={"identity": {"source_build_id": "build"}},
    )


def test_statistics_run_logs_exact_finalized_json_and_sha(tmp_path: Path) -> None:
    port = RecordingPort()
    run_id, digest = track_statistics_run(
        port, StatisticsWriter(tmp_path), run_name="parent", statistics=_statistics()
    )
    path = tmp_path / "evaluations" / "medoid_multivariate" / run_id / "statistics.json"
    assert sha256(path.read_bytes()).hexdigest() == digest
    assert port.artifacts == [(run_id, str(path), "statistics")]
    assert port.finished == [run_id]


def test_statistics_failure_marks_mlflow_run_failed(tmp_path: Path) -> None:
    port = RecordingPort()
    with pytest.raises(FileExistsError, match="immutable"):
        track_statistics_run(
            port,
            StatisticsWriter(tmp_path),
            run_name="parent",
            statistics=_statistics(),
        )
        track_statistics_run(
            port,
            StatisticsWriter(tmp_path),
            run_name="parent",
            statistics=_statistics(),
        )
    assert port.failed == ["run-1"]


def test_statistics_run_payload_emitter_runs_before_finalization(tmp_path: Path) -> None:
    port = RecordingPort()
    seen: list[tuple[str, Path]] = []

    def emit(run_id: str, directory: Path) -> None:
        seen.append((run_id, directory))
        (directory / "payload.txt").write_text("payload", encoding="utf-8")

    run_id, _ = track_statistics_run(
        port,
        StatisticsWriter(tmp_path),
        run_name="parent",
        statistics=_statistics(),
        payload_emitter=emit,
    )
    directory = tmp_path / "evaluations" / "medoid_multivariate" / run_id
    assert seen == [(run_id, directory)]
    assert (directory / "payload.txt").read_text(encoding="utf-8") == "payload"
    assert port.finished == [run_id]
    assert port.failed == []


def test_payload_emitter_failure_finalizes_local_failed_and_fails_mlflow(tmp_path: Path) -> None:
    port = RecordingPort()

    def emit(_run_id: str, _directory: Path) -> None:
        raise RuntimeError("payload failure")

    with pytest.raises(RuntimeError, match="payload failure"):
        track_statistics_run(
            port,
            StatisticsWriter(tmp_path),
            run_name="parent",
            statistics=_statistics(),
            payload_emitter=emit,
        )
    path = tmp_path / "evaluations" / "medoid_multivariate" / "run-1" / "statistics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED"
    assert payload["evidence"]["failure"]["code"] == "RuntimeError"
    assert port.finished == []
    assert port.failed == ["run-1"]


def test_mlflow_logging_failure_never_ends_run_successfully(tmp_path: Path) -> None:
    class FailingPort(RecordingPort):
        def log_params(self, run_id: str, params: dict[str, str]) -> None:
            del run_id, params
            raise RuntimeError("metric logging failed")

    port = FailingPort()
    with pytest.raises(RuntimeError, match="metric logging failed"):
        track_statistics_run(
            port, StatisticsWriter(tmp_path), run_name="parent", statistics=_statistics()
        )
    assert port.finished == []
    assert port.failed == ["run-1"]


def test_local_finalization_failure_fails_mlflow_exactly_once(tmp_path: Path) -> None:
    class FailingWriter(StatisticsWriter):
        def finalize(self, statistics: RunStatistics) -> str:
            del statistics
            raise OSError("finalize failed")

    port = RecordingPort()
    with pytest.raises(OSError, match="finalize failed"):
        track_statistics_run(
            port, FailingWriter(tmp_path), run_name="parent", statistics=_statistics()
        )
    assert port.finished == []
    assert port.failed == ["run-1"]


def test_delta_model_metrics_payload_writes_manifest_artifacts_and_em_metric_points(
    tmp_path: Path,
) -> None:
    def fold(index: int, history: tuple[float, ...]) -> SimpleNamespace:
        return SimpleNamespace(
            fold_id=f"fold_{index:03d}",
            valid=True,
            train_model_observation_count=10,
            train_log_likelihood=-100.0,
            oos_predictive_log_likelihood_per_observation=-2.0,
            aic=40.0,
            bic=50.0,
            multistart_success_rate=0.875,
            multistart_result=SimpleNamespace(
                winner=SimpleNamespace(
                    seed=11,
                    iterations=len(history),
                    em_log_likelihood_history=history,
                )
            ),
        )

    first = SimpleNamespace(
        candidate_id="gaussian_hmm_k2_full",
        feature_order=("vix_delta_1obs",),
        folds=(fold(1, (-100.0, -90.0)),),
    )
    second = SimpleNamespace(
        candidate_id="gaussian_hmm_k3_full",
        feature_order=("vix_delta_1obs",),
        folds=(fold(1, (-120.0, -110.0)),),
    )
    grid = SimpleNamespace(feature_order=("vix_delta_1obs",), evaluations=(first, second))
    port = RecordingPort()

    module._emit_delta_model_metrics(port, "run-1", tmp_path, grid)

    manifest = json.loads((tmp_path / "model_metrics" / "manifest.json").read_text())
    assert manifest["candidate_ids"] == [first.candidate_id, second.candidate_id]
    assert len(manifest["candidates"]) == 2
    assert all(len(item["performance"]) == 5 for item in manifest["candidates"])
    assert any(path == "model_metrics" for _, _, path in port.artifacts)
    summary = module.summarize_em_convergence(first)
    points = module._em_metric_points(summary)
    assert [(point.step, point.value) for point in points] == [(1, -10.0), (2, -9.0)]


def test_statistics_run_rejects_non_running_dossier(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="RUNNING"):
        track_statistics_run(
            RecordingPort(),
            StatisticsWriter(tmp_path),
            run_name="parent",
            statistics=RunStatistics(
                EvaluationId.MEDOID_MULTIVARIATE,
                "run",
                RunType.PARENT,
                "parent",
                Status.FINISHED,
                datetime(2026, 1, 1, tzinfo=UTC),
                ended_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        )


class _Writer:
    def preflight(self) -> Path:
        return Path("evaluations")


def test_evaluation_tracking_creates_exact_v3_hierarchies(monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def track(_port, _writer, *, run_name, statistics, parent_run_id=None, payload_emitter=None):
        del _port, _writer, statistics, payload_emitter
        calls.append((run_name, parent_run_id))
        return f"run-{len(calls)}", "a" * 64

    monkeypatch.setattr(module, "track_statistics_run", track)
    monkeypatch.setattr(module, "_candidate_evidence", lambda *_, **__: {})
    monkeypatch.setattr(module, "asdict", lambda _: {})
    monkeypatch.setattr(
        module, "_track_grid", lambda *_: tuple((str(index), str(index)) for index in range(12))
    )
    candidate_grid = type(
        "Grid",
        (),
        {"evaluations": tuple(SimpleNamespace(candidate_id=str(index)) for index in range(12))},
    )()
    multi_type = type("Multi", (), {})
    medoid_type = type("Medoid", (), {})
    delta_type = type("Delta", (), {})
    monkeypatch.setattr(module, "MedoidMultivariateEvaluation", multi_type)
    monkeypatch.setattr(module, "MedoidUnivariateEvaluation", medoid_type)
    monkeypatch.setattr(module, "Delta1UnivariateEvaluation", delta_type)
    lineage = SimpleNamespace()
    spec = SimpleNamespace(feature_order=("a",))

    multi = multi_type()
    multi.lineage, multi.feature_spec, multi.candidate_grid = lineage, spec, candidate_grid
    multi.medoid_multivariate_statistical_champion, multi.no_champion_reason = "candidate", None
    assert (
        len(
            module.track_evaluation_result(
                RecordingPort(), _Writer(), result=multi
            ).candidate_run_ids
        )
        == 12
    )

    grids = tuple(
        SimpleNamespace(feature_name=f"f{index}", candidate_grid=candidate_grid)
        for index in range(8)
    )
    medoid = medoid_type()
    medoid.lineage, medoid.feature_spec, medoid.feature_grids = lineage, spec, grids
    medoid.medoid_univariate_evaluation_champion, medoid.no_champion_reason = "candidate", None
    assert (
        len(
            module.track_evaluation_result(
                RecordingPort(), _Writer(), result=medoid
            ).candidate_run_ids
        )
        == 96
    )

    delta = delta_type()
    delta.lineage = lineage
    delta.feature_grids = tuple(
        SimpleNamespace(feature_name=name, candidate_grid=candidate_grid)
        for name in DELTA1_FEATURES
    )
    delta.delta1_univariate_evaluation_champion, delta.no_champion_reason = "candidate", None
    result = module.track_evaluation_result(RecordingPort(), _Writer(), result=delta)
    assert len(result.feature_run_ids) == 13
    assert len(result.candidate_run_ids) == 156
