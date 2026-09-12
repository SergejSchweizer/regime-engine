from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from market_regime_engine.mlflow_support.metric_export import (
    MetricExportLedger,
    export_model_metric_points,
)
from market_regime_engine.mlflow_support.ports import MetricPoint
from market_regime_engine.mlflow_support.tracking import FileMlflowTrackingPort

pytestmark = pytest.mark.integration


def test_file_mlflow_port_persists_parent_child_params_metrics_and_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = (tmp_path / "mlruns").as_uri()
    port = FileMlflowTrackingPort(tracking_uri, experiment_name="regime-engine-test")
    parent = port.start_run(run_name="evaluation-xetra-build-1")
    child = port.start_run(run_name="gaussian_hmm_k2_full", parent_run_id=parent)
    port.log_params(
        child,
        {"candidate_id": "gaussian_hmm_k2_full", "covariance_type": "full"},
    )
    port.log_metric_points(
        child,
        (
            MetricPoint(
                key="fold_oos_predictive_loglik_per_obs",
                value=-1.25,
                step=1,
                timestamp_ms=1_700_000_000_000,
            ),
        ),
    )
    artifact = tmp_path / "evidence.json"
    artifact.write_text('{"ok":true}\n', encoding="utf-8")
    port.log_artifact(child, str(artifact), "evaluation")

    client = MlflowClient(tracking_uri=tracking_uri)
    child_run = client.get_run(child)
    assert child_run.data.tags["mlflow.parentRunId"] == parent
    assert child_run.data.tags["mlflow.runName"] == "gaussian_hmm_k2_full"
    assert child_run.data.params["candidate_id"] == "gaussian_hmm_k2_full"
    assert child_run.data.params["covariance_type"] == "full"
    history = client.get_metric_history(child, "fold_oos_predictive_loglik_per_obs")
    assert len(history) == 1
    assert history[0].value == -1.25
    assert history[0].step == 1
    assert history[0].timestamp == 1_700_000_000_000
    artifacts = client.list_artifacts(child, "evaluation")
    assert [item.path for item in artifacts] == ["evaluation/evidence.json"]


def test_file_mlflow_port_reuses_one_logical_logged_model_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = (tmp_path / "mlruns").as_uri()
    port = FileMlflowTrackingPort(tracking_uri, experiment_name="regime-engine-test")
    first_run = port.start_run(run_name="first")
    second_run = port.start_run(run_name="retry")
    first = port.create_logged_model(
        name="evaluation-a-fold-1-candidate-a",
        source_run_id=first_run,
        model_type="candidate-a",
        tags={"regime_engine.evaluation_run_key": "evaluation-a"},
    )
    second = port.create_logged_model(
        name="evaluation-a-fold-1-candidate-a",
        source_run_id=second_run,
        model_type="candidate-a",
        tags={"regime_engine.evaluation_run_key": "evaluation-a"},
    )

    assert second == first


@pytest.mark.parametrize("failure_after", (1, 2, 3))
def test_metric_export_resume_has_no_duplicate_points_or_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_after: int,
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = (tmp_path / "mlruns").as_uri()
    port = FileMlflowTrackingPort(tracking_uri, experiment_name="regime-engine-test")
    run_id = port.start_run(run_name="metric-export")
    model_id = port.create_logged_model(
        name="evaluation-a-fold-1-candidate-a",
        source_run_id=run_id,
        model_type="candidate-a",
        tags={"regime_engine.evaluation_run_key": "evaluation-a"},
    )
    points = (
        MetricPoint("valid_fold_count", 1.0, 0, 100),
        MetricPoint("invalid_fold_count", 0.0, 0, 100),
        MetricPoint("valid_fold_rate", 1.0, 0, 100),
    )
    original = port.log_model_metric_points
    calls = 0

    def fail_after_first(point_model_id: str, point_batch: tuple[MetricPoint, ...]) -> None:
        nonlocal calls
        calls += 1
        original(point_model_id, point_batch)
        if calls == failure_after:
            raise RuntimeError("forced metric export interruption")

    monkeypatch.setattr(port, "log_model_metric_points", fail_after_first)
    ledger = MetricExportLedger(tmp_path / "ledger")
    with pytest.raises(RuntimeError, match="forced"):
        export_model_metric_points(
            port,
            model_id,
            logical_model_key="evaluation-a-fold-1-candidate-a",
            points=points,
            ledger=ledger,
        )

    monkeypatch.setattr(port, "log_model_metric_points", original)
    export_model_metric_points(
        port,
        model_id,
        logical_model_key="evaluation-a-fold-1-candidate-a",
        points=points,
        ledger=ledger,
    )
    export_model_metric_points(
        port,
        model_id,
        logical_model_key="evaluation-a-fold-1-candidate-a",
        points=points,
        ledger=ledger,
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    reference_run_id = port.start_run(run_name="metric-export-reference")
    reference_model_id = port.create_logged_model(
        name="evaluation-a-fold-1-candidate-reference",
        source_run_id=reference_run_id,
        model_type="candidate-a",
        tags={"regime_engine.evaluation_run_key": "evaluation-a"},
    )
    export_model_metric_points(
        port,
        reference_model_id,
        logical_model_key="evaluation-a-fold-1-candidate-reference",
        points=points,
        ledger=MetricExportLedger(tmp_path / "reference-ledger"),
    )

    def history(run: str) -> list[tuple[str, float, int, int]]:
        return [
            (item.key, item.value, item.step, item.timestamp)
            for key in sorted(point.key for point in points)
            for item in client.get_metric_history(run, key)
        ]

    assert history(run_id) == history(reference_run_id)
    assert all(item.emitted for item in ledger.states("evaluation-a-fold-1-candidate-a"))
    with pytest.raises(ValueError, match="conflict"):
        ledger.ensure(
            "evaluation-a-fold-1-candidate-a",
            MetricPoint("valid_fold_count", 2.0, 0, 100),
        )
