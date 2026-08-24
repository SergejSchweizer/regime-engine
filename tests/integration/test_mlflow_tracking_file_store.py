from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

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
