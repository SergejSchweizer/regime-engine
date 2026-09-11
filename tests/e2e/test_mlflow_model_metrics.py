from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from market_regime_engine.mlflow_support.ports import MetricPoint
from market_regime_engine.mlflow_support.tracking import FileMlflowTrackingPort


def _verifier():
    path = Path(__file__).parents[2] / "scripts" / "verify_mlflow_model_metrics.py"
    spec = importlib.util.spec_from_file_location("verify_mlflow_model_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Model Metrics verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_model_metrics_verifier_checks_exact_expected_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = (tmp_path / "mlruns").as_uri()
    port = FileMlflowTrackingPort(tracking_uri, experiment_name="regime-engine-audit")
    run_id = port.start_run(run_name="audit")
    model_name = "evaluation-a-fold-1-candidate-a"
    tags = {
        "regime_engine.metric_catalog_version": "1",
        "regime_engine.evaluation_plan_hash": "plan",
        "regime_engine.dataset_snapshot_key": "dataset",
        "regime_engine.evaluation_run_key": "run",
        "regime_engine.feature_order_sha256": "features",
        "regime_engine.feature_dimension": "2",
        "regime_engine.scope": "candidate",
    }
    model_id = port.create_logged_model(
        name=model_name,
        source_run_id=run_id,
        model_type="candidate-a",
        tags=tags,
    )
    points = (
        MetricPoint("valid_fold_count", 1.0, 0, 100),
        MetricPoint("invalid_fold_count", 0.0, 0, 100),
    )
    port.log_model_metric_points(model_id, points)

    report = _verifier().audit(
        tracking_uri,
        "regime-engine-audit",
        {
            "models": [
                {
                    "name": model_name,
                    "points": [
                        {
                            "key": point.key,
                            "value": point.value,
                            "step": point.step,
                            "timestamp_ms": point.timestamp_ms,
                        }
                        for point in points
                    ],
                }
            ]
        },
    )
    assert report["status"] == "verified"
    assert report["counts"]["missing_point_count"] == 0
