from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from market_regime_engine.mlflow_support.ports import MetricPoint
from market_regime_engine.mlflow_support.tracking import FileMlflowTrackingPort


def _verifier() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "verify_mlflow_model_metrics.py"
    spec = importlib.util.spec_from_file_location("verify_mlflow_model_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Model Metrics verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _standard_tags() -> dict[str, str]:
    return {
        "regime_engine.metric_catalog_version": "1",
        "regime_engine.evaluation_plan_hash": "plan",
        "regime_engine.dataset_snapshot_key": "dataset",
        "regime_engine.evaluation_run_key": "run",
        "regime_engine.feature_order_sha256": "features",
        "regime_engine.feature_dimension": "2",
        "regime_engine.scope": "candidate",
    }


def _file_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    points: tuple[MetricPoint, ...],
    *,
    tags: dict[str, str] | None = None,
) -> tuple[str, str]:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = (tmp_path / "mlruns").as_uri()
    port = FileMlflowTrackingPort(tracking_uri, experiment_name="regime-engine-audit")
    run_id = port.start_run(run_name="audit")
    model_name = "evaluation-a-fold-1-candidate-a"
    model_id = port.create_logged_model(
        name=model_name,
        source_run_id=run_id,
        model_type="candidate-a",
        tags=_standard_tags() if tags is None else tags,
    )
    port.log_model_metric_points(model_id, points)
    return tracking_uri, model_name


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
        MetricPoint("oos_predictive_loglik_per_obs", -1.25, 1, 100),
        MetricPoint("oos_predictive_loglik_per_obs", -1.5, 2, 200),
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
    assert report["counts"]["metric_point_count"] == len(points)


@pytest.mark.parametrize(
    ("expected_point", "conflicting", "missing", "unexpected"),
    [
        (
            {"key": "valid_fold_count", "value": 2.0, "step": 0, "timestamp_ms": 100},
            1,
            0,
            0,
        ),
        (
            {"key": "valid_fold_count", "value": 1.0, "step": 0, "timestamp_ms": 101},
            1,
            0,
            0,
        ),
        (
            {"key": "valid_fold_count", "value": 1.0, "step": 1, "timestamp_ms": 100},
            0,
            1,
            1,
        ),
    ],
)
def test_model_metrics_verifier_rejects_non_exact_value_timestamp_or_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expected_point: dict[str, object],
    conflicting: int,
    missing: int,
    unexpected: int,
) -> None:
    tracking_uri, model_name = _file_model(
        tmp_path,
        monkeypatch,
        (MetricPoint("valid_fold_count", 1.0, 0, 100),),
    )

    report = _verifier().audit(
        tracking_uri,
        "regime-engine-audit",
        {"models": [{"name": model_name, "points": [expected_point]}]},
    )

    assert report["status"] == "failed"
    assert report["counts"]["conflicting_point_count"] == conflicting
    assert report["counts"]["missing_point_count"] == missing
    assert report["counts"]["unexpected_point_count"] == unexpected


def test_model_metrics_verifier_rejects_duplicate_file_backed_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = MetricPoint("valid_fold_count", 1.0, 0, 100)
    tracking_uri, model_name = _file_model(tmp_path, monkeypatch, (point, point))

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
                    ],
                }
            ]
        },
    )

    assert report["status"] == "failed"
    assert report["counts"]["duplicate_point_count"] == 1


def test_model_metrics_verifier_rejects_unknown_metric_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = MetricPoint("unclassified_metric", 1.0, 0, 100)
    tracking_uri, model_name = _file_model(tmp_path, monkeypatch, (point,))

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
                    ],
                }
            ]
        },
    )

    assert report["status"] == "failed"
    assert report["counts"]["unknown_metric_key_count"] == 1


def test_model_metrics_verifier_checks_domain_for_unexpected_metric_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = MetricPoint("outer_oos_predictive_loglik_per_obs", -1.25, 1, 100)
    tracking_uri, model_name = _file_model(
        tmp_path,
        monkeypatch,
        (point,),
    )

    report = _verifier().audit(
        tracking_uri,
        "regime-engine-audit",
        {"models": [{"name": model_name, "points": []}]},
    )

    assert report["status"] == "failed"
    assert report["counts"]["comparison_domain_violation_count"] == 1


def test_model_metrics_verifier_rejects_extra_catalogued_points(
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
    expected_point = MetricPoint("valid_fold_count", 1.0, 0, 100)
    port.log_model_metric_points(
        model_id,
        (
            expected_point,
            MetricPoint("oos_predictive_loglik_per_obs", -1.25, 1, 100),
        ),
    )

    report = _verifier().audit(
        tracking_uri,
        "regime-engine-audit",
        {
            "models": [
                {
                    "name": model_name,
                    "points": [
                        {
                            "key": expected_point.key,
                            "value": expected_point.value,
                            "step": expected_point.step,
                            "timestamp_ms": expected_point.timestamp_ms,
                        }
                    ],
                }
            ]
        },
    )

    assert report["status"] == "failed"
    assert report["counts"]["unexpected_point_count"] == 1
    assert report["counts"]["unexpected_metric_key_count"] == 1


def test_model_metrics_verifier_rejects_missing_fold_domain_tag(
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
    point = MetricPoint("outer_oos_predictive_loglik_per_obs", -1.25, 1, 100)
    port.log_model_metric_points(model_id, (point,))

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
                    ],
                }
            ]
        },
    )

    assert report["status"] == "failed"
    assert report["counts"]["comparison_domain_violation_count"] == 1


def test_model_metrics_verifier_accepts_compatible_comparison_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = (tmp_path / "mlruns").as_uri()
    port = FileMlflowTrackingPort(tracking_uri, experiment_name="regime-engine-audit")
    run_id = port.start_run(run_name="audit")
    tags = {
        "regime_engine.metric_catalog_version": "1",
        "regime_engine.evaluation_plan_hash": "plan",
        "regime_engine.dataset_snapshot_key": "dataset",
        "regime_engine.evaluation_run_key": "run",
        "regime_engine.feature_order_sha256": "features",
        "regime_engine.feature_dimension": "2",
        "regime_engine.scope": "candidate",
    }
    point = MetricPoint("oos_predictive_loglik_per_obs", -1.25, 1, 100)
    expected_models = []
    for model_name in ("evaluation-a-fold-1-candidate-a", "evaluation-a-fold-1-candidate-b"):
        model_id = port.create_logged_model(
            name=model_name,
            source_run_id=run_id,
            model_type="candidate",
            tags=tags,
        )
        port.log_model_metric_points(model_id, (point,))
        expected_models.append(
            {
                "name": model_name,
                "points": [
                    {
                        "key": point.key,
                        "value": point.value,
                        "step": point.step,
                        "timestamp_ms": point.timestamp_ms,
                    }
                ],
            }
        )

    report = _verifier().audit(
        tracking_uri,
        "regime-engine-audit",
        {
            "models": expected_models,
            "comparison_groups": [
                {
                    "metric_key": point.key,
                    "model_names": [model["name"] for model in expected_models],
                    "comparison_domain": "same_feature_vector_source_plan",
                }
            ],
        },
    )

    assert report["status"] == "verified"
    assert report["counts"]["comparison_domain_violation_count"] == 0


def test_model_metrics_verifier_rejects_duplicate_comparison_group_model_names(
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
    point = MetricPoint("oos_predictive_loglik_per_obs", -1.25, 1, 100)
    port.log_model_metric_points(model_id, (point,))

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
                    ],
                }
            ],
            "comparison_groups": [
                {
                    "metric_key": point.key,
                    "model_names": [model_name, model_name],
                    "comparison_domain": "same_feature_vector_source_plan",
                }
            ],
        },
    )

    assert report["status"] == "failed"
    assert report["counts"]["comparison_domain_violation_count"] == 1
