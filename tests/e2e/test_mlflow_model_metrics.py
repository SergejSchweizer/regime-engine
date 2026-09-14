from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from mlflow.tracking import MlflowClient

from market_regime_engine.mlflow_support.plots import render_model_metric_comparison
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
        require_nonempty_model_set=True,
    )
    assert report["status"] == "verified"
    assert report["counts"]["missing_point_count"] == 0
    assert report["counts"]["metric_point_count"] == len(points)
    assert report["model_count_by_scope"] == {"candidate": 1}
    assert report["models"][0]["metric_key_count"] == 3
    assert report["models"][0]["metric_point_count"] == len(points)
    assert report["namespace"]["run_count"] == 1
    assert report["namespace"]["historical_objects_zero"] is False


def test_model_metrics_verifier_requires_nonempty_completed_model_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = (tmp_path / "empty").as_uri()
    FileMlflowTrackingPort(tracking_uri, experiment_name="regime-engine-audit")

    report = _verifier().audit(
        tracking_uri,
        "regime-engine-audit",
        {"models": []},
        require_nonempty_model_set=True,
    )

    assert report["status"] == "failed"
    assert report["counts"]["nonempty_model_set_violation_count"] == 1


def test_model_metrics_verifier_reports_active_and_deleted_namespace_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = (tmp_path / "namespace").as_uri()
    port = FileMlflowTrackingPort(tracking_uri, experiment_name="regime-engine-audit")
    run_id = port.start_run(run_name="audit")
    port.end_run(run_id)
    deleted_run_id = port.start_run(run_name="deleted-audit")
    port.end_run(deleted_run_id)
    MlflowClient(tracking_uri=tracking_uri).delete_run(deleted_run_id)

    report = _verifier().audit(
        tracking_uri,
        "regime-engine-audit",
        {"models": []},
        require_clean_namespace=True,
    )

    assert report["status"] == "failed"
    assert report["namespace"]["run_count"] == 2
    assert report["namespace"]["run_status_counts"] == {"FINISHED": 2}
    assert report["namespace"]["run_lifecycle_counts"] == {"active": 1, "deleted": 1}
    assert report["counts"]["historical_namespace_violation_count"] == 1


def test_model_metrics_verifier_accepts_empty_clean_namespace_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = (tmp_path / "clean").as_uri()
    FileMlflowTrackingPort(tracking_uri, experiment_name="regime-engine-audit")

    report = _verifier().audit(
        tracking_uri,
        "regime-engine-audit",
        {"models": []},
        require_clean_namespace=True,
    )

    assert report["status"] == "verified"
    assert report["namespace"]["historical_objects_zero"] is True
    assert report["counts"]["historical_namespace_violation_count"] == 0


def test_model_metrics_verifier_fails_closed_when_deleted_model_inventory_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = (tmp_path / "unobservable-deleted-models").as_uri()
    FileMlflowTrackingPort(tracking_uri, experiment_name="regime-engine-audit")
    verifier = _verifier()
    monkeypatch.setattr(verifier, "_deleted_logged_model_ids", lambda _client: None)

    report = verifier.audit(
        tracking_uri,
        "regime-engine-audit",
        {"models": []},
        require_clean_namespace=True,
    )

    assert report["status"] == "failed"
    assert report["namespace"]["deleted_logged_model_inventory_available"] is False
    assert report["namespace"]["historical_objects_zero"] is False
    assert report["counts"]["deleted_logged_model_inventory_unavailable_count"] == 1
    assert report["counts"]["historical_namespace_violation_count"] == 1


def test_model_metrics_verifier_proves_resumed_history_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = (
        MetricPoint("valid_fold_count", 1.0, 0, 100),
        MetricPoint("oos_predictive_loglik_per_obs", -1.25, 1, 100),
    )
    baseline_uri, model_name = _file_model(
        tmp_path / "baseline",
        monkeypatch,
        points,
    )
    resumed_uri, _ = _file_model(
        tmp_path / "resumed",
        monkeypatch,
        points,
    )

    report = _verifier().audit(
        resumed_uri,
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
        baseline_tracking_uri=baseline_uri,
        baseline_experiment_name="regime-engine-audit",
    )

    assert report["status"] == "verified"
    parity = report["resume_parity"]
    assert parity["status"] == "verified"
    assert parity["missing_point_count"] == 0
    assert parity["unexpected_point_count"] == 0
    assert parity["conflicting_point_count"] == 0
    assert parity["lineage_tag_mismatch_count"] == 0


def test_model_metrics_verifier_rejects_resumed_history_difference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_point = MetricPoint("valid_fold_count", 1.0, 0, 100)
    resumed_point = MetricPoint("valid_fold_count", 2.0, 0, 100)
    baseline_uri, model_name = _file_model(
        tmp_path / "baseline",
        monkeypatch,
        (baseline_point,),
    )
    resumed_uri, _ = _file_model(
        tmp_path / "resumed",
        monkeypatch,
        (resumed_point,),
    )

    report = _verifier().audit(
        resumed_uri,
        "regime-engine-audit",
        {
            "models": [
                {
                    "name": model_name,
                    "points": [
                        {
                            "key": resumed_point.key,
                            "value": resumed_point.value,
                            "step": resumed_point.step,
                            "timestamp_ms": resumed_point.timestamp_ms,
                        }
                    ],
                }
            ]
        },
        baseline_tracking_uri=baseline_uri,
        baseline_experiment_name="regime-engine-audit",
    )

    assert report["status"] == "failed"
    parity = report["resume_parity"]
    assert parity["status"] == "failed"
    assert parity["missing_point_count"] == 1
    assert parity["unexpected_point_count"] == 1
    assert parity["conflicting_point_count"] == 1


def test_model_metrics_can_regenerate_comparison_plot_from_histories_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = MetricPoint("oos_predictive_loglik_per_obs", -1.25, 1, 100)
    tracking_uri, model_name = _file_model(tmp_path, monkeypatch, (point,))
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("regime-engine-audit")
    assert experiment is not None
    model = client.search_logged_models([experiment.experiment_id])[0]
    model_points = {
        model_name: tuple(
            MetricPoint(item.key, item.value, item.step, item.timestamp)
            for item in model.metrics or ()
        )
    }

    manifest = render_model_metric_comparison(
        point.key,
        model_points,
        {model_name: _standard_tags()},
        tmp_path / "plots",
    )

    assert manifest.plot_type == "model_metric_comparison"
    assert manifest.source_metric_keys == (point.key,)
    assert len(manifest.source_artifact_hash) == 64
    assert Path(manifest.png_path).is_file()


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


def test_model_metrics_verifier_rejects_stale_metric_catalog_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = (tmp_path / "stale-catalog").as_uri()
    port = FileMlflowTrackingPort(tracking_uri, experiment_name="regime-engine-audit")
    run_id = port.start_run(run_name="audit")
    model_name = "evaluation-a-fold-1-candidate-a"
    tags = {**_standard_tags(), "regime_engine.metric_catalog_version": "0"}
    model_id = port.create_logged_model(
        name=model_name,
        source_run_id=run_id,
        model_type="candidate-a",
        tags=tags,
    )
    point = MetricPoint("valid_fold_count", 1.0, 0, 100)
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
    assert report["counts"]["metric_catalog_version_violation_count"] == 1


def test_model_metrics_verifier_compares_float_values_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracking_uri, model_name = _file_model(
        tmp_path,
        monkeypatch,
        (MetricPoint("valid_fold_rate", -0.0, 0, 100),),
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
                            "key": "valid_fold_rate",
                            "value": 0.0,
                            "step": 0,
                            "timestamp_ms": 100,
                        }
                    ],
                }
            ]
        },
    )

    assert report["status"] == "failed"
    assert report["counts"]["conflicting_point_count"] == 1


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


def _strict_expectation(module: Any, model_name: str, point: MetricPoint) -> dict[str, Any]:
    tags = _standard_tags()
    return module.build_expectation_bundle(
        {
            "provenance": {
                "generator": "independent-test-evidence",
                "source_artifact": "independent-evidence.json",
                "source_artifact_sha256": "a" * 64,
                "evaluation_run_key": tags["regime_engine.evaluation_run_key"],
                "evaluation_plan_hash": tags["regime_engine.evaluation_plan_hash"],
                "dataset_snapshot_key": tags["regime_engine.dataset_snapshot_key"],
                "feature_order_sha256": tags["regime_engine.feature_order_sha256"],
                "feature_dimension": tags["regime_engine.feature_dimension"],
            },
            "models": [
                {
                    "name": model_name,
                    "required_tags": tags,
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
        }
    )


def test_strict_expectation_contract_is_content_addressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracking_uri, model_name = _file_model(
        tmp_path,
        monkeypatch,
        (MetricPoint("valid_fold_count", 1.0, 0, 100),),
    )
    module = _verifier()
    expectation = _strict_expectation(
        module,
        model_name,
        MetricPoint("valid_fold_count", 1.0, 0, 100),
    )

    report = module.audit(
        tracking_uri,
        "regime-engine-audit",
        expectation,
        require_expectation_contract=True,
    )

    assert report["expectation_contract_violations"] == []
    assert report["counts"]["expectation_contract_violation_count"] == 0
    assert len(expectation["provenance"]["evidence_sha256"]) == 64

    expectation["models"][0]["points"][0]["value"] = 2.0
    tampered = module.audit(
        tracking_uri,
        "regime-engine-audit",
        expectation,
        require_expectation_contract=True,
    )
    assert tampered["status"] == "failed"
    assert tampered["counts"]["expectation_contract_violation_count"] == 1


def test_strict_expectation_contract_rejects_stale_metric_catalog_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracking_uri, model_name = _file_model(
        tmp_path,
        monkeypatch,
        (MetricPoint("valid_fold_count", 1.0, 0, 100),),
    )
    module = _verifier()
    expectation = _strict_expectation(
        module,
        model_name,
        MetricPoint("valid_fold_count", 1.0, 0, 100),
    )
    expectation["models"][0]["required_tags"]["regime_engine.metric_catalog_version"] = "0"

    report = module.audit(
        tracking_uri,
        "regime-engine-audit",
        expectation,
        require_expectation_contract=True,
    )

    assert report["status"] == "failed"
    assert any(
        "unsupported metric catalog version" in violation
        for violation in report["expectation_contract_violations"]
    )


def test_strict_terminal_contract_rejects_running_or_pending_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracking_uri, model_name = _file_model(
        tmp_path,
        monkeypatch,
        (MetricPoint("valid_fold_count", 1.0, 0, 100),),
    )
    module = _verifier()
    report = module.audit(
        tracking_uri,
        "regime-engine-audit",
        _strict_expectation(
            module,
            model_name,
            MetricPoint("valid_fold_count", 1.0, 0, 100),
        ),
        require_expectation_contract=True,
        require_terminal_model_runs=True,
    )

    assert report["status"] == "failed"
    assert report["counts"]["nonready_model_count"] == 1
    assert report["counts"]["model_source_run_status_violation_count"] == 1


def test_comparison_group_missing_metric_is_a_domain_violation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = (tmp_path / "missing-group-metric").as_uri()
    port = FileMlflowTrackingPort(tracking_uri, experiment_name="regime-engine-audit")
    run_id = port.start_run(run_name="audit")
    tags = _standard_tags()
    point = MetricPoint("oos_predictive_loglik_per_obs", -1.25, 1, 100)
    names = ("evaluation-a-fold-1-candidate-a", "evaluation-a-fold-1-candidate-b")
    expected_models = []
    for index, model_name in enumerate(names):
        model_id = port.create_logged_model(
            name=model_name,
            source_run_id=run_id,
            model_type="candidate",
            tags=tags,
        )
        actual_point = point if index == 0 else MetricPoint("valid_fold_count", 1.0, 0, 100)
        port.log_model_metric_points(model_id, (actual_point,))
        expected_models.append(
            {
                "name": model_name,
                "points": [
                    {
                        "key": actual_point.key,
                        "value": actual_point.value,
                        "step": actual_point.step,
                        "timestamp_ms": actual_point.timestamp_ms,
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
                    "model_names": list(names),
                    "comparison_domain": "same_feature_vector_source_plan",
                }
            ],
        },
    )

    assert report["status"] == "failed"
    assert report["counts"]["comparison_domain_violation_count"] >= 1
