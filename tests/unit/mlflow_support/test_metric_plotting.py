from __future__ import annotations

from datetime import UTC, datetime

import pytest

from market_regime_engine.mlflow_support.plots import (
    render_model_metric_comparison,
    validate_model_metric_comparison,
)
from market_regime_engine.mlflow_support.ports import MetricPoint


def _tags(*, feature_hash: str = "features-a", fold_id: str | None = None) -> dict[str, str]:
    tags = {
        "regime_engine.dataset_snapshot_key": "dataset-a",
        "regime_engine.evaluation_plan_hash": "plan-a",
        "regime_engine.feature_order_sha256": feature_hash,
        "regime_engine.feature_dimension": "2",
    }
    if fold_id is not None:
        tags["regime_engine.outer_fold_id"] = fold_id
    return tags


def _point(key: str, value: float, step: int) -> MetricPoint:
    return MetricPoint(
        key=key,
        value=value,
        step=step,
        timestamp_ms=int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1000),
    )


def test_model_metric_plot_uses_only_compatible_logged_model_histories(tmp_path) -> None:
    points = {
        "model-a": (_point("oos_predictive_loglik_per_obs", -1.0, 0),),
        "model-b": (_point("oos_predictive_loglik_per_obs", -1.2, 0),),
    }
    tags = {model_id: _tags() for model_id in points}

    definition = validate_model_metric_comparison("oos_predictive_loglik_per_obs", points, tags)
    entry = render_model_metric_comparison("oos_predictive_loglik_per_obs", points, tags, tmp_path)

    assert definition.comparison_domain == "same_feature_vector_source_plan"
    assert (tmp_path / "model_metrics" / "oos_predictive_loglik_per_obs.png").is_file()
    assert entry.source_metric_keys == ("oos_predictive_loglik_per_obs",)


def test_soft_nmi_allows_different_feature_vectors_on_same_source_plan() -> None:
    points = {
        "model-a": (_point("soft_regime_nmi", 0.4, 1),),
        "model-b": (_point("soft_regime_nmi", 0.6, 1),),
    }
    tags = {"model-a": _tags(), "model-b": _tags(feature_hash="features-b")}

    assert (
        validate_model_metric_comparison("soft_regime_nmi", points, tags).comparison_domain
        == "same_source_plan"
    )


def test_cross_dimension_likelihood_and_duplicate_steps_fail_closed() -> None:
    points = {
        "model-a": (_point("oos_predictive_loglik_per_obs", -1.0, 0),),
        "model-b": (_point("oos_predictive_loglik_per_obs", -1.2, 0),),
    }
    with pytest.raises(ValueError, match="comparison domain"):
        validate_model_metric_comparison(
            "oos_predictive_loglik_per_obs",
            points,
            {"model-a": _tags(), "model-b": _tags(feature_hash="features-b")},
        )

    duplicate = {
        "model-a": (_point("soft_regime_nmi", 0.4, 1), _point("soft_regime_nmi", 0.5, 1)),
    }
    with pytest.raises(ValueError, match="duplicate"):
        validate_model_metric_comparison("soft_regime_nmi", duplicate, {"model-a": _tags()})


def test_fold_local_metric_requires_one_outer_fold() -> None:
    points = {
        "model-a": (_point("outer_oos_predictive_loglik_per_obs", -1.0, 1),),
        "model-b": (_point("outer_oos_predictive_loglik_per_obs", -1.2, 1),),
    }
    with pytest.raises(ValueError, match="comparison domain"):
        validate_model_metric_comparison(
            "outer_oos_predictive_loglik_per_obs",
            points,
            {
                "model-a": _tags(fold_id="outer_fold_001"),
                "model-b": _tags(fold_id="outer_fold_002"),
            },
        )
