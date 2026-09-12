from __future__ import annotations

from datetime import UTC, datetime

from market_regime_engine.mlflow_support.plot_data import (
    build_nine_plot_data,
    build_plot_data,
)
from market_regime_engine.mlflow_support.ports import MetricPoint


def _tags() -> dict[str, str]:
    return {
        "regime_engine.dataset_snapshot_key": "dataset-a",
        "regime_engine.evaluation_plan_hash": "plan-a",
        "regime_engine.feature_order_sha256": "features-a",
        "regime_engine.feature_dimension": "2",
        "regime_engine.state_identity_scope": "model_version_local",
        "regime_engine.label_identity": "labels-v1",
        "regime_engine.forecast_contract_hash": "forecast-a",
        "regime_engine.backtest_contract_hash": "backtest-a",
        "regime_engine.backtest_data_snapshot_identity": "snapshot-a",
    }


def _point(key: str, value: float, step: int) -> MetricPoint:
    return MetricPoint(
        key=key,
        value=value,
        step=step,
        timestamp_ms=int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1000),
    )


def test_nine_plot_matrix_has_explicit_unavailable_statuses() -> None:
    points = {
        "model-a": (
            _point("fit_quality_train_loglik_per_obs", -1.0, 1),
            _point("fit_quality_oos_predictive_loglik_per_obs", -1.2, 1),
            _point("fit_quality_aic", 10.0, 1),
            _point("fit_quality_bic", 12.0, 1),
            _point("fit_quality_hqc", 11.0, 1),
        )
    }
    result = build_nine_plot_data(points, {"model-a": _tags()})
    assert len(result) == 9
    by_id = {item.plot_id: item for item in result}
    assert by_id["likelihood"].status == "available"
    assert by_id["information_criteria"].status == "available"
    assert by_id["backtest"].status == "not_available"
    assert by_id["backtest"].unavailable_reason is not None


def test_plot_data_rejects_unknown_metric_and_detects_cross_dimension() -> None:
    try:
        build_plot_data(
            "likelihood",
            {"model-a": (_point("not_catalogued", 1.0, 1),)},
            {"model-a": _tags()},
        )
    except KeyError as error:
        assert "not_catalogued" in str(error)
    else:
        raise AssertionError("unknown metrics must fail closed")

    points = {
        "model-a": (_point("fit_quality_train_loglik_per_obs", -1.0, 1),),
        "model-b": (_point("fit_quality_train_loglik_per_obs", -1.1, 1),),
    }
    tags = {"model-a": _tags(), "model-b": {**_tags(), "regime_engine.feature_order_sha256": "b"}}
    result = build_plot_data("likelihood", points, tags)
    assert result.status == "not_available"
    assert result.unavailable_reason is not None


def test_plot_data_hash_changes_when_logged_metric_changes() -> None:
    tags = {"model-a": _tags()}
    first = build_plot_data(
        "likelihood",
        {
            "model-a": (
                _point("fit_quality_train_loglik_per_obs", -1.0, 1),
                _point("fit_quality_oos_predictive_loglik_per_obs", -1.2, 1),
            )
        },
        tags,
    )
    second = build_plot_data(
        "likelihood",
        {
            "model-a": (
                _point("fit_quality_train_loglik_per_obs", -1.1, 1),
                _point("fit_quality_oos_predictive_loglik_per_obs", -1.2, 1),
            )
        },
        tags,
    )
    assert first.status == second.status == "available"
    assert first.source_hash != second.source_hash


def test_state_indexed_plot_is_not_cross_model_comparison() -> None:
    points = {
        model_id: (
            _point("state_diag_posterior_probability_state_0", 0.8, 1),
            _point("state_diag_viterbi_state", 0.0, 1),
        )
        for model_id in ("model-a", "model-b")
    }
    result = build_plot_data("state_posterior", points, {model_id: _tags() for model_id in points})
    assert result.status == "not_available"
    assert result.unavailable_reason is not None
