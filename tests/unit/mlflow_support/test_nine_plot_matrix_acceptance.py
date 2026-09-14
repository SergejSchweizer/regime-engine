from __future__ import annotations

from datetime import UTC, datetime

from market_regime_engine.mlflow_support.plot_data import (
    PLOT_FAMILY_SPECS,
    build_nine_plot_data,
)
from market_regime_engine.mlflow_support.ports import MetricPoint

_TAGS = {
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


def _all_nine_family_points() -> tuple[MetricPoint, ...]:
    keys = (
        "fit_quality_train_loglik_per_obs",
        "fit_quality_oos_predictive_loglik_per_obs",
        "fit_quality_aic",
        "fit_quality_bic",
        "fit_quality_hqc",
        "state_diag_posterior_probability_state_0",
        "state_diag_viterbi_state",
        "state_diag_emission_mean_state_0_feature_0",
        "state_diag_emission_variance_state_0_feature_0",
        "predictive_forecast_h1_residual",
        "predictive_h1_rmse",
        "classification_hard_nmi",
        "state_diag_expected_duration_state_0",
        "state_diag_transition_probability_state_0_to_state_0",
        "backtest_net_return",
        "backtest_equity",
        "backtest_drawdown",
    )
    timestamp_ms = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1000)
    return tuple(
        MetricPoint(key=key, value=float(index + 1), step=0, timestamp_ms=timestamp_ms)
        for index, key in enumerate(keys)
    )


def test_complete_nine_plot_matrix_is_available_and_order_independent() -> None:
    points = _all_nine_family_points()
    forward = build_nine_plot_data({"model-a": points}, {"model-a": _TAGS})
    reverse = build_nine_plot_data({"model-a": tuple(reversed(points))}, {"model-a": _TAGS})

    assert tuple(item.plot_id for item in forward) == tuple(PLOT_FAMILY_SPECS)
    assert {item.plot_id for item in forward if item.status == "available"} == set(
        PLOT_FAMILY_SPECS
    )
    assert all(item.unavailable_reason is None for item in forward)
    assert tuple(item.canonical_json for item in forward) == tuple(
        item.canonical_json for item in reverse
    )
    assert tuple(item.source_hash for item in forward) == tuple(
        item.source_hash for item in reverse
    )
