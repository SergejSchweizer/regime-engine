from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from market_regime_engine.evaluations.predictive_metrics import (
    ForecastContract,
    build_predictive_forecast_evidence,
)
from market_regime_engine.mlflow_support.predictive_projection import forecast_model_metric_points


def _inputs() -> tuple[tuple[datetime, ...], tuple[float, ...]]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return tuple(start + timedelta(days=index) for index in range(5)), (1.0, 2.0, 3.0, 4.0, 5.0)


def test_forecasts_use_origin_information_and_exact_reference_metrics() -> None:
    timestamps, targets = _inputs()
    evidence = build_predictive_forecast_evidence(
        timestamps=timestamps,
        filtered_probabilities=((1.0, 0.0),) * len(timestamps),
        transition_matrix=((1.0, 0.0), (0.0, 1.0)),
        state_target_means=(2.0, 10.0),
        target_values=targets,
        contract=ForecastContract("return", horizons=(1, 2)),
    )

    assert evidence.status == "available"
    assert [item.predicted for item in evidence.observations] == [2.0] * 7
    first = evidence.summaries[0]
    assert first.observation_count == 4
    assert first.missing_target_count == 0
    assert first.mae == pytest.approx(1.5)
    assert first.rmse == pytest.approx((14.0 / 4.0) ** 0.5)
    assert first.mape is not None
    assert first.r2 is not None
    points = forecast_model_metric_points(evidence)
    assert points
    assert all(point.key.startswith("predictive_") for point in points)


def test_future_target_mutation_does_not_change_earlier_prediction() -> None:
    timestamps, targets = _inputs()
    contract = ForecastContract("target", horizons=(1,))
    first = build_predictive_forecast_evidence(
        timestamps=timestamps,
        filtered_probabilities=((0.5, 0.5),) * len(timestamps),
        transition_matrix=((0.9, 0.1), (0.2, 0.8)),
        state_target_means=(1.0, 3.0),
        target_values=targets,
        contract=contract,
    )
    mutated = list(targets)
    mutated[-1] = 10_000.0
    second = build_predictive_forecast_evidence(
        timestamps=timestamps,
        filtered_probabilities=((0.5, 0.5),) * len(timestamps),
        transition_matrix=((0.9, 0.1), (0.2, 0.8)),
        state_target_means=(1.0, 3.0),
        target_values=mutated,
        contract=contract,
    )
    assert [item.predicted for item in first.observations[:-1]] == [
        item.predicted for item in second.observations[:-1]
    ]


def test_zero_targets_make_mape_explicitly_unavailable() -> None:
    timestamps, _targets = _inputs()
    evidence = build_predictive_forecast_evidence(
        timestamps=timestamps,
        filtered_probabilities=((1.0, 0.0),) * len(timestamps),
        transition_matrix=((1.0, 0.0), (0.0, 1.0)),
        state_target_means=(0.0, 1.0),
        target_values=(1.0, 0.0, 2.0, 3.0, 4.0),
        contract=ForecastContract("target"),
    )
    assert evidence.summaries[0].mape is None
    assert evidence.summaries[0].mape_unavailable_reason is not None


def test_missing_targets_produce_explicit_unavailable_evidence() -> None:
    timestamps, _targets = _inputs()
    evidence = build_predictive_forecast_evidence(
        timestamps=timestamps,
        filtered_probabilities=((1.0, 0.0),) * len(timestamps),
        transition_matrix=((1.0, 0.0), (0.0, 1.0)),
        state_target_means=(0.0, 1.0),
        target_values=(None, None, None, None, None),
        contract=ForecastContract("target"),
    )
    assert evidence.status == "not_available"
    assert evidence.metric_points == ()
    assert evidence.unavailable_reason is not None
