"""Standard MLflow Model Metrics projection for explicit forecast evidence."""

from __future__ import annotations

from market_regime_engine.evaluations.predictive_metrics import ForecastEvidence
from market_regime_engine.mlflow_support.ports import MetricPoint


def forecast_model_metric_points(evidence: ForecastEvidence) -> tuple[MetricPoint, ...]:
    """Return forecast points without fitting, source access, or custom MLflow APIs."""

    if not isinstance(evidence, ForecastEvidence):
        raise TypeError("forecast projection requires ForecastEvidence")
    return evidence.metric_points


__all__ = ["forecast_model_metric_points"]
