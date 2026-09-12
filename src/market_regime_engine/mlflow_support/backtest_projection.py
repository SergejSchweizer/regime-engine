"""Standard MLflow Model Metrics projection for explicit backtest evidence."""

from __future__ import annotations

from market_regime_engine.evaluations.backtest_metrics import BacktestEvidence
from market_regime_engine.mlflow_support.ports import MetricPoint


def backtest_model_metric_points(evidence: BacktestEvidence) -> tuple[MetricPoint, ...]:
    """Return backtest points without model fitting, source access, or custom APIs."""

    if not isinstance(evidence, BacktestEvidence):
        raise TypeError("backtest projection requires BacktestEvidence")
    return evidence.metric_points


__all__ = ["backtest_model_metric_points"]
