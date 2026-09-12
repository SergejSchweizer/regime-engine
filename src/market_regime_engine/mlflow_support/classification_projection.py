"""Standard Model Metrics projection for optional labeled-state evidence."""

from __future__ import annotations

from market_regime_engine.evaluations.state_classification_metrics import ClassificationEvidence
from market_regime_engine.mlflow_support.ports import MetricPoint


def classification_model_metric_points(evidence: ClassificationEvidence) -> tuple[MetricPoint, ...]:
    if not isinstance(evidence, ClassificationEvidence):
        raise TypeError("classification projection requires ClassificationEvidence")
    return evidence.metric_points


__all__ = ["classification_model_metric_points"]
