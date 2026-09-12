"""Pure, exhaustive conversion of flat evidence fields to metric points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real

from market_regime_engine.mlflow_support.metric_catalog import (
    metric_definition,
    require_metric_definition,
    validate_metric_points,
)
from market_regime_engine.mlflow_support.ports import MetricPoint


def _numeric_values(value: object, key: str) -> tuple[float, ...] | None:
    if isinstance(value, bool):
        return (float(value),)
    if isinstance(value, Real):
        return (float(value),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, Real):
                raise ValueError(f"metric evidence history {key!r} must contain only numbers")
            values.append(float(item))
        return tuple(values)
    return None


def extract_metric_points(
    evidence: Mapping[str, object],
    *,
    timestamp_ms: int,
) -> tuple[MetricPoint, ...]:
    """Extract every numeric flat evidence field without recomputation.

    Evidence keys are already canonical metric keys. Non-numeric metadata is
    ignored, while every numeric field must be registered in the central
    catalog. Histories retain their source order as consecutive metric steps;
    scalar values use step zero. No rounding or aggregation is performed.
    """

    if timestamp_ms < 0:
        raise ValueError("metric timestamps must be non-negative")
    points: list[MetricPoint] = []
    for key in sorted(evidence):
        if not isinstance(key, str) or not key or key.strip() != key:
            raise ValueError("metric evidence keys must be non-empty trimmed strings")
        values = _numeric_values(evidence[key], key)
        if values is None:
            continue
        definition = require_metric_definition(key)
        if definition.value_kind == "scalar" and len(values) > 1:
            raise ValueError(f"scalar metric {key!r} cannot contain a history")
        for step, value in enumerate(values):
            points.append(
                MetricPoint(
                    key=key,
                    value=value,
                    step=step,
                    timestamp_ms=timestamp_ms,
                )
            )
    result = tuple(points)
    validate_metric_points(result)
    return result


def require_catalogued_numeric_evidence(evidence: Mapping[str, object]) -> None:
    """Fail closed when a numeric evidence field has no catalog definition."""

    for key, value in evidence.items():
        if _numeric_values(value, str(key)) is not None and metric_definition(str(key)) is None:
            raise ValueError(f"numeric evidence field is not catalogued: {key!r}")


__all__ = ["extract_metric_points", "require_catalogued_numeric_evidence"]
