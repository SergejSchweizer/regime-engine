from __future__ import annotations

from pathlib import Path

import pytest

from market_regime_engine.mlflow_support.metric_export import (
    MetricExportLedger,
    metric_export_batch_key,
)
from market_regime_engine.mlflow_support.ports import MetricPoint


def _points() -> tuple[MetricPoint, ...]:
    return (
        MetricPoint("valid_fold_count", 2.0, 0, 100),
        MetricPoint("invalid_fold_count", 0.0, 0, 100),
    )


def test_batch_identity_is_deterministic_and_order_independent() -> None:
    points = _points()
    assert metric_export_batch_key("evaluation-a-model-a", points) == metric_export_batch_key(
        "evaluation-a-model-a", tuple(reversed(points))
    )
    assert metric_export_batch_key("evaluation-a-model-a", points) != metric_export_batch_key(
        "evaluation-a-model-b", points
    )


def test_ledger_persists_one_immutable_batch_identity(tmp_path: Path) -> None:
    ledger = MetricExportLedger(tmp_path)
    first = ledger.ensure_batch("evaluation-a-model-a", _points())
    second = MetricExportLedger(tmp_path).ensure_batch(
        "evaluation-a-model-a", tuple(reversed(_points()))
    )

    assert second == first
    with pytest.raises(ValueError, match="batch conflict"):
        ledger.ensure_batch(
            "evaluation-a-model-a",
            (MetricPoint("valid_fold_count", 3.0, 0, 100), _points()[1]),
        )
