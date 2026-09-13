from __future__ import annotations

from pathlib import Path

import pytest

from market_regime_engine.mlflow_support.metric_export import (
    MetricExportLedger,
    export_model_metric_points,
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


def test_ledger_binds_logical_batch_to_one_logged_model(tmp_path: Path) -> None:
    ledger = MetricExportLedger(tmp_path)
    ledger.ensure_batch("evaluation-a-model-a", _points(), model_id="model-1")

    with pytest.raises(ValueError, match="model conflict"):
        ledger.ensure_batch("evaluation-a-model-a", _points(), model_id="model-2")


class _MemoryTrackingPort:
    def __init__(self, remote: tuple[MetricPoint, ...] = ()) -> None:
        self.remote = remote
        self.logged: list[MetricPoint] = []

    def get_model_metric_points(self, model_id: str) -> tuple[MetricPoint, ...]:
        return self.remote + tuple(self.logged)

    def log_model_metric_points(self, model_id: str, points: tuple[MetricPoint, ...]) -> None:
        self.logged.extend(points)


def test_export_rejects_unexpected_remote_points(tmp_path: Path) -> None:
    unexpected = MetricPoint("valid_fold_rate", 1.0, 0, 100)
    port = _MemoryTrackingPort((unexpected,))

    with pytest.raises(ValueError, match="unexpected points"):
        export_model_metric_points(
            port,
            "model-1",
            logical_model_key="evaluation-a-model-a",
            points=_points(),
            ledger=MetricExportLedger(tmp_path),
        )


def test_export_rejects_missing_remote_point_for_emitted_ledger(tmp_path: Path) -> None:
    port = _MemoryTrackingPort()
    ledger = MetricExportLedger(tmp_path)
    export_model_metric_points(
        port,
        "model-1",
        logical_model_key="evaluation-a-model-a",
        points=_points(),
        ledger=ledger,
    )
    port.logged.clear()

    with pytest.raises(ValueError, match="missing for emitted"):
        export_model_metric_points(
            port,
            "model-1",
            logical_model_key="evaluation-a-model-a",
            points=_points(),
            ledger=ledger,
        )
