"""Focused, hermetic acceptance tests for resumable metric export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from market_regime_engine.mlflow_support.metric_export import (
    MetricExportLedger,
    export_model_metric_points,
)
from market_regime_engine.mlflow_support.ports import MetricPoint


def _points() -> tuple[MetricPoint, ...]:
    return (
        MetricPoint("valid_fold_count", 2.0, 0, 100),
        MetricPoint("invalid_fold_count", 0.0, 0, 100),
        MetricPoint("valid_fold_rate", 1.0, 0, 100),
    )


class _LocalMetricHistoryPort:
    """Small local fake that models acceptance before a client-side interruption."""

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self._history: dict[str, list[MetricPoint]] = {}
        self._fail_on_call = fail_on_call
        self._log_calls = 0

    def get_model_metric_points(self, model_id: str) -> tuple[MetricPoint, ...]:
        return tuple(self._history.get(model_id, ()))

    def log_model_metric_points(self, model_id: str, points: tuple[MetricPoint, ...]) -> None:
        self._log_calls += 1
        self._history.setdefault(model_id, []).extend(points)
        if self._log_calls == self._fail_on_call:
            raise RuntimeError("forced interruption after accepted metric batch")

    def history_bytes(self, model_id: str) -> bytes:
        history = sorted(
            self._history.get(model_id, ()),
            key=lambda point: (point.key, point.step, point.timestamp_ms, point.value.hex()),
        )
        payload = [
            {
                "key": point.key,
                "step": point.step,
                "timestamp_ms": point.timestamp_ms,
                "value_hex": point.value.hex(),
            }
            for point in history
        ]
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


@pytest.mark.parametrize("failure_after", (1, 2, 3))
def test_interrupted_export_retry_is_duplicate_free_and_byte_equivalent(
    tmp_path: Path,
    failure_after: int,
) -> None:
    points = _points()
    interrupted_port = _LocalMetricHistoryPort(fail_on_call=failure_after)
    interrupted_ledger = MetricExportLedger(tmp_path / "interrupted-ledger")

    with pytest.raises(RuntimeError, match="forced interruption"):
        export_model_metric_points(
            interrupted_port,
            "model-interrupted",
            logical_model_key="evaluation-a-model-a",
            points=points,
            ledger=interrupted_ledger,
        )

    # A retry sees batches accepted before the interruption and only exports the rest.
    interrupted_port._fail_on_call = None
    export_model_metric_points(
        interrupted_port,
        "model-interrupted",
        logical_model_key="evaluation-a-model-a",
        points=points,
        ledger=interrupted_ledger,
    )
    export_model_metric_points(
        interrupted_port,
        "model-interrupted",
        logical_model_key="evaluation-a-model-a",
        points=points,
        ledger=interrupted_ledger,
    )

    reference_port = _LocalMetricHistoryPort()
    export_model_metric_points(
        reference_port,
        "model-reference",
        logical_model_key="evaluation-a-model-a",
        points=points,
        ledger=MetricExportLedger(tmp_path / "reference-ledger"),
    )

    resumed_history = interrupted_port.get_model_metric_points("model-interrupted")
    identities = [(point.key, point.step) for point in resumed_history]
    assert len(resumed_history) == len(points)
    assert len(set(identities)) == len(points)
    assert interrupted_port.history_bytes("model-interrupted") == reference_port.history_bytes(
        "model-reference"
    )
    assert all(state.emitted for state in interrupted_ledger.states("evaluation-a-model-a"))


def test_changed_retry_is_rejected_without_mutating_accepted_history(tmp_path: Path) -> None:
    points = _points()
    port = _LocalMetricHistoryPort(fail_on_call=1)
    ledger = MetricExportLedger(tmp_path / "ledger")

    with pytest.raises(RuntimeError, match="forced interruption"):
        export_model_metric_points(
            port,
            "model-1",
            logical_model_key="evaluation-a-model-a",
            points=points,
            ledger=ledger,
        )

    before = port.history_bytes("model-1")
    changed_points = (
        MetricPoint("valid_fold_count", 3.0, 0, 100),
        *points[1:],
    )
    with pytest.raises(ValueError, match="batch conflict"):
        export_model_metric_points(
            port,
            "model-1",
            logical_model_key="evaluation-a-model-a",
            points=changed_points,
            ledger=ledger,
        )

    assert port.history_bytes("model-1") == before
