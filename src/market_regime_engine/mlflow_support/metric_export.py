"""Crash-safe, idempotent export of metric points to MLflow LoggedModels."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from market_regime_engine.mlflow_support.metric_catalog import validate_metric_points
from market_regime_engine.mlflow_support.ports import MetricPoint, TrackingPort


@dataclass(frozen=True, slots=True)
class MetricExportState:
    logical_model_key: str
    key: str
    step: int
    value: float
    timestamp_ms: int
    emitted: bool


class MetricExportLedger:
    """Durable point ledger used to resume interrupted MLflow exports."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(mode=0o750, parents=True, exist_ok=True)
        self._database = self._root / "mlflow-metric-export.sqlite3"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metric_points (
                    logical_model_key TEXT NOT NULL,
                    metric_key TEXT NOT NULL,
                    step INTEGER NOT NULL CHECK (step >= 0),
                    value REAL NOT NULL,
                    timestamp_ms INTEGER NOT NULL CHECK (timestamp_ms >= 0),
                    emitted INTEGER NOT NULL CHECK (emitted IN (0, 1)),
                    PRIMARY KEY (logical_model_key, metric_key, step)
                )
                """
            )

    def ensure(self, logical_model_key: str, point: MetricPoint) -> MetricExportState:
        if not logical_model_key or logical_model_key.strip() != logical_model_key:
            raise ValueError("logical_model_key must be a non-empty trimmed string")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM metric_points WHERE logical_model_key = ? AND metric_key = ? "
                "AND step = ?",
                (logical_model_key, point.key, point.step),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO metric_points(logical_model_key, metric_key, step, value, "
                    "timestamp_ms, emitted) VALUES (?, ?, ?, ?, ?, 0)",
                    (
                        logical_model_key,
                        point.key,
                        point.step,
                        point.value,
                        point.timestamp_ms,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM metric_points WHERE logical_model_key = ? AND metric_key = ? "
                    "AND step = ?",
                    (logical_model_key, point.key, point.step),
                ).fetchone()
            assert row is not None
            if float(row["value"]).hex() != point.value.hex() or int(row["timestamp_ms"]) != (
                point.timestamp_ms
            ):
                raise ValueError(
                    f"metric export conflict for {logical_model_key}:{point.key}@{point.step}"
                )
            connection.commit()
        return MetricExportState(
            logical_model_key=logical_model_key,
            key=point.key,
            step=point.step,
            value=point.value,
            timestamp_ms=point.timestamp_ms,
            emitted=bool(row["emitted"]),
        )

    def mark_emitted(self, logical_model_key: str, point: MetricPoint) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                "UPDATE metric_points SET emitted = 1 WHERE logical_model_key = ? "
                "AND metric_key = ? AND step = ? AND value = ? AND timestamp_ms = ?",
                (
                    logical_model_key,
                    point.key,
                    point.step,
                    point.value,
                    point.timestamp_ms,
                ),
            ).rowcount
            if updated != 1:
                raise ValueError("cannot mark an unknown or conflicting metric point emitted")
            connection.commit()

    def states(self, logical_model_key: str) -> tuple[MetricExportState, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM metric_points WHERE logical_model_key = ? ORDER BY metric_key, step",
                (logical_model_key,),
            ).fetchall()
        return tuple(
            MetricExportState(
                logical_model_key=logical_model_key,
                key=str(row["metric_key"]),
                step=int(row["step"]),
                value=float(row["value"]),
                timestamp_ms=int(row["timestamp_ms"]),
                emitted=bool(row["emitted"]),
            )
            for row in rows
        )


def export_model_metric_points(
    port: TrackingPort,
    model_id: str,
    *,
    logical_model_key: str,
    points: tuple[MetricPoint, ...],
    ledger: MetricExportLedger,
) -> None:
    """Reconcile and export points one at a time, preserving deterministic order."""

    validate_metric_points(points)
    reader = getattr(port, "get_model_metric_points", None)
    if not callable(reader):
        raise TypeError("tracking port must expose get_model_metric_points for resumable export")
    remote_points = tuple(reader(model_id))
    for point in points:
        state = ledger.ensure(logical_model_key, point)
        if state.emitted:
            continue
        matching = tuple(
            item for item in remote_points if item.key == point.key and item.step == point.step
        )
        if matching:
            if any(
                item.value.hex() == point.value.hex() and item.timestamp_ms == point.timestamp_ms
                for item in matching
            ):
                ledger.mark_emitted(logical_model_key, point)
                continue
            raise ValueError(
                f"MLflow metric conflict for {logical_model_key}:{point.key}@{point.step}"
            )
        port.log_model_metric_points(model_id, (point,))
        ledger.mark_emitted(logical_model_key, point)


__all__ = ["MetricExportLedger", "MetricExportState", "export_model_metric_points"]
