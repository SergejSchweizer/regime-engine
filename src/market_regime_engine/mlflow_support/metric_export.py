"""Crash-safe, idempotent export of metric points to MLflow LoggedModels."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from hashlib import sha256
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


@dataclass(frozen=True, slots=True)
class MetricExportBatchState:
    logical_model_key: str
    batch_key: str
    point_count: int


def _require_logical_model_key(logical_model_key: str) -> None:
    if not logical_model_key or logical_model_key.strip() != logical_model_key:
        raise ValueError("logical_model_key must be a non-empty trimmed string")


def _require_model_id(model_id: str) -> None:
    if not model_id or model_id.strip() != model_id:
        raise ValueError("model_id must be a non-empty trimmed string")


def _canonical_points(points: tuple[MetricPoint, ...]) -> tuple[MetricPoint, ...]:
    """Validate points and return their stable order for every retry."""

    validate_metric_points(points)
    return tuple(
        sorted(
            points,
            key=lambda item: (item.key, item.step, item.timestamp_ms, item.value.hex()),
        )
    )


def metric_export_batch_key(
    logical_model_key: str,
    points: tuple[MetricPoint, ...],
) -> str:
    """Return the order-independent identity of one complete metric batch."""

    _require_logical_model_key(logical_model_key)
    canonical_points = _canonical_points(points)
    payload = {
        "logical_model_key": logical_model_key,
        "points": [
            {
                "key": point.key,
                "step": point.step,
                "timestamp_ms": point.timestamp_ms,
                "value_hex": point.value.hex(),
            }
            for point in canonical_points
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


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
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metric_batches (
                    logical_model_key TEXT PRIMARY KEY,
                    batch_key TEXT NOT NULL,
                    point_count INTEGER NOT NULL CHECK (point_count >= 0)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metric_model_bindings (
                    logical_model_key TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL
                )
                """
            )

    def ensure_batch(
        self,
        logical_model_key: str,
        points: tuple[MetricPoint, ...],
        *,
        model_id: str | None = None,
    ) -> MetricExportBatchState:
        """Record one immutable batch identity or fail on a changed retry."""

        _require_logical_model_key(logical_model_key)
        if model_id is not None:
            _require_model_id(model_id)
        batch_key = metric_export_batch_key(logical_model_key, points)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT batch_key, point_count FROM metric_batches WHERE logical_model_key = ?",
                (logical_model_key,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO metric_batches(logical_model_key, batch_key, point_count) "
                    "VALUES (?, ?, ?)",
                    (logical_model_key, batch_key, len(points)),
                )
            elif str(row["batch_key"]) != batch_key or int(row["point_count"]) != len(points):
                raise ValueError(f"metric export batch conflict for {logical_model_key}")
            if model_id is not None:
                binding = connection.execute(
                    "SELECT model_id FROM metric_model_bindings WHERE logical_model_key = ?",
                    (logical_model_key,),
                ).fetchone()
                if binding is None:
                    connection.execute(
                        "INSERT INTO metric_model_bindings(logical_model_key, model_id) "
                        "VALUES (?, ?)",
                        (logical_model_key, model_id),
                    )
                elif str(binding["model_id"]) != model_id:
                    raise ValueError(
                        f"metric export model conflict for {logical_model_key}: "
                        f"{binding['model_id']} != {model_id}"
                    )
            connection.commit()
        return MetricExportBatchState(logical_model_key, batch_key, len(points))

    def ensure(self, logical_model_key: str, point: MetricPoint) -> MetricExportState:
        _require_logical_model_key(logical_model_key)
        _canonical_points((point,))
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
        _require_logical_model_key(logical_model_key)
        _canonical_points((point,))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT value, timestamp_ms FROM metric_points "
                "WHERE logical_model_key = ? AND metric_key = ? AND step = ?",
                (
                    logical_model_key,
                    point.key,
                    point.step,
                ),
            ).fetchone()
            if row is None:
                raise ValueError("cannot mark an unknown or conflicting metric point emitted")
            if (
                float(row["value"]).hex() != point.value.hex()
                or int(row["timestamp_ms"]) != point.timestamp_ms
            ):
                raise ValueError("cannot mark an unknown or conflicting metric point emitted")
            connection.execute(
                "UPDATE metric_points SET emitted = 1 WHERE logical_model_key = ? "
                "AND metric_key = ? AND step = ?",
                (logical_model_key, point.key, point.step),
            )
            connection.commit()

    def states(self, logical_model_key: str) -> tuple[MetricExportState, ...]:
        _require_logical_model_key(logical_model_key)
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

    canonical_points = _canonical_points(points)
    ledger.ensure_batch(logical_model_key, canonical_points, model_id=model_id)
    reader = getattr(port, "get_model_metric_points", None)
    if not callable(reader):
        raise TypeError("tracking port must expose get_model_metric_points for resumable export")
    remote_points = _canonical_points(tuple(reader(model_id)))
    remote_by_identity = {(point.key, point.step): point for point in remote_points}
    expected_identities = {(point.key, point.step) for point in canonical_points}
    unexpected_remote = set(remote_by_identity) - expected_identities
    if unexpected_remote:
        rendered = ", ".join(f"{key}@{step}" for key, step in sorted(unexpected_remote))
        raise ValueError(f"MLflow metric batch contains unexpected points: {rendered}")
    for point in canonical_points:
        state = ledger.ensure(logical_model_key, point)
        remote = remote_by_identity.get((point.key, point.step))
        if remote is not None and (
            remote.value.hex() != point.value.hex() or remote.timestamp_ms != point.timestamp_ms
        ):
            raise ValueError(
                f"MLflow metric conflict for {logical_model_key}:{point.key}@{point.step}"
            )
        if state.emitted:
            if remote is None:
                raise ValueError(
                    f"MLflow metric missing for emitted ledger point "
                    f"{logical_model_key}:{point.key}@{point.step}"
                )
            continue
        if remote is not None:
            ledger.mark_emitted(logical_model_key, point)
            continue
        port.log_model_metric_points(model_id, (point,))
        ledger.mark_emitted(logical_model_key, point)


__all__ = [
    "MetricExportBatchState",
    "MetricExportLedger",
    "MetricExportState",
    "export_model_metric_points",
    "metric_export_batch_key",
]
