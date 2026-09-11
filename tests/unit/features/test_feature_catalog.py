from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from psycopg import IsolationLevel

from market_regime_engine.features import FeatureRequest, SourceMode
from market_regime_engine.features.postgres_source import PostgresFeatureSource

NOW = datetime(2026, 9, 9, tzinfo=UTC)


class DynamicCursor:
    def __init__(self, catalog: list[tuple[Any, ...]], rows: list[tuple[Any, ...]]) -> None:
        self.catalog = catalog
        self.rows = rows
        self.executed: list[tuple[Any, Any]] = []
        self.fetchall_count = 0

    def execute(self, query: Any, params: Any = None) -> None:
        self.executed.append((query, params))

    def fetchone(self) -> tuple[Any, ...]:
        return ("build-1", "a" * 64, 4, 3, 2, NOW, NOW.replace(day=10), NOW)

    def fetchall(self) -> list[tuple[Any, ...]]:
        self.fetchall_count += 1
        return self.catalog if self.fetchall_count == 1 else self.rows

    def __enter__(self) -> DynamicCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class DynamicConnection:
    def __init__(self, cursor: DynamicCursor) -> None:
        self.cursor_value = cursor
        self.read_only = False
        self.isolation_level: object | None = None
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> DynamicCursor:
        return self.cursor_value

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def catalog() -> list[tuple[Any, ...]]:
    return [
        ("timestamp_m1", 1, "timestamp with time zone", "timestamptz"),
        ("feature_a", 2, "double precision", "float8"),
        ("feature_b", 3, "double precision", "float8"),
    ]


def test_dynamic_read_uses_one_snapshot_and_returns_catalog_lineage() -> None:
    cursor = DynamicCursor(catalog(), [(NOW, 1.0, 2.0), (NOW.replace(day=10), 2.0, 3.0)])
    connection = DynamicConnection(cursor)
    source = PostgresFeatureSource(lambda: connection)
    feature_catalog, snapshot = source.read_with_catalog(
        FeatureRequest(("feature_b", "feature_a"), None, None, SourceMode.SCHEMA_DISCOVERY)
    )

    assert feature_catalog.feature_names == ("feature_a", "feature_b")
    assert feature_catalog.lineage.source_build_id == "build-1"
    assert snapshot.feature_names == ("feature_b", "feature_a")
    assert snapshot.rows[0].values == (1.0, 2.0)
    assert len(feature_catalog.catalog_hash) == 64
    assert cursor.fetchall_count == 2
    assert connection.read_only is True
    assert connection.isolation_level is IsolationLevel.REPEATABLE_READ
    assert connection.committed and connection.closed and not connection.rolled_back
    assert "information_schema.columns" in str(cursor.executed[1][0])


def test_dynamic_read_validates_requested_names_against_same_catalog_snapshot() -> None:
    cursor = DynamicCursor(catalog(), [])
    connection = DynamicConnection(cursor)
    source = PostgresFeatureSource(lambda: connection)
    with pytest.raises(ValueError, match="unregistered"):
        source.read(FeatureRequest(("feature_missing",), None, None, SourceMode.SCHEMA_DISCOVERY))
    assert connection.rolled_back and connection.closed


@pytest.mark.parametrize(
    ("bad_catalog", "message"),
    [
        (
            [
                ("timestamp_m1", 1, "timestamp without time zone", "timestamp"),
                ("feature_a", 2, "double precision", "float8"),
            ],
            "timestamp-with-time-zone",
        ),
        (
            [
                ("timestamp_m1", 1, "timestamp with time zone", "timestamptz"),
                ("feature_a", 2, "numeric", "numeric"),
            ],
            "unsupported Gold feature type",
        ),
        (
            [
                ("timestamp_m1", 1, "timestamp with time zone", "timestamptz"),
                ("Feature-A", 2, "double precision", "float8"),
            ],
            "unsafe feature identifier",
        ),
        (
            [
                ("timestamp_m1", 1, "timestamp with time zone", "timestamptz"),
                ("feature_a", 2, "double precision", "float8"),
                ("feature_a", 3, "double precision", "float8"),
            ],
            "unique",
        ),
    ],
)
def test_dynamic_catalog_schema_failures_roll_back(
    bad_catalog: list[tuple[Any, ...]], message: str
) -> None:
    cursor = DynamicCursor(bad_catalog, [])
    connection = DynamicConnection(cursor)
    source = PostgresFeatureSource(lambda: connection)
    with pytest.raises(ValueError, match=message):
        source.read(FeatureRequest(("feature_a",), None, None, SourceMode.SCHEMA_DISCOVERY))
    assert connection.rolled_back and connection.closed


def test_resolved_feature_mode_remains_explicit_and_catalog_method_is_rejected() -> None:
    source = PostgresFeatureSource(
        lambda: pytest.fail("connection must not be used"), ("feature_a",)
    )
    with pytest.raises(ValueError, match="dynamic catalog mode"):
        source.read_with_catalog(
            FeatureRequest(("feature_a",), None, None, SourceMode.SCHEMA_DISCOVERY)
        )
