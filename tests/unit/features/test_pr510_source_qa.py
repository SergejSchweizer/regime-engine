from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from psycopg import IsolationLevel

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.features.ports import FeatureRequest
from market_regime_engine.features.postgres_source import MacroFeaturesPostgresSource

NOW = datetime(2026, 9, 1, tzinfo=UTC)


def _lineage(build: str = "build-1") -> tuple[Any, ...]:
    return (build, "a" * 64, 6, 5, 2, NOW, NOW.replace(day=2), NOW)


class CatalogCursor:
    description: tuple[Any, ...] | None = None

    def __init__(self, columns: list[tuple[Any, ...]], rows: list[tuple[Any, ...]]) -> None:
        self.columns = columns
        self.rows = rows
        self.executed: list[tuple[str, tuple[Any, ...] | None]] = []
        self.current: list[tuple[Any, ...]] = []

    def execute(self, query: Any, params: tuple[Any, ...] | None = None) -> None:
        text = str(query)
        self.executed.append((text, params))
        if "gold_sync_state" in text:
            self.current = [_lineage()]
        elif "pg_catalog.pg_class" in text:
            self.current = self.columns
        else:
            self.current = self.rows

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.current[0] if self.current else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.current

    def __enter__(self) -> CatalogCursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class ReadOnlyConnection:
    read_only = False
    isolation_level = IsolationLevel.READ_COMMITTED

    def __init__(self, cursor: CatalogCursor) -> None:
        self.cursor_value = cursor
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> CatalogCursor:
        return self.cursor_value

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def _source(
    columns: list[tuple[Any, ...]], rows: list[tuple[Any, ...]]
) -> tuple[MacroFeaturesPostgresSource, ReadOnlyConnection]:
    connection = ReadOnlyConnection(CatalogCursor(columns, rows))
    return MacroFeaturesPostgresSource(cast(Any, lambda: connection)), connection


def test_only_canonical_relation_is_queried_and_dynamic_columns_are_included() -> None:
    columns = [
        ("timestamp_m1", 1, "timestamp with time zone", "timestamptz"),
        ("f1", 2, "double precision", "float8"),
        ("new_loader_feature", 3, "double precision", "float8"),
    ]
    source, connection = _source(columns, [(NOW, 1.0, 2.0)])
    catalog, snapshot = source.read_with_catalog(FeatureRequest.all_features())

    assert catalog.feature_names == ("f1", "new_loader_feature")
    assert snapshot.feature_names == catalog.feature_names
    assert connection.read_only is True
    assert connection.isolation_level is IsolationLevel.REPEATABLE_READ
    assert connection.committed and connection.closed and not connection.rolled_back
    sql_text = "\n".join(query for query, _ in connection.cursor_value.executed)
    assert "macro_features_daily" not in sql_text
    assert "macro_raw" not in sql_text
    assert ("macro_loader", "macro_features") in [
        params for _, params in connection.cursor_value.executed if params is not None
    ]


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ([], "catalog is empty"),
        (
            [
                ("timestamp_m1", 1, "timestamp without time zone", "timestamp"),
                ("f1", 2, "double precision", "float8"),
            ],
            "timestamp-with-time-zone",
        ),
        (
            [
                ("timestamp_m1", 1, "timestamp with time zone", "timestamptz"),
                ("f1", 2, "numeric", "numeric"),
            ],
            "unsupported Gold feature type",
        ),
        (
            [
                ("timestamp_m1", 1, "timestamp with time zone", "timestamptz"),
                ("timestamp_m1", 2, "timestamp with time zone", "timestamptz"),
            ],
            "exactly one",
        ),
    ],
)
def test_catalog_schema_and_relation_failures_are_fail_closed(
    columns: list[tuple[Any, ...]], message: str
) -> None:
    source, _connection = _source(columns, [])
    with pytest.raises(ValueError, match=message):
        source.read_with_catalog(FeatureRequest.all_features())


def test_lineage_fingerprint_mutation_changes_catalog_identity() -> None:
    columns = [
        ("timestamp_m1", 1, "timestamp with time zone", "timestamptz"),
        ("f1", 2, "double precision", "float8"),
    ]
    first, _ = _source(columns, [(NOW, 1.0)])
    first_catalog, _ = first.read_with_catalog(FeatureRequest.all_features())
    changed_lineage = SourceLineage(
        "macro_loader.macro_features",
        "build-2",
        "b" * 64,
        6,
        5,
        "macro_loader.macro_features",
        NOW,
        row_count=2,
        min_timestamp=NOW,
        max_timestamp=NOW.replace(day=2),
    )
    assert changed_lineage.source_build_id != first_catalog.lineage.source_build_id
    assert changed_lineage.data_sha256 != first_catalog.lineage.data_sha256


def test_canonical_source_has_no_legacy_relation_or_wildcard_fallback() -> None:
    source_root = Path(__file__).parents[2] / "src" / "market_regime_engine"
    production = "\n".join(
        path.read_text(encoding="utf-8") for path in (source_root / "features").rglob("*.py")
    )
    assert "macro_loader.macro_features_daily" not in production
    assert "macro_loader.macro_raw" not in production
    assert "SELECT *" not in production
