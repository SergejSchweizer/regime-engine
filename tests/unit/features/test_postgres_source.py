from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from psycopg import IsolationLevel

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.features import FeatureRequest, PostgresFeatureSource, SourceMode

NOW = datetime(2026, 8, 24, tzinfo=UTC)


class FakeCursor:
    description = None

    def __init__(self, lineage: tuple[Any, ...], rows: list[tuple[Any, ...]]) -> None:
        self.lineage = lineage
        self.rows = rows
        self.executed: list[tuple[Any, Any]] = []
        self._fetch_stage = 0

    def execute(self, query: Any, params: Any = None) -> None:
        self.executed.append((query, params))

    def fetchone(self) -> tuple[Any, ...] | None:
        self._fetch_stage += 1
        return self.lineage

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class DynamicCatalogCursor(FakeCursor):
    def __init__(
        self,
        lineage: tuple[Any, ...],
        columns: list[tuple[Any, ...]],
        rows: list[tuple[Any, ...]],
    ) -> None:
        super().__init__(lineage, rows)
        self.columns = columns
        self._catalog_fetched = False

    def fetchall(self) -> list[tuple[Any, ...]]:
        if not self._catalog_fetched:
            self._catalog_fetched = True
            return self.columns
        return self.rows


class OneRowCursor:
    description = None

    def __init__(self, row: tuple[Any, ...] | None) -> None:
        self.row = row

    def execute(self, query: Any, params: Any = None) -> None:
        del query, params

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.row


class ColumnsCursor:
    description = None

    def __init__(self, columns: list[tuple[Any, ...]]) -> None:
        self.columns = columns

    def execute(self, query: Any, params: Any = None) -> None:
        del query, params

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.columns


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.read_only = False
        self.isolation_level: object | None = None
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def lineage_row() -> tuple[Any, ...]:
    return (
        "build-7",
        "a" * 64,
        2,
        1,
        2,
        NOW,
        NOW.replace(day=25),
        NOW.replace(hour=1),
    )


def test_selection_mode_preserves_nulls_and_snapshot_transaction() -> None:
    cursor = FakeCursor(
        lineage_row(),
        [(NOW, 1.0, None), (NOW.replace(day=25), 2.0, 3.0)],
    )
    connection = FakeConnection(cursor)
    source = PostgresFeatureSource(lambda: connection, ("f1", "f2"))
    snapshot = source.read(FeatureRequest(("f2", "f1"), None, None, SourceMode.FEATURE_SELECTION))
    assert snapshot.feature_names == ("f2", "f1")
    assert snapshot.rows[0].values == (1.0, None)
    assert snapshot.skipped_incomplete_row_count == 0
    assert connection.committed and connection.closed and not connection.rolled_back
    assert connection.read_only is True
    assert connection.isolation_level is IsolationLevel.REPEATABLE_READ
    first_query = str(cursor.executed[0][0])
    assert "source_build_id" in first_query


def test_resolved_model_mode_excludes_incomplete_rows_without_fill() -> None:
    cursor = FakeCursor(
        lineage_row(),
        [(NOW, 1.0, None), (NOW.replace(day=25), 2.0, 3.0)],
    )
    connection = FakeConnection(cursor)
    snapshot = PostgresFeatureSource(lambda: connection, ("f1", "f2")).read(
        FeatureRequest(("f1", "f2"), NOW, NOW.replace(day=25), SourceMode.RESOLVED_MODEL)
    )
    assert len(snapshot.rows) == 1
    assert snapshot.rows[0].timestamp == NOW.replace(day=25)
    assert snapshot.skipped_incomplete_row_count == 1


def test_unregistered_identifier_is_rejected_before_connection() -> None:
    called = False

    def connect() -> FakeConnection:
        nonlocal called
        called = True
        raise AssertionError("must not connect")

    source = PostgresFeatureSource(connect, ("f1",))
    with pytest.raises(ValueError, match="unregistered"):
        source.read(FeatureRequest(("f1;DROP",), None, None, SourceMode.FEATURE_SELECTION))
    assert called is False


def test_incompatible_source_schema_version_fails_closed() -> None:
    values = list(lineage_row())
    values[2] = 1
    connection = FakeConnection(FakeCursor(tuple(values), []))
    source = PostgresFeatureSource(lambda: connection, ("f1",))
    with pytest.raises(ValueError, match="schema_version"):
        source.read(FeatureRequest(("f1",), None, None, SourceMode.FEATURE_SELECTION))
    assert connection.rolled_back and connection.closed


def test_nonfinite_and_non_monotonic_rows_fail_closed_and_rollback() -> None:
    for rows, match in (
        ([(NOW, float("inf"))], "finite"),
        ([(NOW, 1.0), (NOW, 2.0)], "strictly increasing"),
    ):
        cursor = FakeCursor(lineage_row(), rows)
        connection = FakeConnection(cursor)
        source = PostgresFeatureSource(lambda connection=connection: connection, ("f1",))
        with pytest.raises(ValueError, match=match):
            source.read(FeatureRequest(("f1",), None, None, SourceMode.FEATURE_SELECTION))
        assert connection.rolled_back and connection.closed


def test_dynamic_catalog_full_and_subset_requests_use_the_expected_columns() -> None:
    columns = [
        ("timestamp_m1", 1, "timestamp with time zone", "timestamptz"),
        ("f1", 2, "double precision", "float8"),
        ("f2", 3, "double precision", "float8"),
    ]
    full_cursor = DynamicCatalogCursor(
        lineage_row(),
        columns,
        [(NOW, 1.0, 2.0), (NOW.replace(day=25), 3.0, 4.0)],
    )
    full_connection = FakeConnection(full_cursor)
    full_catalog, full_snapshot = PostgresFeatureSource(lambda: full_connection).read_with_catalog(
        FeatureRequest.all_features()
    )
    assert full_catalog.feature_names == ("f1", "f2")
    assert full_snapshot.materialized_feature_data_sha256 is not None
    assert (
        full_catalog.materialized_feature_data_sha256
        == full_snapshot.materialized_feature_data_sha256
    )

    subset_cursor = DynamicCatalogCursor(lineage_row(), columns, [(NOW, 2.0)])
    subset_connection = FakeConnection(subset_cursor)
    subset_catalog, subset_snapshot = PostgresFeatureSource(
        lambda: subset_connection
    ).read_with_catalog(FeatureRequest(("f2",), None, None, SourceMode.FEATURE_SELECTION))
    assert subset_catalog.feature_names == ("f1", "f2")
    assert subset_snapshot.feature_names == ("f2",)
    assert subset_catalog.materialized_feature_data_sha256 is None


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (None, "missing sync-state"),
        (("build",) * 7, "sync-state shape"),
        ((*lineage_row()[:3], 2, *lineage_row()[4:]), "feature_version"),
        ((*lineage_row()[:4], -1, *lineage_row()[5:]), "row_count"),
        (
            (
                lineage_row()[0],
                lineage_row()[1],
                lineage_row()[2],
                lineage_row()[3],
                lineage_row()[4],
                NOW.replace(day=25),
                NOW,
                lineage_row()[7],
            ),
            "bounds are inverted",
        ),
        (
            (
                lineage_row()[0],
                lineage_row()[1],
                lineage_row()[2],
                lineage_row()[3],
                lineage_row()[4],
                datetime(2026, 8, 24),
                lineage_row()[6],
                lineage_row()[7],
            ),
            "min_timestamp must be timezone-aware",
        ),
    ],
)
def test_lineage_contract_failures_are_explicit(row: tuple[Any, ...] | None, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PostgresFeatureSource(lambda: None)._read_lineage(OneRowCursor(row))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ([], "catalog is empty"),
        ([("f1", 1, "double precision", "float8")], "exactly one"),
        (
            [
                ("timestamp_m1", 1, "timestamp with time zone", "timestamptz"),
                ("timestamp_m1", 2, "timestamp with time zone", "timestamptz"),
            ],
            "exactly one",
        ),
        (
            [("timestamp_m1", 1, "timestamp without time zone", "timestamp")],
            "timestamp-with-time-zone",
        ),
        (
            [
                ("timestamp_m1", 1, "timestamp with time zone", "timestamptz"),
                ("f1", 2, "double precision"),
            ],
            "column shape",
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
                ("Bad-Feature", 2, "double precision", "float8"),
            ],
            "unsafe feature identifier",
        ),
        (
            [
                ("timestamp_m1", 1, "timestamp with time zone", "timestamptz"),
                ("f1", "two", "double precision", "float8"),
            ],
            "catalog ordinal",
        ),
        (
            [("timestamp_m1", 1, "timestamp with time zone", "timestamptz")],
            "no dynamic Gold feature",
        ),
    ],
)
def test_legacy_catalog_contract_failures_are_explicit(
    columns: list[tuple[Any, ...]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PostgresFeatureSource._read_catalog(  # type: ignore[arg-type]
            ColumnsCursor(columns),
            SourceLineage(
                "dataset",
                "build",
                "a" * 64,
                2,
                1,
                "table",
                NOW,
            ),
        )
