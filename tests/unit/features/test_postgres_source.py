from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

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


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
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
    first_query = str(cursor.executed[0][0])
    assert "REPEATABLE READ READ ONLY" in first_query


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
