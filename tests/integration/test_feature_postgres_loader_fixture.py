from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from market_regime_engine.features.ports import FeatureRequest, SourceMode
from market_regime_engine.features.postgres_source import PostgresFeatureSource

pytestmark = pytest.mark.integration


class LoaderFixtureCursor:
    def __init__(self, lineage: tuple[Any, ...], rows: tuple[tuple[Any, ...], ...]) -> None:
        self.lineage = lineage
        self.rows = rows
        self.execute_count = 0
        self.description = None

    def execute(self, query: Any, params: object = None) -> None:
        del query, params
        self.execute_count += 1

    def fetchone(self) -> tuple[Any, ...] | None:
        assert self.execute_count == 2
        return self.lineage

    def fetchall(self) -> tuple[tuple[Any, ...], ...]:
        assert self.execute_count == 3
        return self.rows

    def __enter__(self) -> LoaderFixtureCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb


class LoaderFixtureConnection:
    def __init__(self, lineage: tuple[Any, ...], rows: tuple[tuple[Any, ...], ...]) -> None:
        self.fixture_cursor = LoaderFixtureCursor(lineage, rows)
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> LoaderFixtureCursor:
        return self.fixture_cursor

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def loader_fixture() -> tuple[tuple[Any, ...], tuple[tuple[Any, ...], ...]]:
    start = datetime(2026, 1, 2, tzinfo=UTC)
    end = datetime(2026, 1, 4, tzinfo=UTC)
    lineage = (
        "loader-build-42",
        "a" * 64,
        1,
        1,
        3,
        start,
        end,
        datetime(2026, 1, 5, tzinfo=UTC),
    )
    rows = (
        (start, 1.0, 10.0),
        (datetime(2026, 1, 3, tzinfo=UTC), None, 11.0),
        (end, 3.0, 12.0),
    )
    return lineage, rows


def test_loader_shaped_fixture_preserves_lineage_nulls_and_read_only_snapshot_lifecycle() -> None:
    lineage, rows = loader_fixture()
    connection = LoaderFixtureConnection(lineage, rows)
    source = PostgresFeatureSource(lambda: connection, ("feature_a", "feature_b"))
    snapshot = source.read(
        FeatureRequest(
            feature_names=("feature_a", "feature_b"),
            start=lineage[5],
            end=lineage[6],
            mode=SourceMode.FEATURE_SELECTION,
        )
    )
    assert snapshot.lineage.source_build_id == "loader-build-42"
    assert snapshot.lineage.data_sha256 == "a" * 64
    assert snapshot.lineage.row_count == 3
    assert snapshot.lineage.min_timestamp == lineage[5]
    assert snapshot.lineage.max_timestamp == lineage[6]
    assert tuple(row.values for row in snapshot.rows) == (
        (1.0, 10.0),
        (None, 11.0),
        (3.0, 12.0),
    )
    assert snapshot.skipped_incomplete_row_count == 0
    assert connection.fixture_cursor.execute_count == 3
    assert connection.committed is True
    assert connection.rolled_back is False
    assert connection.closed is True


def test_loader_shaped_fixture_resolved_mode_excludes_null_row_without_fill() -> None:
    lineage, rows = loader_fixture()
    connection = LoaderFixtureConnection(lineage, rows)
    source = PostgresFeatureSource(lambda: connection, ("feature_a", "feature_b"))
    snapshot = source.read(
        FeatureRequest(
            feature_names=("feature_a", "feature_b"),
            start=lineage[5],
            end=lineage[6],
            mode=SourceMode.RESOLVED_MODEL,
        )
    )
    assert tuple(row.timestamp for row in snapshot.rows) == (lineage[5], lineage[6])
    assert snapshot.skipped_incomplete_row_count == 1
    assert connection.committed is True
    assert connection.closed is True
