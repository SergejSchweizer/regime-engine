from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from psycopg import IsolationLevel

from market_regime_engine.features import FeatureRequest, SourceMode
from market_regime_engine.features.postgres_source import PostgresFeatureSource

START = datetime(2026, 9, 1, tzinfo=UTC)


class SchemaCursor:
    def __init__(
        self,
        relations: list[tuple[Any, ...]],
        columns: list[tuple[Any, ...]],
        rows_by_relation: list[list[tuple[Any, ...]]],
    ) -> None:
        self.relations = relations
        self.columns = columns
        self.rows_by_relation = rows_by_relation
        self.executed: list[tuple[Any, Any]] = []
        self.fetchall_count = 0

    def execute(self, query: Any, params: Any = None) -> None:
        self.executed.append((query, params))

    def fetchone(self) -> tuple[Any, ...]:
        return (
            "build-schema-1",
            "a" * 64,
            4,
            3,
            2,
            START,
            START + timedelta(days=2),
            START,
        )

    def fetchall(self) -> list[tuple[Any, ...]]:
        self.fetchall_count += 1
        if self.fetchall_count == 1:
            return self.relations
        if self.fetchall_count == 2:
            return self.columns
        return self.rows_by_relation[self.fetchall_count - 3]

    def __enter__(self) -> SchemaCursor:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb


class SchemaConnection:
    def __init__(self, cursor: SchemaCursor) -> None:
        self._cursor = cursor
        self.read_only = False
        self.isolation_level: object | None = None
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> SchemaCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def _relations() -> list[tuple[Any, ...]]:
    return [
        ("regime_loader", "alpha_features", "r"),
        ("regime_loader", "beta_features", "r"),
    ]


def _columns() -> list[tuple[Any, ...]]:
    return [
        (
            "regime_loader",
            "alpha_features",
            "r",
            "timestamp_m1",
            1,
            "timestamp with time zone",
            "timestamptz",
        ),
        ("regime_loader", "alpha_features", "r", "alpha", 2, "double precision", "float8"),
        (
            "regime_loader",
            "beta_features",
            "r",
            "timestamp_m1",
            1,
            "timestamp with time zone",
            "timestamptz",
        ),
        ("regime_loader", "beta_features", "r", "beta", 2, "double precision", "float8"),
    ]


def _connection(
    *,
    relations: list[tuple[Any, ...]] | None = None,
    columns: list[tuple[Any, ...]] | None = None,
    rows_by_relation: list[list[tuple[Any, ...]]] | None = None,
) -> SchemaConnection:
    cursor = SchemaCursor(
        _relations() if relations is None else relations,
        _columns() if columns is None else columns,
        (
            [
                (START, 1.0),
                (START + timedelta(days=1), 2.0),
            ],
            [
                (START + timedelta(days=1), 20.0),
                (START + timedelta(days=2), 30.0),
            ],
        )
        if rows_by_relation is None
        else rows_by_relation,
    )
    return SchemaConnection(cursor)


def test_schema_wide_request_discovers_and_unions_every_relation() -> None:
    connection = _connection()
    catalog, snapshot = PostgresFeatureSource(lambda: connection).read_schema_wide_with_catalog(
        FeatureRequest.all_features(start=START, end=START + timedelta(days=2))
    )

    assert catalog.feature_names == ("alpha", "beta")
    assert tuple(row.timestamp for row in snapshot.rows) == (
        START,
        START + timedelta(days=1),
        START + timedelta(days=2),
    )
    assert tuple(row.values for row in snapshot.rows) == (
        (1.0, None),
        (2.0, 20.0),
        (None, 30.0),
    )
    assert catalog.materialized_feature_data_sha256 == snapshot.materialized_feature_data_sha256
    assert catalog.materialized_row_count == 3
    assert catalog.materialized_min_timestamp == START
    assert catalog.materialized_max_timestamp == START + timedelta(days=2)
    assert len(catalog.catalog_hash) == 64
    assert connection.read_only is True
    assert connection.isolation_level is IsolationLevel.REPEATABLE_READ
    assert connection.committed and connection.closed and not connection.rolled_back
    assert len(connection._cursor.executed) == 5
    assert connection._cursor.executed[-1][1] == (START, START + timedelta(days=2))


def test_schema_wide_accepts_explicit_postgres_timestamp_precision() -> None:
    columns = _columns()
    columns[0] = (
        "regime_loader",
        "alpha_features",
        "r",
        "timestamp_m1",
        1,
        "timestamp(6) with time zone",
        "timestamptz",
    )
    connection = _connection(columns=columns)
    catalog, snapshot = PostgresFeatureSource(lambda: connection).read_schema_wide_with_catalog(
        FeatureRequest.all_features()
    )

    assert catalog.feature_names == ("alpha", "beta")
    assert snapshot.rows[0].timestamp == START


def test_schema_wide_catalog_and_materialization_are_order_independent() -> None:
    first = _connection()
    second = _connection(
        relations=list(reversed(_relations())),
        columns=list(reversed(_columns())),
        rows_by_relation=[
            [(START + timedelta(days=1), 2.0), (START, 1.0)],
            [(START + timedelta(days=2), 30.0), (START + timedelta(days=1), 20.0)],
        ],
    )
    first_catalog, first_snapshot = PostgresFeatureSource(
        lambda: first
    ).read_schema_wide_with_catalog(FeatureRequest.all_features())
    second_catalog, second_snapshot = PostgresFeatureSource(
        lambda: second
    ).read_schema_wide_with_catalog(FeatureRequest.all_features())

    assert second_catalog.feature_names == first_catalog.feature_names
    assert second_catalog.catalog_hash == first_catalog.catalog_hash
    assert (
        second_snapshot.materialized_feature_data_sha256
        == first_snapshot.materialized_feature_data_sha256
    )


def test_schema_wide_materialization_preserves_explicit_database_nulls() -> None:
    connection = _connection(
        rows_by_relation=[
            [(START, None)],
            [(START, 20.0)],
        ]
    )

    _, snapshot = PostgresFeatureSource(lambda: connection).read_schema_wide_with_catalog(
        FeatureRequest.all_features()
    )

    assert snapshot.rows[0].values == (None, 20.0)


def test_schema_wide_discovery_rejects_unsupported_relation_kind() -> None:
    connection = _connection(relations=[("regime_loader", "foreign_features", "f")])
    with pytest.raises(ValueError, match="unsupported relation kind"):
        PostgresFeatureSource(lambda: connection).read_schema_wide_with_catalog(
            FeatureRequest.all_features()
        )
    assert connection.rolled_back and connection.closed


def test_schema_wide_discovery_rejects_duplicate_bare_feature_names() -> None:
    columns = _columns()
    columns[-1] = (
        "regime_loader",
        "beta_features",
        "r",
        "alpha",
        2,
        "double precision",
        "float8",
    )
    connection = _connection(columns=columns)
    with pytest.raises(ValueError, match="duplicate feature name alpha"):
        PostgresFeatureSource(lambda: connection).read_schema_wide_with_catalog(
            FeatureRequest.all_features()
        )


def test_new_database_feature_changes_snapshot_identity_without_code_configuration_change() -> None:
    baseline = _connection()
    extended_columns = [
        *_columns(),
        ("regime_loader", "beta_features", "r", "beta_new", 3, "double precision", "float8"),
    ]
    extended_rows = [
        [(START, 1.0), (START + timedelta(days=1), 2.0)],
        [
            (START + timedelta(days=1), 20.0, 200.0),
            (START + timedelta(days=2), 30.0, 300.0),
        ],
    ]
    baseline_catalog, baseline_snapshot = PostgresFeatureSource(
        lambda: baseline
    ).read_schema_wide_with_catalog(FeatureRequest.all_features())
    extended = _connection(columns=extended_columns, rows_by_relation=extended_rows)
    extended_catalog, extended_snapshot = PostgresFeatureSource(
        lambda: extended
    ).read_schema_wide_with_catalog(FeatureRequest.all_features())

    assert extended_catalog.feature_names == ("alpha", "beta", "beta_new")
    assert extended_catalog.catalog_hash != baseline_catalog.catalog_hash
    assert (
        extended_snapshot.materialized_feature_data_sha256
        != baseline_snapshot.materialized_feature_data_sha256
    )
    assert extended_snapshot.rows[1].values == (2.0, 20.0, 200.0)


def test_schema_wide_api_rejects_a_caller_feature_allowlist() -> None:
    connection = _connection()
    with pytest.raises(ValueError, match="does not accept a feature allowlist"):
        PostgresFeatureSource(lambda: connection).read_schema_wide_with_catalog(
            FeatureRequest(("alpha",), None, None, SourceMode.SCHEMA_DISCOVERY)
        )


@pytest.mark.parametrize(
    ("relations", "columns", "message"),
    [
        ([], _columns(), "contains no relations"),
        ([("regime_loader", "alpha_features")], _columns(), "relation catalog shape"),
        ([(None, "alpha_features", "r")], _columns(), "invalid text"),
        ([("other_schema", "alpha_features", "r")], _columns(), "escaped"),
        ([("regime_loader", "bad-relation", "r")], _columns(), "unsafe feature relation"),
        ([("regime_loader", "alpha_features", "r")] * 2, _columns(), "duplicate relations"),
        ([("regime_loader", "alpha_features", "r")], [], "has no columns"),
    ],
)
def test_schema_wide_relation_catalog_failures(
    relations: list[tuple[Any, ...]],
    columns: list[tuple[Any, ...]],
    message: str,
) -> None:
    connection = _connection(relations=relations, columns=columns)
    with pytest.raises(ValueError, match=message):
        PostgresFeatureSource(lambda: connection).read_schema_wide_with_catalog(
            FeatureRequest.all_features()
        )
    assert connection.rolled_back and connection.closed


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        (_columns()[1:], "exactly one"),
        ([*_columns(), _columns()[0]], "exactly one"),
        (
            [
                *_columns()[:1],
                (
                    "regime_loader",
                    "alpha_features",
                    "r",
                    "alpha",
                    2,
                    "numeric",
                    "numeric",
                ),
                *_columns()[2:],
            ],
            "unsupported Gold feature type",
        ),
        (
            [
                *_columns()[:1],
                (
                    "regime_loader",
                    "alpha_features",
                    "r",
                    "Bad-Feature",
                    2,
                    "double precision",
                    "float8",
                ),
                *_columns()[2:],
            ],
            "unsafe feature identifier",
        ),
        (
            [
                *_columns()[:1],
                (
                    "regime_loader",
                    "alpha_features",
                    "r",
                    "alpha",
                    "two",
                    "double precision",
                    "float8",
                ),
                *_columns()[2:],
            ],
            "ordinal_position",
        ),
        (
            [
                (
                    "regime_loader",
                    "alpha_features",
                    "r",
                    "timestamp_m1",
                    1,
                    "timestamp without time zone",
                    "timestamp",
                ),
                *_columns()[1:],
            ],
            "timestamp-with-time-zone",
        ),
        (
            [
                *_columns()[:-1],
                (
                    "regime_loader",
                    "beta_features",
                    "r",
                    "alpha",
                    2,
                    "double precision",
                    "float8",
                ),
            ],
            "duplicate feature name",
        ),
    ],
)
def test_schema_wide_column_contract_failures(columns: list[tuple[Any, ...]], message: str) -> None:
    connection = _connection(columns=columns)
    with pytest.raises(ValueError, match=message):
        PostgresFeatureSource(lambda: connection).read_schema_wide_with_catalog(
            FeatureRequest.all_features()
        )
    assert connection.rolled_back and connection.closed


def test_schema_wide_column_catalog_shape_and_unknown_relation_fail_closed() -> None:
    for columns, message in (
        (
            [
                *_columns(),
                ("regime_loader", "alpha_features", "r", "broken", 2, "double precision"),
            ],
            "column catalog shape",
        ),
        (
            [
                *_columns(),
                ("regime_loader", "unknown", "r", "broken", 2, "double precision", "float8"),
            ],
            "unknown feature relation",
        ),
        (
            [
                *_columns()[:1],
                ("regime_loader", "alpha_features", "f", "alpha", 2, "double precision", "float8"),
                *_columns()[2:],
            ],
            "unsupported relation kind",
        ),
    ):
        connection = _connection(columns=columns)
        with pytest.raises(ValueError, match=message):
            PostgresFeatureSource(
                lambda connection=connection: connection
            ).read_schema_wide_with_catalog(FeatureRequest.all_features())
        assert connection.rolled_back and connection.closed


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([[(START,)], [(START + timedelta(days=1), 20.0)]], "row shape"),
        (
            [[(START, 1.0), (START, 2.0)], [(START + timedelta(days=1), 20.0)]],
            "duplicate timestamp",
        ),
        (
            [[(START - timedelta(days=1), 1.0)], [(START + timedelta(days=1), 20.0)]],
            "before source bounds",
        ),
        (
            [[(START, 1.0)], [(START + timedelta(days=3), 20.0)]],
            "after source bounds",
        ),
        (
            [[(START, float("inf"))], [(START + timedelta(days=1), 20.0)]],
            "finite",
        ),
    ],
)
def test_schema_wide_row_materialization_failures(
    rows: list[list[tuple[Any, ...]]], message: str
) -> None:
    connection = _connection(rows_by_relation=rows)
    with pytest.raises(ValueError, match=message):
        PostgresFeatureSource(lambda: connection).read_schema_wide_with_catalog(
            FeatureRequest.all_features()
        )
    assert connection.rolled_back and connection.closed


def test_schema_wide_source_validates_runtime_mode_and_identifier() -> None:
    connection = _connection()
    with pytest.raises(ValueError, match="dynamic catalog mode"):
        PostgresFeatureSource(lambda: connection, ("alpha",)).read_schema_wide_with_catalog(
            FeatureRequest.all_features()
        )
    with pytest.raises(ValueError, match="safe SQL identifier"):
        PostgresFeatureSource(lambda: connection).read_schema_wide_with_catalog(
            FeatureRequest.all_features(), feature_schema="bad-schema"
        )
    with pytest.raises(ValueError, match="registered feature names"):
        PostgresFeatureSource(lambda: connection, ("bad-name",))
    with pytest.raises(ValueError, match="versions"):
        PostgresFeatureSource(lambda: connection, expected_schema_version=0)
