"""Read-only PostgreSQL adapter for the canonical macro-loader view."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from math import isfinite
from typing import Any, Protocol, cast

from psycopg import IsolationLevel, sql

from market_regime_engine.contracts import DATA_TIME_SEMANTICS, SourceLineage
from market_regime_engine.features.ports import (
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
    FeatureRequest,
    FeatureRow,
    FeatureSnapshot,
    SourceMode,
)

_DATASET_ID = "macro_features"
_FEATURE_TABLE = sql.Identifier("macro_loader", "macro_features")
_SYNC_TABLE = sql.Identifier("macro_loader_sync", "gold_sync_state")
_CURRENT_SOURCE_SCHEMA_VERSION = 6
_CURRENT_SOURCE_FEATURE_VERSION = 5
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class CursorLike(Protocol):
    description: Sequence[Any] | None

    def execute(self, query: Any, params: Sequence[Any] | None = None) -> Any: ...

    def fetchone(self) -> Sequence[Any] | None: ...

    def fetchall(self) -> Sequence[Sequence[Any]]: ...

    def __enter__(self) -> CursorLike: ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...


class ConnectionLike(Protocol):
    read_only: bool
    isolation_level: IsolationLevel

    def cursor(self) -> CursorLike: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...


class MacroFeaturesPostgresSource:
    """Materialize one exact read-only snapshot from macro_features."""

    def __init__(
        self,
        connect: Callable[[], ConnectionLike],
        registered_feature_names: Iterable[str] | None = None,
        *,
        expected_schema_version: int = _CURRENT_SOURCE_SCHEMA_VERSION,
        expected_feature_version: int = _CURRENT_SOURCE_FEATURE_VERSION,
    ) -> None:
        self._connect = connect
        self._registered = frozenset(registered_feature_names or ())
        self._expected_schema_version = expected_schema_version
        self._expected_feature_version = expected_feature_version
        if any(_IDENTIFIER_RE.fullmatch(name) is None for name in self._registered):
            raise ValueError("registered feature names must be safe SQL identifiers")
        if expected_schema_version < 1 or expected_feature_version < 1:
            raise ValueError("expected source versions must be positive")

    def read(self, request: FeatureRequest) -> FeatureSnapshot:
        if self._registered:
            self._validate_requested_features(request.feature_names)
        _, snapshot = self._read_snapshot(request, include_catalog=False)
        return snapshot

    def read_with_catalog(
        self, request: FeatureRequest
    ) -> tuple[FeatureCatalogSnapshot, FeatureSnapshot]:
        """Read the fixed materialized-view catalog and rows atomically."""

        if self._registered:
            raise ValueError("read_with_catalog requires dynamic catalog mode")
        catalog, snapshot = self._read_snapshot(request, include_catalog=True)
        assert catalog is not None
        return catalog, snapshot

    def read_all_with_catalog(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[FeatureCatalogSnapshot, FeatureSnapshot]:
        return self.read_with_catalog(FeatureRequest.all_features(start=start, end=end))

    def _read_snapshot(
        self, request: FeatureRequest, *, include_catalog: bool
    ) -> tuple[FeatureCatalogSnapshot | None, FeatureSnapshot]:
        connection = self._connect()
        try:
            connection.read_only = True
            connection.isolation_level = IsolationLevel.REPEATABLE_READ
            with connection.cursor() as cursor:
                lineage = self._read_lineage(cursor)
                catalog = self._read_catalog(cursor, lineage) if not self._registered else None
                if catalog is not None:
                    if request.feature_names:
                        self._validate_requested_features(
                            request.feature_names, catalog.feature_names
                        )
                    effective_request = FeatureRequest(
                        request.feature_names or catalog.feature_names,
                        request.start,
                        request.end,
                        request.mode,
                    )
                else:
                    effective_request = request
                raw_rows = self._read_rows(cursor, effective_request)
                snapshot = self._materialize(effective_request, lineage, raw_rows)
                if catalog is not None and not request.feature_names:
                    catalog = catalog.with_materialization(snapshot)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        if include_catalog and catalog is None:
            raise ValueError("dynamic catalog was not materialized")
        return catalog, snapshot

    def _validate_requested_features(
        self, names: tuple[str, ...], registered: Iterable[str] | None = None
    ) -> None:
        available = self._registered if registered is None else frozenset(registered)
        invalid = [
            name
            for name in names
            if name not in available or _IDENTIFIER_RE.fullmatch(name) is None
        ]
        if invalid:
            raise ValueError(f"unregistered or invalid feature columns: {invalid}")

    def _read_lineage(self, cursor: CursorLike) -> SourceLineage:
        query = sql.SQL(
            "SELECT source_build_id, data_sha256, schema_version, feature_version, "
            "row_count, min_timestamp, max_timestamp, synced_at_utc "
            "FROM {} WHERE dataset_id = %s"
        ).format(_SYNC_TABLE)
        cursor.execute(query, (_DATASET_ID,))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"missing sync-state for {_DATASET_ID}")
        if len(row) != 8:
            raise ValueError("unexpected sync-state shape")
        (
            source_build_id,
            digest,
            schema_version,
            feature_version,
            row_count,
            min_ts,
            max_ts,
            synced,
        ) = row
        schema_version_int = int(schema_version)
        feature_version_int = int(feature_version)
        if schema_version_int != self._expected_schema_version:
            raise ValueError("incompatible source schema_version")
        if feature_version_int != self._expected_feature_version:
            raise ValueError("incompatible source feature_version")
        row_count_int = int(row_count)
        if row_count_int < 0:
            raise ValueError("source row_count cannot be negative")
        min_timestamp = _utc_datetime(min_ts, "source min_timestamp")
        max_timestamp = _utc_datetime(max_ts, "source max_timestamp")
        if min_timestamp > max_timestamp:
            raise ValueError("source timestamp bounds are inverted")
        return SourceLineage(
            source_dataset="macro_loader.macro_features",
            source_build_id=str(source_build_id),
            data_sha256=str(digest),
            schema_version=schema_version_int,
            feature_version=feature_version_int,
            source_table="macro_loader.macro_features",
            synced_at_utc=_utc_datetime(synced, "source synced_at_utc"),
            data_time_semantics=DATA_TIME_SEMANTICS,
            row_count=row_count_int,
            min_timestamp=min_timestamp,
            max_timestamp=max_timestamp,
        )

    @staticmethod
    def _read_catalog(cursor: CursorLike, lineage: SourceLineage) -> FeatureCatalogSnapshot:
        query = sql.SQL(
            "SELECT a.attname, a.attnum, pg_catalog.format_type(a.atttypid, a.atttypmod), "
            "t.typname FROM pg_catalog.pg_class AS c "
            "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
            "JOIN pg_catalog.pg_attribute AS a ON a.attrelid = c.oid "
            "JOIN pg_catalog.pg_type AS t ON t.oid = a.atttypid "
            "WHERE n.nspname = %s AND c.relname = %s AND c.relkind = 'm' "
            "AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attnum ASC"
        )
        cursor.execute(query, ("macro_loader", "macro_features"))
        columns = cursor.fetchall()
        if not columns:
            raise ValueError("macro_features materialized-view catalog is empty or missing")
        timestamp_columns = [row for row in columns if len(row) >= 1 and row[0] == "timestamp_m1"]
        if len(timestamp_columns) != 1:
            raise ValueError("macro_features must contain exactly one timestamp_m1 column")
        timestamp = timestamp_columns[0]
        if len(timestamp) != 4 or timestamp[3] != "timestamptz":
            raise ValueError("macro_features timestamp_m1 must be timestamp-with-time-zone")
        entries: list[FeatureCatalogEntry] = []
        for row in columns:
            if len(row) != 4:
                raise ValueError("unexpected macro_features catalog shape")
            name, ordinal, data_type, udt_name = row
            if name == "timestamp_m1":
                continue
            if data_type != "double precision" or udt_name != "float8":
                raise ValueError(f"unsupported Gold feature type for {name}: {data_type}")
            if not isinstance(name, str) or _IDENTIFIER_RE.fullmatch(name) is None:
                raise ValueError("macro_features contains an unsafe feature identifier")
            if not isinstance(ordinal, int) or isinstance(ordinal, bool):
                raise ValueError("macro_features catalog ordinal must be an integer")
            entries.append(
                FeatureCatalogEntry(
                    name,
                    ordinal,
                    schema_name="macro_loader",
                    relation_name="macro_features",
                    relation_kind="MATERIALIZED VIEW",
                    ordinal_position=ordinal,
                )
            )
        if not entries:
            raise ValueError("macro_features has no dynamic feature columns")
        return FeatureCatalogSnapshot.from_entries(lineage, "timestamp_m1", entries)

    @staticmethod
    def _read_rows(cursor: CursorLike, request: FeatureRequest) -> Sequence[Sequence[Any]]:
        columns = sql.SQL(", ").join(
            [sql.Identifier("timestamp_m1"), *map(sql.Identifier, request.feature_names)]
        )
        clauses: list[sql.Composed | sql.SQL] = []
        parameters: list[Any] = []
        if request.start is not None:
            clauses.append(sql.SQL("timestamp_m1 >= %s"))
            parameters.append(request.start)
        if request.end is not None:
            clauses.append(sql.SQL("timestamp_m1 <= %s"))
            parameters.append(request.end)
        where: sql.Composed | sql.SQL = sql.SQL("")
        if clauses:
            where = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(clauses)
        query = (
            sql.SQL("SELECT {} FROM {}").format(columns, _FEATURE_TABLE)
            + where
            + sql.SQL(" ORDER BY timestamp_m1 ASC")
        )
        cursor.execute(query, tuple(parameters))
        return cursor.fetchall()

    @staticmethod
    def _materialize(
        request: FeatureRequest,
        lineage: SourceLineage,
        raw_rows: Sequence[Sequence[Any]],
    ) -> FeatureSnapshot:
        if lineage.min_timestamp is None or lineage.max_timestamp is None:
            raise ValueError("source lineage is missing validated timestamp bounds")
        rows: list[FeatureRow] = []
        skipped = 0
        previous: datetime | None = None
        for raw in raw_rows:
            if len(raw) != len(request.feature_names) + 1:
                raise ValueError("feature row shape does not match requested columns")
            timestamp = _utc_datetime(raw[0], "feature timestamp")
            if not lineage.min_timestamp <= timestamp <= lineage.max_timestamp:
                raise ValueError("feature row lies outside validated source bounds")
            if previous is not None and timestamp <= previous:
                raise ValueError("feature timestamps must be unique and strictly increasing")
            previous = timestamp
            values: list[float | None] = []
            incomplete = False
            for value in raw[1:]:
                if value is None:
                    values.append(None)
                    incomplete = True
                    continue
                numeric = float(value)
                if not isfinite(numeric):
                    raise ValueError("non-null feature values must be finite")
                values.append(numeric)
            if request.mode is SourceMode.RESOLVED_MODEL and incomplete:
                skipped += 1
                continue
            rows.append(FeatureRow(timestamp=timestamp, values=tuple(values)))
        return FeatureSnapshot(
            lineage=lineage,
            feature_names=request.feature_names,
            rows=tuple(rows),
            skipped_incomplete_row_count=skipped,
        )


def _utc_datetime(value: Any, field: str) -> datetime:
    result = cast(datetime, value)
    if not isinstance(result, datetime):
        raise ValueError(f"{field} must be a datetime")
    if result.tzinfo is None or result.utcoffset() != UTC.utcoffset(result):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return result
