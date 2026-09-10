"""Read-only PostgreSQL adapter for the regime-loader serving replica."""

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

_DATASET_ID = "regime_features_daily"
_FEATURE_TABLE = sql.Identifier("regime_loader", "regime_features_daily")
_SYNC_TABLE = sql.Identifier("regime_loader_sync", "gold_sync_state")
_FEATURE_SCHEMA = "regime_loader"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RELATION_KINDS = {
    "r": "BASE TABLE",
    "p": "PARTITIONED TABLE",
    "v": "VIEW",
    "m": "MATERIALIZED VIEW",
}


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


class PostgresFeatureSource:
    """Materialize one exact read-only source snapshot then close the transaction."""

    def __init__(
        self,
        connect: Callable[[], ConnectionLike],
        registered_feature_names: Iterable[str] | None = None,
        *,
        expected_schema_version: int = 2,
        expected_feature_version: int = 1,
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
        """Read a dynamic catalog and rows in one repeatable-read transaction.

        An empty ``request.feature_names`` is the v4 schema-discovery path and
        reads every validated feature column.  A non-empty tuple remains
        available for callers that intentionally request a subset (for
        example, a resolved production model).
        """

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
        """Capture the complete dynamic feature universe without an allowlist."""

        return self.read_with_catalog(FeatureRequest.all_features(start=start, end=end))

    def read_schema_wide_with_catalog(
        self,
        request: FeatureRequest,
        *,
        feature_schema: str = _FEATURE_SCHEMA,
    ) -> tuple[FeatureCatalogSnapshot, FeatureSnapshot]:
        """Capture every valid relation and feature in one source transaction.

        This is the only source API intended for v4 discovery.  The caller
        must use an empty feature request; supplying names here would turn the
        schema-wide contract back into an allowlist.
        """

        if self._registered:
            raise ValueError("schema-wide discovery requires dynamic catalog mode")
        if request.feature_names:
            raise ValueError("schema-wide discovery does not accept a feature allowlist")
        if _IDENTIFIER_RE.fullmatch(feature_schema) is None:
            raise ValueError("feature_schema must be a safe SQL identifier")

        connection = self._connect()
        try:
            connection.read_only = True
            connection.isolation_level = IsolationLevel.REPEATABLE_READ
            with connection.cursor() as cursor:
                lineage = self._read_lineage(cursor)
                catalog = self._read_schema_wide_catalog(cursor, lineage, feature_schema)
                raw_rows = self._read_schema_wide_rows(cursor, request, catalog)
                snapshot = self._materialize_schema_wide(request, lineage, catalog, raw_rows)
                catalog = catalog.with_materialization(snapshot)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return catalog, snapshot

    def _read_snapshot(
        self, request: FeatureRequest, *, include_catalog: bool
    ) -> tuple[FeatureCatalogSnapshot | None, FeatureSnapshot]:
        connection = self._connect()
        try:
            # psycopg starts a transaction automatically on the first query. Set
            # its characteristics before issuing that query instead of sending a
            # redundant BEGIN inside the implicit transaction.
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
            raise ValueError("missing sync-state for regime_features_daily")
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
        synced_at = _utc_datetime(synced, "source synced_at_utc")
        return SourceLineage(
            source_dataset="regime_loader.regime_features_daily",
            source_build_id=str(source_build_id),
            data_sha256=str(digest),
            schema_version=schema_version_int,
            feature_version=feature_version_int,
            source_table="regime_loader.regime_features_daily",
            synced_at_utc=synced_at,
            data_time_semantics=DATA_TIME_SEMANTICS,
            row_count=row_count_int,
            min_timestamp=min_timestamp,
            max_timestamp=max_timestamp,
        )

    @staticmethod
    def _read_catalog(cursor: CursorLike, lineage: SourceLineage) -> FeatureCatalogSnapshot:
        query = sql.SQL(
            "SELECT column_name, ordinal_position, data_type, udt_name "
            "FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s "
            "ORDER BY ordinal_position ASC"
        )
        cursor.execute(query, ("regime_loader", "regime_features_daily"))
        columns = cursor.fetchall()
        if not columns:
            raise ValueError("feature table catalog is empty")
        timestamp_columns = [row for row in columns if len(row) >= 1 and row[0] == "timestamp_m1"]
        if len(timestamp_columns) != 1:
            raise ValueError("feature table must contain exactly one timestamp_m1 column")
        timestamp = timestamp_columns[0]
        if (
            len(timestamp) < 4
            or timestamp[2] != "timestamp with time zone"
            or timestamp[3] != "timestamptz"
        ):
            raise ValueError("timestamp_m1 must be PostgreSQL timestamp-with-time-zone")
        entries: list[FeatureCatalogEntry] = []
        for row in columns:
            if len(row) != 4:
                raise ValueError("unexpected information_schema column shape")
            name, ordinal, data_type, _udt_name = row
            if name == "timestamp_m1":
                continue
            if data_type != "double precision":
                raise ValueError(f"unsupported Gold feature type for {name}: {data_type}")
            if not isinstance(name, str) or _IDENTIFIER_RE.fullmatch(name) is None:
                raise ValueError("catalog contains an unsafe feature identifier")
            if not isinstance(ordinal, int) or isinstance(ordinal, bool):
                raise ValueError("catalog ordinal must be an integer")
            entries.append(FeatureCatalogEntry(name, ordinal))
        if not entries:
            raise ValueError("feature table has no dynamic Gold feature columns")
        return FeatureCatalogSnapshot.from_entries(lineage, "timestamp_m1", entries)

    @staticmethod
    def _read_schema_wide_catalog(
        cursor: CursorLike,
        lineage: SourceLineage,
        feature_schema: str,
    ) -> FeatureCatalogSnapshot:
        """Discover all feature relations and columns in canonical order."""

        relation_query = sql.SQL(
            "SELECT n.nspname, c.relname, c.relkind "
            "FROM pg_catalog.pg_class AS c "
            "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
            "WHERE n.nspname = %s "
            "AND c.relkind IN ('r', 'p', 'v', 'm', 'f') "
            "ORDER BY n.nspname ASC, c.relname ASC"
        )
        cursor.execute(relation_query, (feature_schema,))
        raw_relations = cursor.fetchall()
        if not raw_relations:
            raise ValueError(f"feature schema {feature_schema} contains no relations")

        relations: list[tuple[str, str, str]] = []
        for row in raw_relations:
            if len(row) != 3:
                raise ValueError("unexpected PostgreSQL relation catalog shape")
            schema_name, relation_name, relation_kind_code = row
            if (
                not isinstance(schema_name, str)
                or not isinstance(relation_name, str)
                or not isinstance(relation_kind_code, str)
            ):
                raise ValueError("PostgreSQL relation catalog contains invalid text")
            if schema_name != feature_schema:
                raise ValueError("relation catalog escaped the configured feature schema")
            if _IDENTIFIER_RE.fullmatch(relation_name) is None:
                raise ValueError(f"unsafe feature relation identifier: {relation_name}")
            relation_kind = _RELATION_KINDS.get(relation_kind_code)
            if relation_kind is None:
                raise ValueError(
                    f"unsupported relation kind {relation_kind_code!r} for "
                    f"{schema_name}.{relation_name}"
                )
            relations.append((schema_name, relation_name, relation_kind))
        if len({(schema, name) for schema, name, _ in relations}) != len(relations):
            raise ValueError("feature relation catalog contains duplicate relations")

        column_query = sql.SQL(
            "SELECT n.nspname, c.relname, c.relkind, a.attname, a.attnum, "
            "pg_catalog.format_type(a.atttypid, a.atttypmod), t.typname "
            "FROM pg_catalog.pg_class AS c "
            "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
            "JOIN pg_catalog.pg_attribute AS a ON a.attrelid = c.oid "
            "JOIN pg_catalog.pg_type AS t ON t.oid = a.atttypid "
            "WHERE n.nspname = %s "
            "AND c.relkind IN ('r', 'p', 'v', 'm', 'f') "
            "AND a.attnum > 0 AND NOT a.attisdropped "
            "ORDER BY n.nspname ASC, c.relname ASC, a.attnum ASC, a.attname ASC"
        )
        cursor.execute(column_query, (feature_schema,))
        raw_columns = cursor.fetchall()
        columns_by_relation: dict[tuple[str, str], list[tuple[Any, ...]]] = {
            (schema, relation): [] for schema, relation, _ in relations
        }
        for row in raw_columns:
            if len(row) != 7:
                raise ValueError("unexpected PostgreSQL column catalog shape")
            schema_name, relation_name, kind_code, *_ = row
            key = (schema_name, relation_name)
            if key not in columns_by_relation:
                raise ValueError("column catalog contains an unknown feature relation")
            if kind_code not in _RELATION_KINDS:
                raise ValueError("column catalog contains an unsupported relation kind")
            columns_by_relation[key].append(tuple(row))

        entries: list[FeatureCatalogEntry] = []
        seen_features: dict[str, tuple[str, str]] = {}
        canonical_ordinal = 1
        for schema_name, relation_name, relation_kind in sorted(relations):
            columns = sorted(
                columns_by_relation[(schema_name, relation_name)],
                key=lambda row: (int(row[4]), str(row[3])),
            )
            if not columns:
                raise ValueError(f"feature relation {schema_name}.{relation_name} has no columns")
            timestamp_columns = [row for row in columns if row[3] == "timestamp_m1"]
            if len(timestamp_columns) != 1:
                raise ValueError(
                    f"feature relation {schema_name}.{relation_name} must contain exactly one "
                    "timestamp_m1 column"
                )
            timestamp = timestamp_columns[0]
            if timestamp[5] != "timestamp with time zone" or timestamp[6] != "timestamptz":
                raise ValueError(
                    f"{schema_name}.{relation_name}.timestamp_m1 must be PostgreSQL "
                    "timestamp-with-time-zone"
                )
            feature_columns = [row for row in columns if row[3] != "timestamp_m1"]
            if not feature_columns:
                raise ValueError(
                    f"feature relation {schema_name}.{relation_name} has no feature columns"
                )
            for row in feature_columns:
                _, _, _, feature_name, ordinal, data_type, udt_name = row
                if (
                    not isinstance(feature_name, str)
                    or _IDENTIFIER_RE.fullmatch(feature_name) is None
                ):
                    raise ValueError(
                        f"unsafe feature identifier: {schema_name}.{relation_name}.{feature_name}"
                    )
                if data_type != "double precision" or udt_name != "float8":
                    raise ValueError(
                        f"unsupported Gold feature type for {schema_name}.{relation_name}."
                        f"{feature_name}: {data_type}/{udt_name}"
                    )
                if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 1:
                    raise ValueError("feature ordinal_position must be a positive integer")
                prior = seen_features.get(feature_name)
                if prior is not None:
                    raise ValueError(
                        f"duplicate feature name {feature_name}: "
                        f"{prior[0]}.{prior[1]} and {schema_name}.{relation_name}"
                    )
                seen_features[feature_name] = (schema_name, relation_name)
                entries.append(
                    FeatureCatalogEntry(
                        feature_name,
                        canonical_ordinal,
                        schema_name=schema_name,
                        relation_name=relation_name,
                        relation_kind=relation_kind,
                        ordinal_position=ordinal,
                    )
                )
                canonical_ordinal += 1
        if not entries:
            raise ValueError(f"feature schema {feature_schema} contains no feature columns")
        return FeatureCatalogSnapshot.from_entries(lineage, "timestamp_m1", entries)

    @staticmethod
    def _read_schema_wide_rows(
        cursor: CursorLike,
        request: FeatureRequest,
        catalog: FeatureCatalogSnapshot,
    ) -> tuple[tuple[str, tuple[FeatureCatalogEntry, ...], Sequence[Sequence[Any]]], ...]:
        if request.feature_names:
            raise ValueError("schema-wide row materialization does not accept a feature allowlist")
        by_relation: dict[tuple[str, str], list[FeatureCatalogEntry]] = {}
        for entry in catalog.entries:
            by_relation.setdefault((entry.schema_name, entry.relation_name), []).append(entry)
        results: list[tuple[str, tuple[FeatureCatalogEntry, ...], Sequence[Sequence[Any]]]] = []
        for schema_name, relation_name in sorted(by_relation):
            entries = tuple(by_relation[(schema_name, relation_name)])
            columns = sql.SQL(", ").join(
                [
                    sql.Identifier("timestamp_m1"),
                    *map(lambda entry: sql.Identifier(entry.feature_name), entries),
                ]
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
                sql.SQL("SELECT {} FROM {}").format(
                    columns,
                    sql.Identifier(schema_name, relation_name),
                )
                + where
                + sql.SQL(" ORDER BY timestamp_m1 ASC")
            )
            cursor.execute(query, tuple(parameters))
            results.append((f"{schema_name}.{relation_name}", entries, cursor.fetchall()))
        return tuple(results)

    @staticmethod
    def _materialize_schema_wide(
        request: FeatureRequest,
        lineage: SourceLineage,
        catalog: FeatureCatalogSnapshot,
        relation_rows: Sequence[
            tuple[str, tuple[FeatureCatalogEntry, ...], Sequence[Sequence[Any]]]
        ],
    ) -> FeatureSnapshot:
        if request.feature_names:
            raise ValueError("schema-wide materialization does not accept a feature allowlist")
        positions = {entry.feature_name: index for index, entry in enumerate(catalog.entries)}
        merged: dict[datetime, list[float | None]] = {}
        for relation_identity, entries, raw_rows in relation_rows:
            seen_timestamps: set[datetime] = set()
            for raw in raw_rows:
                if len(raw) != len(entries) + 1:
                    raise ValueError(f"row shape does not match relation {relation_identity}")
                timestamp = _utc_datetime(raw[0], f"{relation_identity}.timestamp_m1")
                if timestamp in seen_timestamps:
                    raise ValueError(f"duplicate timestamp in relation {relation_identity}")
                if lineage.min_timestamp is not None and timestamp < lineage.min_timestamp:
                    raise ValueError(f"{relation_identity} contains a row before source bounds")
                if lineage.max_timestamp is not None and timestamp > lineage.max_timestamp:
                    raise ValueError(f"{relation_identity} contains a row after source bounds")
                seen_timestamps.add(timestamp)
                values = merged.setdefault(timestamp, [None] * len(catalog.entries))
                for entry, raw_value in zip(entries, raw[1:], strict=True):
                    if raw_value is None:
                        value: float | None = None
                    else:
                        value = float(raw_value)
                        if not isfinite(value):
                            raise ValueError("non-null feature values must be finite")
                    position = positions[entry.feature_name]
                    if values[position] is not None and value is not None:
                        raise ValueError(
                            f"feature {entry.feature_name} received duplicate values at {timestamp}"
                        )
                    values[position] = value

        rows: list[FeatureRow] = []
        skipped = 0
        for timestamp in sorted(merged):
            row_values = tuple(merged[timestamp])
            if request.mode is SourceMode.RESOLVED_MODEL and any(
                value is None for value in row_values
            ):
                skipped += 1
                continue
            rows.append(FeatureRow(timestamp, row_values))
        return FeatureSnapshot(
            lineage=lineage,
            feature_names=catalog.feature_names,
            rows=tuple(rows),
            skipped_incomplete_row_count=skipped,
        )

    @staticmethod
    def _read_rows(
        cursor: CursorLike,
        request: FeatureRequest,
    ) -> Sequence[Sequence[Any]]:
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
