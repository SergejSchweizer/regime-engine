"""Read-only PostgreSQL adapter for the regime-loader serving replica."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from math import isfinite
import re
from typing import Any, Protocol, cast

from psycopg import sql

from market_regime_engine.contracts import DATA_TIME_SEMANTICS, SourceLineage
from market_regime_engine.features.ports import (
    FeatureRequest,
    FeatureRow,
    FeatureSnapshot,
    SourceMode,
)

_DATASET_ID = "regime_features_daily"
_FEATURE_TABLE = sql.Identifier("regime_loader", "regime_features_daily")
_SYNC_TABLE = sql.Identifier("regime_loader_sync", "gold_sync_state")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class CursorLike(Protocol):
    description: Sequence[Any] | None

    def execute(self, query: Any, params: Sequence[Any] | None = None) -> Any: ...

    def fetchone(self) -> Sequence[Any] | None: ...

    def fetchall(self) -> Sequence[Sequence[Any]]: ...

    def __enter__(self) -> CursorLike: ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...


class ConnectionLike(Protocol):
    def cursor(self) -> CursorLike: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...


class PostgresFeatureSource:
    """Materialize one exact read-only source snapshot then close the transaction."""

    def __init__(
        self,
        connect: Callable[[], ConnectionLike],
        registered_feature_names: Iterable[str],
    ) -> None:
        self._connect = connect
        self._registered = frozenset(registered_feature_names)
        if not self._registered:
            raise ValueError("registered_feature_names cannot be empty")
        if any(_IDENTIFIER_RE.fullmatch(name) is None for name in self._registered):
            raise ValueError("registered feature names must be safe SQL identifiers")

    def read(self, request: FeatureRequest) -> FeatureSnapshot:
        self._validate_requested_features(request.feature_names)
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                lineage = self._read_lineage(cursor)
                rows = self._read_rows(cursor, request, lineage)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self._materialize(request, lineage, rows)

    def _validate_requested_features(self, names: tuple[str, ...]) -> None:
        invalid = [
            name
            for name in names
            if name not in self._registered or _IDENTIFIER_RE.fullmatch(name) is None
        ]
        if invalid:
            raise ValueError(f"unregistered or invalid feature columns: {invalid}")

    @staticmethod
    def _read_lineage(cursor: CursorLike) -> SourceLineage:
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
        source_build_id, digest, schema_version, feature_version, row_count, min_ts, max_ts, synced = row
        if int(row_count) < 0:
            raise ValueError("source row_count cannot be negative")
        min_timestamp = cast(datetime, min_ts)
        max_timestamp = cast(datetime, max_ts)
        if min_timestamp.tzinfo is None or max_timestamp.tzinfo is None:
            raise ValueError("source timestamp bounds must be timezone-aware")
        if min_timestamp > max_timestamp:
            raise ValueError("source timestamp bounds are inverted")
        synced_at = cast(datetime, synced).astimezone(UTC)
        return SourceLineage(
            source_dataset="regime_loader.regime_features_daily",
            source_build_id=str(source_build_id),
            data_sha256=str(digest),
            schema_version=int(schema_version),
            feature_version=int(feature_version),
            source_table="regime_loader.regime_features_daily",
            synced_at_utc=synced_at,
            data_time_semantics=DATA_TIME_SEMANTICS,
        )

    @staticmethod
    def _read_rows(
        cursor: CursorLike,
        request: FeatureRequest,
        lineage: SourceLineage,
    ) -> Sequence[Sequence[Any]]:
        del lineage
        columns = sql.SQL(", ").join([sql.Identifier("timestamp_m1"), *map(sql.Identifier, request.feature_names)])
        clauses: list[sql.Composed | sql.SQL] = []
        parameters: list[Any] = []
        if request.start is not None:
            clauses.append(sql.SQL("timestamp_m1 >= %s"))
            parameters.append(request.start)
        if request.end is not None:
            clauses.append(sql.SQL("timestamp_m1 <= %s"))
            parameters.append(request.end)
        where = sql.SQL("")
        if clauses:
            where = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(clauses)
        query = sql.SQL("SELECT {} FROM {}").format(columns, _FEATURE_TABLE) + where + sql.SQL(
            " ORDER BY timestamp_m1 ASC"
        )
        cursor.execute(query, tuple(parameters))
        return cursor.fetchall()

    @staticmethod
    def _materialize(
        request: FeatureRequest,
        lineage: SourceLineage,
        raw_rows: Sequence[Sequence[Any]],
    ) -> FeatureSnapshot:
        rows: list[FeatureRow] = []
        skipped = 0
        previous: datetime | None = None
        for raw in raw_rows:
            if len(raw) != len(request.feature_names) + 1:
                raise ValueError("feature row shape does not match requested columns")
            timestamp = cast(datetime, raw[0])
            if timestamp.tzinfo is None:
                raise ValueError("feature timestamp must be timezone-aware")
            timestamp = timestamp.astimezone(UTC)
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
