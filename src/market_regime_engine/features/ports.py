"""Loader-independent feature-source port."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from typing import Any, Protocol, cast

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.feature_discovery.contracts import content_hash


class SourceMode(StrEnum):
    FEATURE_SELECTION = "feature_selection"
    RESOLVED_MODEL = "resolved_model"


@dataclass(frozen=True, slots=True)
class FeatureCatalogEntry:
    """One validated feature and its physical PostgreSQL origin."""

    feature_name: str
    canonical_ordinal: int
    postgres_type: str = "DOUBLE PRECISION"
    schema_name: str = "regime_loader"
    relation_name: str = "regime_features_daily"
    relation_kind: str = "BASE TABLE"
    ordinal_position: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.feature_name, str) or not re.fullmatch(
            r"[a-z_][a-z0-9_]*", self.feature_name
        ):
            raise ValueError("feature name must be a safe canonical SQL identifier")
        if self.feature_name == "timestamp_m1":
            raise ValueError("timestamp_m1 is metadata, not a feature entry")
        if self.canonical_ordinal < 1:
            raise ValueError("canonical feature ordinal must be positive")
        if self.postgres_type != "DOUBLE PRECISION":
            raise ValueError("v4 feature columns must be DOUBLE PRECISION")
        for value, field_name in (
            (self.schema_name, "schema_name"),
            (self.relation_name, "relation_name"),
        ):
            if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a safe SQL identifier")
        if self.relation_kind not in {
            "BASE TABLE",
            "PARTITIONED TABLE",
            "VIEW",
            "MATERIALIZED VIEW",
        }:
            raise ValueError("unsupported feature relation kind")
        ordinal = self.canonical_ordinal if self.ordinal_position is None else self.ordinal_position
        if ordinal < 1:
            raise ValueError("feature ordinal_position must be positive")
        object.__setattr__(self, "ordinal_position", ordinal)


@dataclass(frozen=True, slots=True)
class FeatureCatalogSnapshot:
    """A lineage-bound, canonically ordered feature catalog snapshot."""

    lineage: SourceLineage
    timestamp_column: str
    entries: tuple[FeatureCatalogEntry, ...]
    materialized_feature_data_sha256: str | None = None
    materialized_row_count: int | None = None
    materialized_min_timestamp: datetime | None = None
    materialized_max_timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.timestamp_column != "timestamp_m1":
            raise ValueError("catalog timestamp column must be timestamp_m1")
        if not self.entries:
            raise ValueError("feature catalog must contain at least one entry")
        if any(not isinstance(entry, FeatureCatalogEntry) for entry in self.entries):
            raise ValueError("feature catalog entries must use FeatureCatalogEntry")
        ordinals = tuple(entry.canonical_ordinal for entry in self.entries)
        names = tuple(entry.feature_name for entry in self.entries)
        if ordinals != tuple(sorted(ordinals)) or len(set(ordinals)) != len(ordinals):
            raise ValueError("feature catalog entries must be unique and ordinal ordered")
        if len(set(names)) != len(names):
            raise ValueError("feature catalog names must be unique")
        materialization = (
            self.materialized_feature_data_sha256,
            self.materialized_row_count,
            self.materialized_min_timestamp,
            self.materialized_max_timestamp,
        )
        if any(value is not None for value in materialization):
            if self.materialized_feature_data_sha256 is None:
                raise ValueError("materialized feature data hash is required")
            _require_sha256(
                self.materialized_feature_data_sha256,
                "materialized_feature_data_sha256",
            )
            if self.materialized_row_count is None or self.materialized_row_count < 0:
                raise ValueError("materialized_row_count must be non-negative")
            if self.materialized_row_count == 0:
                if (
                    self.materialized_min_timestamp is not None
                    or self.materialized_max_timestamp is not None
                ):
                    raise ValueError("empty materialization cannot have timestamp bounds")
            else:
                if (
                    self.materialized_min_timestamp is None
                    or self.materialized_max_timestamp is None
                ):
                    raise ValueError("non-empty materialization requires timestamp bounds")
                _require_utc(self.materialized_min_timestamp, "materialized_min_timestamp")
                _require_utc(self.materialized_max_timestamp, "materialized_max_timestamp")
                if self.materialized_min_timestamp > self.materialized_max_timestamp:
                    raise ValueError("materialized timestamp bounds are inverted")

    @classmethod
    def from_entries(
        cls,
        lineage: SourceLineage,
        timestamp_column: str,
        entries: Iterable[FeatureCatalogEntry],
    ) -> FeatureCatalogSnapshot:
        """Canonicalize an unordered physical mapping by declared ordinal."""

        ordered = tuple(sorted(tuple(entries), key=lambda entry: entry.canonical_ordinal))
        return cls(lineage, timestamp_column, ordered)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(entry.feature_name for entry in self.entries)

    def with_materialization(self, snapshot: FeatureSnapshot) -> FeatureCatalogSnapshot:
        """Bind the exact materialized matrix identity to this catalog."""

        if snapshot.feature_names != self.feature_names:
            raise ValueError("materialized snapshot columns must match the feature catalog")
        return replace(
            self,
            materialized_feature_data_sha256=snapshot.materialized_feature_data_sha256,
            materialized_row_count=len(snapshot.rows),
            materialized_min_timestamp=snapshot.rows[0].timestamp if snapshot.rows else None,
            materialized_max_timestamp=snapshot.rows[-1].timestamp if snapshot.rows else None,
        )

    @property
    def catalog_hash(self) -> str:
        # SourceLineage also carries the non-decision ``data_time_semantics``
        # label.  The discovery canonicalizer intentionally rejects semantic
        # decision fields, so hash the complete identity fields explicitly and
        # omit that fixed compatibility label.
        lineage_identity = (
            self.lineage.source_dataset,
            self.lineage.source_build_id,
            self.lineage.data_sha256,
            self.lineage.schema_version,
            self.lineage.feature_version,
            self.lineage.source_table,
            self.lineage.synced_at_utc,
            self.lineage.row_count,
            self.lineage.min_timestamp,
            self.lineage.max_timestamp,
        )
        entries = tuple(
            (
                entry.schema_name,
                entry.relation_name,
                entry.relation_kind,
                entry.feature_name,
                entry.ordinal_position,
                entry.canonical_ordinal,
                entry.postgres_type,
            )
            for entry in self.entries
        )
        materialization = (
            self.materialized_feature_data_sha256,
            self.materialized_row_count,
            self.materialized_min_timestamp,
            self.materialized_max_timestamp,
        )
        return content_hash((lineage_identity, self.timestamp_column, entries, materialization))


def _require_utc(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field} must be timezone-aware UTC")


@dataclass(frozen=True, slots=True)
class FeatureRequest:
    feature_names: tuple[str, ...]
    start: datetime | None
    end: datetime | None
    mode: SourceMode

    def __post_init__(self) -> None:
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be duplicate-free")
        if not self.feature_names and self.mode is not SourceMode.FEATURE_SELECTION:
            raise ValueError("all-feature requests are only valid for feature selection")
        if self.start is not None:
            _require_utc(self.start, "start")
        if self.end is not None:
            _require_utc(self.end, "end")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("start must not be after end")

    @classmethod
    def all_features(
        cls,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> FeatureRequest:
        """Request the complete dynamic source catalog without a name allowlist."""

        return cls((), start, end, SourceMode.FEATURE_SELECTION)


@dataclass(frozen=True, slots=True)
class FeatureRow:
    timestamp: datetime
    values: tuple[float | None, ...]

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "timestamp")


def materialized_feature_data_hash(
    feature_names: tuple[str, ...], rows: Iterable[FeatureRow]
) -> str:
    """Hash an ordered feature matrix using a versioned, locale-free encoding."""

    payload = {
        "encoding_version": "feature_matrix_v1",
        "feature_names": list(feature_names),
        "rows": [
            [
                row.timestamp.isoformat(),
                [None if value is None else _float_hex(value) for value in row.values],
            ]
            for row in rows
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    lineage: SourceLineage
    feature_names: tuple[str, ...]
    rows: tuple[FeatureRow, ...]
    skipped_incomplete_row_count: int = 0
    materialized_feature_data_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be non-empty and duplicate-free")
        if self.skipped_incomplete_row_count < 0:
            raise ValueError("skipped_incomplete_row_count cannot be negative")
        expected_dimension = len(self.feature_names)
        previous: datetime | None = None
        for row in self.rows:
            if len(row.values) != expected_dimension:
                raise ValueError("feature row values do not match feature_names")
            if previous is not None and row.timestamp <= previous:
                raise ValueError("snapshot timestamps must be unique and strictly increasing")
            previous = row.timestamp
        computed_hash = materialized_feature_data_hash(self.feature_names, self.rows)
        if (
            self.materialized_feature_data_sha256 is not None
            and self.materialized_feature_data_sha256 != computed_hash
        ):
            raise ValueError("materialized feature data hash does not match snapshot rows")
        object.__setattr__(self, "materialized_feature_data_sha256", computed_hash)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def min_timestamp(self) -> datetime | None:
        return self.rows[0].timestamp if self.rows else None

    @property
    def max_timestamp(self) -> datetime | None:
        return self.rows[-1].timestamp if self.rows else None


class FeatureSource(Protocol):
    def read(self, request: FeatureRequest) -> FeatureSnapshot: ...


class DynamicFeatureSource(Protocol):
    def read_with_catalog(
        self, request: FeatureRequest
    ) -> tuple[FeatureCatalogSnapshot, FeatureSnapshot]: ...


class SchemaWideFeatureSource(Protocol):
    def read_schema_wide_with_catalog(
        self,
        request: FeatureRequest,
    ) -> tuple[FeatureCatalogSnapshot, FeatureSnapshot]: ...


_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _require_sha256(value: str, field: str) -> None:
    if (
        len(value) != 64
        or value != value.lower()
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _float_hex(value: object) -> str:
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError("non-null feature values must be numeric") from exc
    if not isfinite(numeric):
        raise ValueError("non-null feature values must be finite")
    return numeric.hex()
