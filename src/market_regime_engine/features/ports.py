"""Loader-independent feature-source port."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.feature_discovery.contracts import content_hash


class SourceMode(StrEnum):
    FEATURE_SELECTION = "feature_selection"
    RESOLVED_MODEL = "resolved_model"


@dataclass(frozen=True, slots=True)
class FeatureCatalogEntry:
    """One validated Gold-table feature and its physical source ordinal."""

    feature_name: str
    canonical_ordinal: int
    postgres_type: str = "DOUBLE PRECISION"

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


@dataclass(frozen=True, slots=True)
class FeatureCatalogSnapshot:
    """A lineage-bound, canonically ordered feature catalog snapshot."""

    lineage: SourceLineage
    timestamp_column: str
    entries: tuple[FeatureCatalogEntry, ...]

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
            (entry.feature_name, entry.canonical_ordinal, entry.postgres_type)
            for entry in self.entries
        )
        return content_hash((lineage_identity, self.timestamp_column, entries))


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
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be non-empty and duplicate-free")
        if self.start is not None:
            _require_utc(self.start, "start")
        if self.end is not None:
            _require_utc(self.end, "end")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("start must not be after end")


@dataclass(frozen=True, slots=True)
class FeatureRow:
    timestamp: datetime
    values: tuple[float | None, ...]

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "timestamp")


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    lineage: SourceLineage
    feature_names: tuple[str, ...]
    rows: tuple[FeatureRow, ...]
    skipped_incomplete_row_count: int = 0

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


class FeatureSource(Protocol):
    def read(self, request: FeatureRequest) -> FeatureSnapshot: ...
