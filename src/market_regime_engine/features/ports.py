"""Loader-independent feature-source port."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from market_regime_engine.contracts import SourceLineage


class SourceMode(StrEnum):
    FEATURE_SELECTION = "feature_selection"
    RESOLVED_MODEL = "resolved_model"


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
