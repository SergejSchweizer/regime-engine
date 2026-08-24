"""Loader-independent feature-source port."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from market_regime_engine.contracts import SourceLineage


class SourceMode(StrEnum):
    FEATURE_SELECTION = "feature_selection"
    RESOLVED_MODEL = "resolved_model"


@dataclass(frozen=True, slots=True)
class FeatureRequest:
    feature_names: tuple[str, ...]
    start: datetime | None
    end: datetime | None
    mode: SourceMode

    def __post_init__(self) -> None:
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be non-empty and duplicate-free")
        if self.start is not None and self.start.utcoffset() is None:
            raise ValueError("start must be timezone-aware")
        if self.end is not None and self.end.utcoffset() is None:
            raise ValueError("end must be timezone-aware")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("start must not be after end")


@dataclass(frozen=True, slots=True)
class FeatureRow:
    timestamp: datetime
    values: tuple[float | None, ...]


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    lineage: SourceLineage
    feature_names: tuple[str, ...]
    rows: tuple[FeatureRow, ...]
    skipped_incomplete_row_count: int = 0


class FeatureSource(Protocol):
    def read(self, request: FeatureRequest) -> FeatureSnapshot: ...
