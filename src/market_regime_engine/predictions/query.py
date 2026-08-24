"""Explicit immutable walk-forward OOS build queries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any, cast

from market_regime_engine.contracts import (
    PREDICTION_SCHEMA_VERSION,
    PredictionMode,
    RegimePrediction,
)
from market_regime_engine.predictions.store import PredictionBuildManifest, PredictionStore


def _require_text(value: str, field: str) -> str:
    if not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty trimmed string")
    return value


def _require_utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return value


@dataclass(frozen=True, slots=True)
class OOSQuery:
    profile_id: str
    build_id: str
    start: datetime | None = None
    end: datetime | None = None

    def __post_init__(self) -> None:
        _require_text(self.profile_id, "profile_id")
        _require_text(self.build_id, "build_id")
        if self.start is not None:
            _require_utc(self.start, "start")
        if self.end is not None:
            _require_utc(self.end, "end")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("OOS slice must satisfy inclusive start <= end")


@dataclass(frozen=True, slots=True)
class OOSPredictionRow:
    prediction: RegimePrediction
    candidate_id: str
    fold_id: str
    evaluation_plan_hash: str
    feature_selection_definition_hash: str
    feature_selection_execution_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "candidate_id",
            "fold_id",
            "evaluation_plan_hash",
            "feature_selection_definition_hash",
            "feature_selection_execution_hash",
        ):
            _require_text(cast(str, getattr(self, field_name)), field_name)


@dataclass(frozen=True, slots=True)
class OOSBuildSlice:
    manifest: PredictionBuildManifest
    requested_start: datetime | None
    requested_end: datetime | None
    rows: tuple[OOSPredictionRow, ...]

    @property
    def profile_id(self) -> str:
        return self.manifest.profile_id

    @property
    def build_id(self) -> str:
        return self.manifest.build_id

    @property
    def prediction_mode(self) -> PredictionMode:
        return self.manifest.prediction_mode


def _row_timestamp(row: dict[str, Any]) -> datetime:
    value = row.get("timestamp")
    if not isinstance(value, datetime):
        raise ValueError("OOS row timestamp must be a datetime")
    return _require_utc(value, "OOS row timestamp")


def _prediction_from_row(row: dict[str, Any]) -> RegimePrediction:
    if row.get("schema_version") != PREDICTION_SCHEMA_VERSION:
        raise ValueError("OOS row has incompatible prediction schema version")
    if row.get("prediction_mode") != PredictionMode.WALK_FORWARD_OOS.value:
        raise ValueError("OOS row prediction_mode must be exactly walk_forward_oos")
    timestamp = _row_timestamp(row)
    raw_state_ids = row.get("state_ids")
    raw_probabilities = row.get("state_probabilities")
    if not isinstance(raw_state_ids, list) or not all(isinstance(value, str) for value in raw_state_ids):
        raise ValueError("OOS row state_ids must be a list of strings")
    if not isinstance(raw_probabilities, list):
        raise ValueError("OOS row state_probabilities must be a list")
    state_ids = tuple(cast(list[str], raw_state_ids))
    try:
        probabilities = tuple(float(value) for value in raw_probabilities)
        confidence = float(row["confidence"])
        entropy = float(row["entropy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("OOS row numeric prediction fields are invalid") from exc
    dominant_state = row.get("dominant_state")
    if not isinstance(dominant_state, str):
        raise ValueError("OOS row dominant_state must be a string")
    return RegimePrediction(
        schema_version=PREDICTION_SCHEMA_VERSION,
        timestamp=timestamp,
        state_ids=state_ids,
        state_probabilities=probabilities,
        dominant_state=dominant_state,
        confidence=confidence,
        entropy=entropy,
    )


def _metadata_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"OOS row {key} must be a string")
    return _require_text(value, key)


def query_oos_build(store: PredictionStore, query: OOSQuery) -> OOSBuildSlice:
    """Read one explicit immutable build and apply an inclusive UTC timestamp slice."""

    manifest = store.load_manifest(query.profile_id, query.build_id)
    if manifest.profile_id != query.profile_id or manifest.build_id != query.build_id:
        raise ValueError("OOS manifest identity differs from explicit query")
    if manifest.prediction_mode is not PredictionMode.WALK_FORWARD_OOS:
        raise ValueError("requested prediction build is not walk_forward_oos")

    raw_rows = cast(list[dict[str, Any]], store.read_table(query.profile_id, query.build_id).to_pylist())
    parsed: list[OOSPredictionRow] = []
    timestamps: list[datetime] = []
    for row in raw_rows:
        if row.get("profile_id") != query.profile_id:
            raise ValueError("OOS row profile_id differs from immutable build profile")
        prediction = _prediction_from_row(row)
        timestamps.append(prediction.timestamp)
        if query.start is not None and prediction.timestamp < query.start:
            continue
        if query.end is not None and prediction.timestamp > query.end:
            continue
        parsed.append(
            OOSPredictionRow(
                prediction=prediction,
                candidate_id=_metadata_text(row, "candidate_id"),
                fold_id=_metadata_text(row, "fold_id"),
                evaluation_plan_hash=_metadata_text(row, "evaluation_plan_hash"),
                feature_selection_definition_hash=_metadata_text(
                    row, "feature_selection_definition_hash"
                ),
                feature_selection_execution_hash=_metadata_text(
                    row, "feature_selection_execution_hash"
                ),
            )
        )
    if any(left >= right for left, right in pairwise(timestamps)):
        raise ValueError("OOS build timestamps must be globally unique and strictly increasing")
    return OOSBuildSlice(
        manifest=manifest,
        requested_start=query.start,
        requested_end=query.end,
        rows=tuple(parsed),
    )
