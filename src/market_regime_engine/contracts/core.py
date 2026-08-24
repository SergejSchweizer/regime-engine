"""Backend-independent immutable domain contracts for regime-engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isclose, isfinite
from string import hexdigits
from typing import TypeAlias

DATA_TIME_SEMANTICS = "current_vintage_observation_day"
PREDICTION_SCHEMA_VERSION = "RegimePrediction.v1"
INVOCATION_SCHEMA_VERSION = "RegimeInvocationResponse.v1"
ERROR_SCHEMA_VERSION = "RegimeError.v1"


def _require_text(value: str, field: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty trimmed string")


def _require_utc(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field} must be timezone-aware UTC")


def _require_sha256(value: str | None, field: str) -> None:
    if value is None:
        return
    if len(value) != 64 or any(character not in hexdigits for character in value):
        raise ValueError(f"{field} must be a 64-character SHA-256 hex digest")


def _require_probabilities(values: tuple[float, ...], field: str) -> None:
    if not values:
        raise ValueError(f"{field} cannot be empty")
    if any(not isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{field} must contain finite non-negative probabilities")
    if not isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"{field} must sum to one within 1e-10")


@dataclass(frozen=True, slots=True)
class SourceLineage:
    """Exact immutable upstream source identity bound to a read snapshot."""

    source_dataset: str
    source_build_id: str
    data_sha256: str
    schema_version: int
    feature_version: int
    source_table: str
    synced_at_utc: datetime
    data_time_semantics: str = DATA_TIME_SEMANTICS

    def __post_init__(self) -> None:
        for field_name in ("source_dataset", "source_build_id", "source_table"):
            _require_text(getattr(self, field_name), field_name)
        _require_sha256(self.data_sha256, "data_sha256")
        if self.schema_version < 1 or self.feature_version < 1:
            raise ValueError("source schema/feature versions must be positive")
        _require_utc(self.synced_at_utc, "synced_at_utc")
        if self.data_time_semantics != DATA_TIME_SEMANTICS:
            raise ValueError(f"unsupported data_time_semantics: {self.data_time_semantics}")


@dataclass(frozen=True, slots=True)
class FeatureSelectionLineage:
    """Definition and execution hashes are intentionally separate contracts."""

    feature_selection_definition_hash: str | None
    feature_selection_execution_hash: str | None

    def __post_init__(self) -> None:
        _require_sha256(
            self.feature_selection_definition_hash, "feature_selection_definition_hash"
        )
        _require_sha256(
            self.feature_selection_execution_hash, "feature_selection_execution_hash"
        )


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Immutable model/version plus the persisted causal filter boundary."""

    profile_id: str
    profile_config_version: int
    model_name: str
    model_version: str
    feature_contract_hash: str
    feature_order: tuple[str, ...]
    inference_origin_timestamp: datetime
    trained_through_timestamp: datetime
    terminal_filtered_probabilities: tuple[float, ...]
    covariance_type: str = "full"
    model_alias: str | None = None
    alias_resolved_at_utc: datetime | None = None

    def __post_init__(self) -> None:
        _require_text(self.profile_id, "profile_id")
        if self.profile_config_version < 1:
            raise ValueError("profile_config_version must be positive")
        for field_name in ("model_name", "model_version"):
            _require_text(getattr(self, field_name), field_name)
        _require_sha256(self.feature_contract_hash, "feature_contract_hash")
        if not self.feature_order or len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("feature_order must be non-empty and duplicate-free")
        if self.covariance_type != "full":
            raise ValueError("Gaussian covariance_type must be exactly 'full'")
        _require_utc(self.inference_origin_timestamp, "inference_origin_timestamp")
        _require_utc(self.trained_through_timestamp, "trained_through_timestamp")
        if self.inference_origin_timestamp > self.trained_through_timestamp:
            raise ValueError("inference origin cannot be after trained-through timestamp")
        _require_probabilities(
            self.terminal_filtered_probabilities, "terminal_filtered_probabilities"
        )
        if self.model_alias is not None:
            _require_text(self.model_alias, "model_alias")
        if self.alias_resolved_at_utc is not None:
            _require_utc(self.alias_resolved_at_utc, "alias_resolved_at_utc")


class PredictionMode(StrEnum):
    FIXED_MODEL_LATEST = "fixed_model_latest"
    FIXED_MODEL_REPLAY = "fixed_model_replay"
    WALK_FORWARD_OOS = "walk_forward_oos"


@dataclass(frozen=True, slots=True)
class RegimePrediction:
    schema_version: str
    timestamp: datetime
    state_ids: tuple[str, ...]
    state_probabilities: tuple[float, ...]
    dominant_state: str
    confidence: float
    entropy: float

    def __post_init__(self) -> None:
        if self.schema_version != PREDICTION_SCHEMA_VERSION:
            raise ValueError(f"prediction schema must be {PREDICTION_SCHEMA_VERSION}")
        _require_utc(self.timestamp, "timestamp")
        if not self.state_ids or len(set(self.state_ids)) != len(self.state_ids):
            raise ValueError("state_ids must be non-empty and unique")
        if len(self.state_ids) != len(self.state_probabilities):
            raise ValueError("state_ids/probabilities dimensions differ")
        _require_probabilities(self.state_probabilities, "state_probabilities")
        if self.dominant_state not in self.state_ids:
            raise ValueError("dominant_state must be one of state_ids")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite in [0, 1]")
        if not isfinite(self.entropy) or self.entropy < 0.0:
            raise ValueError("entropy must be finite and non-negative")


class InvocationOperation(StrEnum):
    LATEST = "latest"
    REPLAY = "replay"


@dataclass(frozen=True, slots=True)
class LatestInvocation:
    as_of: datetime | None = None
    model_version: str | None = None
    operation: InvocationOperation = InvocationOperation.LATEST

    def __post_init__(self) -> None:
        if self.operation is not InvocationOperation.LATEST:
            raise ValueError("LatestInvocation operation must be latest")
        if self.as_of is not None:
            _require_utc(self.as_of, "as_of")
        if self.model_version is not None:
            _require_text(self.model_version, "model_version")


@dataclass(frozen=True, slots=True)
class ReplayInvocation:
    start: datetime
    end: datetime
    model_version: str | None = None
    operation: InvocationOperation = InvocationOperation.REPLAY

    def __post_init__(self) -> None:
        if self.operation is not InvocationOperation.REPLAY:
            raise ValueError("ReplayInvocation operation must be replay")
        _require_utc(self.start, "start")
        _require_utc(self.end, "end")
        if self.start > self.end:
            raise ValueError("replay start must not be after end")
        if self.model_version is not None:
            _require_text(self.model_version, "model_version")


InvocationRequest: TypeAlias = LatestInvocation | ReplayInvocation


@dataclass(frozen=True, slots=True)
class RegimeInvocationResponse:
    schema_version: str
    request_id: str
    profile_id: str
    operation: InvocationOperation
    prediction_mode: PredictionMode
    requested_time_fields: tuple[tuple[str, str], ...]
    model: ModelIdentity
    source: SourceLineage
    selection: FeatureSelectionLineage
    warmup_observation_count: int
    skipped_incomplete_row_count: int
    predictions: tuple[RegimePrediction, ...]

    def __post_init__(self) -> None:
        if self.schema_version != INVOCATION_SCHEMA_VERSION:
            raise ValueError(f"invocation schema must be {INVOCATION_SCHEMA_VERSION}")
        _require_text(self.request_id, "request_id")
        _require_text(self.profile_id, "profile_id")
        if self.profile_id != self.model.profile_id:
            raise ValueError("response profile_id must match model profile_id")
        if self.warmup_observation_count < 0 or self.skipped_incomplete_row_count < 0:
            raise ValueError("observation counts cannot be negative")
        if not self.predictions:
            raise ValueError("successful invocation must contain at least one prediction")
        if self.operation is InvocationOperation.LATEST and len(self.predictions) != 1:
            raise ValueError("latest must contain exactly one prediction")
        expected_mode = {
            InvocationOperation.LATEST: PredictionMode.FIXED_MODEL_LATEST,
            InvocationOperation.REPLAY: PredictionMode.FIXED_MODEL_REPLAY,
        }[self.operation]
        if self.prediction_mode is not expected_mode:
            raise ValueError("invocation operation/prediction_mode mismatch")


@dataclass(frozen=True, slots=True)
class RegimeError:
    schema_version: str
    request_id: str
    error_code: str
    message: str
    retryable: bool
    details: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != ERROR_SCHEMA_VERSION:
            raise ValueError(f"error schema must be {ERROR_SCHEMA_VERSION}")
        for field_name in ("request_id", "error_code", "message"):
            _require_text(getattr(self, field_name), field_name)
