"""Immutable contracts and canonical JSON for full evaluation reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite
from string import hexdigits
from typing import Any, cast

EVALUATION_REPORT_SCHEMA_VERSION = "EvaluationReport.v1"

type JsonScalar = str | int | float | bool | None
type FrozenJson = JsonScalar | tuple["FrozenJson", ...] | tuple[tuple[str, "FrozenJson"], ...]
type JsonObject = tuple[tuple[str, FrozenJson], ...]

_FORBIDDEN_KEYS = {
    "raw_source_rows",
    "raw_feature_rows",
    "database_password",
    "password",
    "dsn",
    "secret",
    "secret_file_contents",
    "model_binary",
    "model_binary_payload",
}


def _text(value: str, field_name: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a non-empty trimmed string")


def _sha(value: str, field_name: str) -> None:
    if (
        len(value) != 64
        or value != value.lower()
        or any(character not in hexdigits for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field_name} must be timezone-aware UTC")


def _identifier_tuple(values: tuple[str, ...], field_name: str) -> None:
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{field_name} must be non-empty and duplicate-free")
    for value in values:
        _text(value, field_name)


def freeze_json(value: Any, *, _key: str | None = None) -> FrozenJson:
    """Convert JSON-compatible evidence into an immutable deterministic representation."""

    if _key in _FORBIDDEN_KEYS:
        raise ValueError(f"forbidden evaluation-report field: {_key}")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("evaluation-report floats must be finite")
        return value
    if isinstance(value, datetime):
        _utc(value, _key or "timestamp")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, bytes | bytearray | memoryview):
        raise ValueError("binary payloads are forbidden in evaluation reports")
    if isinstance(value, dict):
        result: list[tuple[str, FrozenJson]] = []
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("evaluation-report object keys must be strings")
            result.append((key, freeze_json(value[key], _key=key)))
        return tuple(result)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item, _key=_key) for item in value)
    raise TypeError(f"unsupported evaluation-report value type: {type(value).__name__}")


def object_payload(value: dict[str, Any]) -> JsonObject:
    frozen = freeze_json(value)
    if not _is_object(frozen):
        raise TypeError("report section payload must be an object")
    return cast(JsonObject, frozen)


def thaw_json(value: FrozenJson) -> Any:
    """Return ordinary JSON-compatible containers from frozen report evidence."""

    return _thaw(value)


def _is_object(value: FrozenJson) -> bool:
    return isinstance(value, tuple) and all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, datetime):
        _utc(value, "timestamp")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, tuple):
        if _is_object(value):
            return {key: _thaw(item) for key, item in value}
        return [_thaw(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {field.name: _thaw(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("evaluation-report floats must be finite")
    if isinstance(value, bytes | bytearray | memoryview):
        raise ValueError("binary payloads are forbidden in evaluation reports")
    return value


@dataclass(frozen=True, slots=True)
class ReportMetadata:
    parent_run_id: str
    parent_run_status: str
    profile_id: str
    profile_config_version: int
    source_dataset: str
    source_build_id: str
    data_sha256: str
    source_schema_version: int
    source_feature_version: int
    source_synced_at_utc: datetime
    data_time_semantics: str
    repository_git_sha: str
    build_provenance: str
    evaluation_plan_hash: str
    evaluation_cutoff: datetime
    feature_order: tuple[str, ...]
    feature_selection_definition_hash: str
    feature_selection_execution_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "parent_run_id",
            "parent_run_status",
            "profile_id",
            "source_dataset",
            "source_build_id",
            "data_time_semantics",
            "repository_git_sha",
            "build_provenance",
        ):
            _text(getattr(self, field_name), field_name)
        if self.profile_config_version < 1:
            raise ValueError("profile_config_version must be positive")
        if self.source_schema_version < 1 or self.source_feature_version < 1:
            raise ValueError("source schema/feature versions must be positive")
        for field_name in (
            "data_sha256",
            "evaluation_plan_hash",
            "feature_selection_definition_hash",
            "feature_selection_execution_hash",
        ):
            _sha(getattr(self, field_name), field_name)
        _utc(self.source_synced_at_utc, "source_synced_at_utc")
        _utc(self.evaluation_cutoff, "evaluation_cutoff")
        _identifier_tuple(self.feature_order, "feature_order")


@dataclass(frozen=True, slots=True)
class FeatureSelectionReport:
    policy_id: str
    final_features: tuple[str, ...]
    feature_selection_definition_hash: str
    feature_selection_execution_hash: str
    evidence: JsonObject

    def __post_init__(self) -> None:
        _text(self.policy_id, "policy_id")
        _identifier_tuple(self.final_features, "final_features")
        _sha(self.feature_selection_definition_hash, "feature_selection_definition_hash")
        _sha(self.feature_selection_execution_hash, "feature_selection_execution_hash")
        if not _is_object(self.evidence):
            raise ValueError("feature-selection evidence must be an immutable JSON object")


@dataclass(frozen=True, slots=True)
class FoldReport:
    fold_id: str
    fold_index: int
    valid: bool
    failure_reason: str | None
    evidence: JsonObject

    def __post_init__(self) -> None:
        _text(self.fold_id, "fold_id")
        if self.fold_index < 1:
            raise ValueError("fold_index must be positive")
        if self.valid == (self.failure_reason is not None):
            raise ValueError("valid fold has no failure reason; invalid fold requires one")
        if not _is_object(self.evidence):
            raise ValueError("fold evidence must be an immutable JSON object")


@dataclass(frozen=True, slots=True)
class CandidateReport:
    candidate_id: str
    state_count: int
    source_build_id: str
    evaluation_plan_hash: str
    feature_order: tuple[str, ...]
    feature_selection_definition_hash: str
    feature_selection_execution_hash: str
    folds: tuple[FoldReport, ...]
    summary: JsonObject

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.candidate_id, "candidate_id"),
            (self.source_build_id, "source_build_id"),
        ):
            _text(value, field_name)
        if self.state_count not in (2, 3, 4, 5):
            raise ValueError("state_count must be K=2,3,4,5")
        _sha(self.evaluation_plan_hash, "evaluation_plan_hash")
        _sha(self.feature_selection_definition_hash, "feature_selection_definition_hash")
        _sha(self.feature_selection_execution_hash, "feature_selection_execution_hash")
        _identifier_tuple(self.feature_order, "feature_order")
        fold_ids = tuple(item.fold_id for item in self.folds)
        if not fold_ids or len(set(fold_ids)) != len(fold_ids):
            raise ValueError("candidate folds must be non-empty and duplicate-free")
        if not _is_object(self.summary):
            raise ValueError("candidate summary must be an immutable JSON object")

    @property
    def fold_ids(self) -> tuple[str, ...]:
        return tuple(item.fold_id for item in self.folds)


@dataclass(frozen=True, slots=True)
class CrossCandidateComparisonReport:
    champion_candidate_id: str
    ranked_candidate_ids: tuple[str, ...]
    common_valid_fold_ids: tuple[str, ...]
    evidence: JsonObject

    def __post_init__(self) -> None:
        _text(self.champion_candidate_id, "champion_candidate_id")
        _identifier_tuple(self.ranked_candidate_ids, "ranked_candidate_ids")
        if self.champion_candidate_id != self.ranked_candidate_ids[0]:
            raise ValueError("champion must be first ranked candidate")
        if len(set(self.common_valid_fold_ids)) != len(self.common_valid_fold_ids):
            raise ValueError("common_valid_fold_ids cannot contain duplicates")
        if not _is_object(self.evidence):
            raise ValueError("comparison evidence must be an immutable JSON object")


@dataclass(frozen=True, slots=True)
class ReportIntegrity:
    report_payload_sha256: str

    def __post_init__(self) -> None:
        _sha(self.report_payload_sha256, "report_payload_sha256")


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    schema_version: str
    metadata: ReportMetadata
    configured_candidate_ids: tuple[str, ...]
    planned_fold_ids: tuple[str, ...]
    feature_selection: FeatureSelectionReport
    candidates: tuple[CandidateReport, ...]
    comparison: CrossCandidateComparisonReport
    integrity: ReportIntegrity | None = None

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_REPORT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {EVALUATION_REPORT_SCHEMA_VERSION}")
        _identifier_tuple(self.configured_candidate_ids, "configured_candidate_ids")
        _identifier_tuple(self.planned_fold_ids, "planned_fold_ids")
        expected = _expected_candidate_ids(self.metadata.profile_config_version)
        if self.configured_candidate_ids != expected:
            raise ValueError("configured candidate IDs do not match profile-version universe")
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        if candidate_ids != self.configured_candidate_ids:
            raise ValueError("candidate reports must exactly match configured candidate order")
        for candidate in self.candidates:
            if candidate.fold_ids != self.planned_fold_ids:
                raise ValueError("every candidate report must contain the exact planned fold IDs")
            if candidate.source_build_id != self.metadata.source_build_id:
                raise ValueError("candidate source build differs from report metadata")
            if candidate.evaluation_plan_hash != self.metadata.evaluation_plan_hash:
                raise ValueError("candidate evaluation plan differs from report metadata")
            if candidate.feature_order != self.metadata.feature_order:
                raise ValueError("candidate feature order differs from report metadata")
            if (
                candidate.feature_selection_definition_hash
                != self.metadata.feature_selection_definition_hash
                or candidate.feature_selection_execution_hash
                != self.metadata.feature_selection_execution_hash
            ):
                raise ValueError("candidate feature-selection hashes differ from report metadata")
        if self.feature_selection.final_features != self.metadata.feature_order:
            raise ValueError(
                "feature-selection final features differ from frozen report feature order"
            )
        if (
            self.feature_selection.feature_selection_definition_hash
            != self.metadata.feature_selection_definition_hash
            or self.feature_selection.feature_selection_execution_hash
            != self.metadata.feature_selection_execution_hash
        ):
            raise ValueError("feature-selection hashes differ from report metadata")
        if self.comparison.champion_candidate_id not in self.configured_candidate_ids:
            raise ValueError("comparison champion is outside configured candidates")
        if any(
            value not in self.configured_candidate_ids
            for value in self.comparison.ranked_candidate_ids
        ):
            raise ValueError("comparison ranking contains an unknown candidate")
        if any(
            value not in self.planned_fold_ids for value in self.comparison.common_valid_fold_ids
        ):
            raise ValueError("comparison common support contains an unknown fold")


def _expected_candidate_ids(profile_config_version: int) -> tuple[str, ...]:
    # Local import avoids making the report-contract module part of profile import initialization.
    from market_regime_engine.profiles.resolution import expected_candidate_ids

    return expected_candidate_ids(profile_config_version)


def report_payload_dict(report: EvaluationReport, *, include_integrity: bool) -> dict[str, Any]:
    payload = _thaw(report)
    assert isinstance(payload, dict)
    if not include_integrity:
        payload.pop("integrity", None)
    return payload


def canonical_payload_bytes(report: EvaluationReport) -> bytes:
    return json.dumps(
        report_payload_dict(report, include_integrity=False),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def compute_report_payload_sha256(report: EvaluationReport) -> str:
    return sha256(canonical_payload_bytes(report)).hexdigest()


def seal_report(report: EvaluationReport) -> EvaluationReport:
    digest = compute_report_payload_sha256(report)
    return replace(report, integrity=ReportIntegrity(digest))


def verify_report(report: EvaluationReport) -> str:
    if report.integrity is None:
        raise ValueError("evaluation report is not sealed")
    digest = compute_report_payload_sha256(report)
    if digest != report.integrity.report_payload_sha256:
        raise ValueError("evaluation report payload SHA-256 mismatch")
    return digest


def canonical_json_bytes(report: EvaluationReport) -> bytes:
    verify_report(report)
    return json.dumps(
        report_payload_dict(report, include_integrity=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
