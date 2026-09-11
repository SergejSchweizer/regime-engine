"""Finite, deterministic schema for local evaluation statistics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from math import isfinite

SCHEMA_VERSION = 1
GLOBAL_V4_SCHEMA_VERSION = 1
GLOBAL_V4_EVALUATION_ID = "global_regime_v4"
_FORBIDDEN = ("dsn", "password", "secret", "credential", "raw_feature", "source_rows")
_EVIDENCE_GROUPS = {
    "identity",
    "lineage",
    "input",
    "model",
    "folds",
    "states",
    "aggregate",
    "feature_discovery",
    "agreement",
    "champion",
    "optimization",
    "failure",
}
_GLOBAL_V4_EVIDENCE_GROUPS = {
    "identity",
    "lineage",
    "input",
    "quality",
    "distance",
    "clustering",
    "prototypes",
    "teacher",
    "feature_scores",
    "prefix_search",
    "final_grid",
    "outer_folds",
    "agreement",
    "validity",
    "stability",
    "deployment_selection",
    "failure",
}
_REQUIRED_GLOBAL_V4_EVIDENCE_GROUPS = _GLOBAL_V4_EVIDENCE_GROUPS - {
    "deployment_selection",
    "failure",
}
_EVIDENCE_GROUPS.update(_GLOBAL_V4_EVIDENCE_GROUPS)


class RunType(StrEnum):
    PARENT = "parent"
    FEATURE = "feature"
    CANDIDATE = "candidate"


class Status(StrEnum):
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


def _utc(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{name} must be timezone-aware UTC")


def _safe(value: object, field_name: str = "") -> None:
    if any(token in field_name.lower() for token in _FORBIDDEN):
        raise ValueError(f"forbidden statistics field: {field_name}")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("statistics values must be finite")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("statistics mapping keys must be strings")
            _safe(item, key)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _safe(item, field_name)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError("statistics values must be JSON primitives, mappings, or sequences")


def _evaluation_id_value(value: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("evaluation_id must be a non-empty trimmed string")
    return value


@dataclass(frozen=True, slots=True)
class GlobalV4Evidence:
    """Canonical, model-binary-free evidence payload for one global-v4 run."""

    source_build_id: str
    source_data_hash: str
    catalog_hash: str
    profile_hash: str
    repository_hash: str
    outer_plan_hash: str
    evidence: dict[str, object]
    schema_version: int = GLOBAL_V4_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GLOBAL_V4_SCHEMA_VERSION:
            raise ValueError("global v4 evidence schema version is unsupported")
        if not self.source_build_id or self.source_build_id.strip() != self.source_build_id:
            raise ValueError("source_build_id must be a non-empty trimmed string")
        for value, name in (
            (self.source_data_hash, "source_data_hash"),
            (self.catalog_hash, "catalog_hash"),
            (self.profile_hash, "profile_hash"),
            (self.repository_hash, "repository_hash"),
            (self.outer_plan_hash, "outer_plan_hash"),
        ):
            if (
                len(value) != 64
                or value != value.lower()
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not isinstance(self.evidence, dict):
            raise TypeError("global v4 evidence groups must be a mapping")
        unknown = set(self.evidence) - _GLOBAL_V4_EVIDENCE_GROUPS
        if unknown:
            raise ValueError(f"unknown global v4 evidence groups: {sorted(unknown)}")
        missing = _REQUIRED_GLOBAL_V4_EVIDENCE_GROUPS - set(self.evidence)
        if missing:
            raise ValueError(f"missing global v4 evidence groups: {sorted(missing)}")
        _safe(self.evidence)

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-compatible payload without operational run metadata."""

        return {
            "evaluation_id": GLOBAL_V4_EVALUATION_ID,
            "schema_version": self.schema_version,
            "source_build_id": self.source_build_id,
            "source_data_hash": self.source_data_hash,
            "catalog_hash": self.catalog_hash,
            "profile_hash": self.profile_hash,
            "repository_hash": self.repository_hash,
            "outer_plan_hash": self.outer_plan_hash,
            "evidence": self.evidence,
        }

    def canonical_json(self) -> bytes:
        return (
            json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            + "\n"
        ).encode("utf-8")

    @property
    def evidence_hash(self) -> str:
        return sha256(self.canonical_json()).hexdigest()


@dataclass(frozen=True, slots=True)
class RunStatistics:
    evaluation_id: str
    mlflow_run_id: str
    run_type: RunType
    run_name: str
    status: Status
    started_at: datetime
    parent_run_id: str | None = None
    ended_at: datetime | None = None
    evidence: dict[str, object] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("statistics schema version is unsupported")
        _evaluation_id_value(self.evaluation_id)
        for name in ("mlflow_run_id", "run_name"):
            value = getattr(self, name)
            if not value or value.strip() != value:
                raise ValueError(f"{name} must be a non-empty trimmed string")
        if self.parent_run_id is not None and (
            not self.parent_run_id or self.parent_run_id.strip() != self.parent_run_id
        ):
            raise ValueError("parent_run_id must be a non-empty trimmed string when present")
        _utc(self.started_at, "started_at")
        if self.ended_at is not None:
            _utc(self.ended_at, "ended_at")
            if self.ended_at < self.started_at:
                raise ValueError("ended_at must not precede started_at")
        if self.status is Status.RUNNING and self.ended_at is not None:
            raise ValueError("RUNNING statistics cannot have ended_at")
        if self.status is not Status.RUNNING and self.ended_at is None:
            raise ValueError("final statistics require ended_at")
        unknown = set(self.evidence) - _EVIDENCE_GROUPS
        if unknown:
            raise ValueError(f"unknown statistics evidence groups: {sorted(unknown)}")
        if self.status is Status.FAILED:
            failure = self.evidence.get("failure")
            if not isinstance(failure, dict) or not all(
                isinstance(failure.get(key), str) and failure[key].strip()
                for key in ("code", "reason")
            ):
                raise ValueError("FAILED statistics require failure code and reason")
        _safe(self.evidence)

    def canonical_json(self) -> bytes:
        payload = asdict(self)
        payload["evaluation_id"] = _evaluation_id_value(self.evaluation_id)
        payload["run_type"] = self.run_type.value
        payload["status"] = self.status.value
        payload["started_at"] = self.started_at.isoformat()
        payload["ended_at"] = self.ended_at.isoformat() if self.ended_at else None
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return (canonical + "\n").encode()
