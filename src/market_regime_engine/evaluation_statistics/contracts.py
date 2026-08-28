"""Finite, deterministic schema for local evaluation statistics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite

from market_regime_engine.evaluations.contracts import EvaluationId

SCHEMA_VERSION = 1
_FORBIDDEN = ("dsn", "password", "secret", "credential", "raw_feature", "source_rows")
_EVIDENCE_GROUPS = {
    "identity",
    "lineage",
    "input",
    "model",
    "folds",
    "states",
    "aggregate",
    "feature_selection",
    "agreement",
    "champion",
    "failure",
}


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


@dataclass(frozen=True, slots=True)
class RunStatistics:
    evaluation_id: EvaluationId
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
        payload["evaluation_id"] = self.evaluation_id.value
        payload["run_type"] = self.run_type.value
        payload["status"] = self.status.value
        payload["started_at"] = self.started_at.isoformat()
        payload["ended_at"] = self.ended_at.isoformat() if self.ended_at else None
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return (canonical + "\n").encode()
