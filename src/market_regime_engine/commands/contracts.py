"""Immutable request/result schemas for operator commands."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class OperatorAction(StrEnum):
    EVALUATE = "evaluate"
    FINAL_REFIT = "final-refit"
    REGISTER = "register"
    PUBLISH_OOS = "publish-oos"
    STATUS = "status"


def _validate_fields(fields: tuple[tuple[str, str], ...], *, label: str) -> None:
    keys = tuple(key for key, _ in fields)
    if keys != tuple(sorted(keys)):
        raise ValueError(f"{label} fields must be sorted by key")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} fields must have unique keys")
    if any(not key or key.strip() != key for key in keys):
        raise ValueError(f"{label} field keys must be non-empty trimmed strings")
    if any(not value or value.strip() != value for _, value in fields):
        raise ValueError(f"{label} field values must be non-empty trimmed strings")


@dataclass(frozen=True, slots=True)
class OperatorRequest:
    action: OperatorAction
    profile_id: str
    parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_id or self.profile_id.strip() != self.profile_id:
            raise ValueError("profile_id must be a non-empty trimmed string")
        _validate_fields(self.parameters, label="request")

    def parameter(self, key: str) -> str | None:
        for field, value in self.parameters:
            if field == key:
                return value
        return None

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "action": self.action.value,
                "parameters": dict(self.parameters),
                "profile_id": self.profile_id,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )


@dataclass(frozen=True, slots=True)
class OperatorResult:
    action: OperatorAction
    profile_id: str
    fields: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.profile_id or self.profile_id.strip() != self.profile_id:
            raise ValueError("result profile_id must be a non-empty trimmed string")
        _validate_fields(self.fields, label="result")

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "action": self.action.value,
                "fields": dict(self.fields),
                "profile_id": self.profile_id,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )


class OperatorService(Protocol):
    def execute(self, request: OperatorRequest) -> OperatorResult: ...
