"""Strict request parsing, safe serialization, and HTTP error mapping for MLflow routes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, cast

from mlflow.exceptions import MlflowException

from market_regime_engine.contracts import (
    ERROR_SCHEMA_VERSION,
    LatestInvocation,
    ReplayInvocation,
)
from market_regime_engine.serving.latest_handler import StaleDefaultChampionError
from market_regime_engine.serving.replay_limits import ReplayGuardrailError


@dataclass(slots=True)
class ApiInputError(Exception):
    error_code: str
    message: str
    status_code: int = 400
    retryable: bool = False

    def __str__(self) -> str:
        return self.message


def parse_utc_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ApiInputError("invalid_timestamp", f"{field_name} must be an RFC3339 UTC string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ApiInputError(
            "invalid_timestamp",
            f"{field_name} must be an RFC3339 UTC string",
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ApiInputError("invalid_timestamp", f"{field_name} must use UTC Z/zero offset")
    return parsed.astimezone(UTC)


def _optional_model_version(payload: dict[str, object]) -> str | None:
    value = payload.get("model_version")
    if value is None:
        return None
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ApiInputError(
            "invalid_model_version",
            "model_version must be a non-empty trimmed string",
        )
    return value


def _reject_unknown(payload: dict[str, object], allowed: frozenset[str]) -> None:
    unknown = tuple(sorted(set(payload) - allowed))
    if unknown:
        raise ApiInputError(
            "unknown_request_field",
            f"unknown request field(s): {', '.join(unknown)}",
        )


def parse_invocation_payload(payload: object) -> LatestInvocation | ReplayInvocation:
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ApiInputError("invalid_request_body", "request body must be a JSON object")
    body = cast(dict[str, object], payload)
    if "profile_id" in body:
        raise ApiInputError("profile_in_body_forbidden", "profile_id is path-only")
    operation = body.get("operation")
    if operation == "latest":
        _reject_unknown(body, frozenset({"operation", "as_of", "model_version"}))
        as_of = body.get("as_of")
        return LatestInvocation(
            as_of=None if as_of is None else parse_utc_timestamp(as_of, "as_of"),
            model_version=_optional_model_version(body),
        )
    if operation == "replay":
        _reject_unknown(body, frozenset({"operation", "start", "end", "model_version"}))
        if "start" not in body or "end" not in body:
            raise ApiInputError(
                "missing_replay_bound",
                "replay requires both start and end",
            )
        try:
            return ReplayInvocation(
                start=parse_utc_timestamp(body["start"], "start"),
                end=parse_utc_timestamp(body["end"], "end"),
                model_version=_optional_model_version(body),
            )
        except ValueError as exc:
            raise ApiInputError("invalid_replay_interval", "replay requires start <= end") from exc
    if operation is None:
        raise ApiInputError("missing_operation", "operation is required")
    raise ApiInputError("unsupported_operation", "operation must be latest or replay")


def parse_json_body(raw: bytes, *, is_json: bool) -> LatestInvocation | ReplayInvocation:
    if not is_json:
        raise ApiInputError("invalid_content_type", "request Content-Type must be application/json")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiInputError("malformed_json", "request body is not valid JSON") from exc
    return parse_invocation_payload(payload)


def parse_oos_query(arguments: dict[str, str]) -> tuple[datetime | None, datetime | None]:
    unknown = tuple(sorted(set(arguments) - {"start", "end"}))
    if unknown:
        raise ApiInputError(
            "unknown_query_field",
            f"unknown query field(s): {', '.join(unknown)}",
        )
    start = arguments.get("start")
    end = arguments.get("end")
    parsed_start = None if start is None else parse_utc_timestamp(start, "start")
    parsed_end = None if end is None else parse_utc_timestamp(end, "end")
    if parsed_start is not None and parsed_end is not None and parsed_start > parsed_end:
        raise ApiInputError("invalid_oos_interval", "OOS slice requires start <= end")
    return parsed_start, parsed_end


def to_jsonable(value: object) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("serialized datetime must be timezone-aware UTC")
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported API response value: {type(value).__name__}")


def _error_payload(
    *,
    request_id: str,
    error_code: str,
    message: str,
    retryable: bool,
    details: tuple[tuple[str, str], ...] = (),
) -> dict[str, object]:
    return {
        "schema_version": ERROR_SCHEMA_VERSION,
        "request_id": request_id,
        "error_code": error_code,
        "message": message,
        "retryable": retryable,
        "details": dict(details),
    }


def map_exception(exc: Exception, request_id: str) -> tuple[dict[str, object], int]:
    if isinstance(exc, ApiInputError):
        return (
            _error_payload(
                request_id=request_id,
                error_code=exc.error_code,
                message=exc.message,
                retryable=exc.retryable,
            ),
            exc.status_code,
        )
    if isinstance(exc, ReplayGuardrailError):
        return (
            _error_payload(
                request_id=request_id,
                error_code=exc.error_code,
                message=exc.message,
                retryable=exc.retryable,
            ),
            exc.status_code,
        )
    if isinstance(exc, StaleDefaultChampionError):
        return (
            _error_payload(
                request_id=request_id,
                error_code=exc.error_code,
                message="default champion latest exceeds configured staleness limits",
                retryable=exc.retryable,
            ),
            exc.status_code,
        )
    if isinstance(exc, KeyError):
        return (
            _error_payload(
                request_id=request_id,
                error_code="resource_not_found",
                message="requested profile or resource was not found",
                retryable=False,
            ),
            404,
        )
    if isinstance(exc, FileNotFoundError):
        return (
            _error_payload(
                request_id=request_id,
                error_code="oos_build_not_found",
                message="requested immutable OOS build was not found",
                retryable=False,
            ),
            404,
        )
    if isinstance(exc, MlflowException):
        error_code = str(getattr(exc, "error_code", ""))
        if error_code == "RESOURCE_DOES_NOT_EXIST":
            return (
                _error_payload(
                    request_id=request_id,
                    error_code="model_version_not_found",
                    message="requested model version was not found",
                    retryable=False,
                ),
                404,
            )
        return (
            _error_payload(
                request_id=request_id,
                error_code="registry_unavailable",
                message="model registry dependency is unavailable",
                retryable=True,
            ),
            503,
        )
    if isinstance(exc, ValueError):
        return (
            _error_payload(
                request_id=request_id,
                error_code="semantic_validation_failed",
                message="model/source request could not be satisfied",
                retryable=False,
            ),
            422,
        )
    return (
        _error_payload(
            request_id=request_id,
            error_code="dependency_unavailable",
            message="regime-engine dependency is unavailable",
            retryable=True,
        ),
        503,
    )
