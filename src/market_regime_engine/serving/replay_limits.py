"""Pinned fixed-model replay limits and preflight validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(slots=True)
class ReplayGuardrailError(Exception):
    status_code: int
    error_code: str
    message: str
    retryable: bool

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class ReplayLimits:
    max_rows: int = 10_000
    max_internal_rows: int = 15_000
    max_range_days: int = 14_610
    timeout_seconds: float = 60.0
    max_response_bytes: int = 26_214_400
    max_concurrency_per_worker: int = 1

    def __post_init__(self) -> None:
        integer_limits = (
            self.max_rows,
            self.max_internal_rows,
            self.max_range_days,
            self.max_response_bytes,
            self.max_concurrency_per_worker,
        )
        if any(value < 1 for value in integer_limits) or self.timeout_seconds <= 0.0:
            raise ValueError("replay limits must all be positive")

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> ReplayLimits:
        return cls(
            max_rows=_int(env, "REGIME_REPLAY_MAX_ROWS", 10_000),
            max_internal_rows=_int(env, "REGIME_REPLAY_MAX_INTERNAL_ROWS", 15_000),
            max_range_days=_int(env, "REGIME_REPLAY_MAX_RANGE_DAYS", 14_610),
            timeout_seconds=_float(env, "REGIME_REPLAY_TIMEOUT_SECONDS", 60.0),
            max_response_bytes=_int(env, "REGIME_REPLAY_MAX_RESPONSE_BYTES", 26_214_400),
            max_concurrency_per_worker=_int(env, "REGIME_REPLAY_MAX_CONCURRENCY_PER_WORKER", 1),
        )

    def validate_interval(self, start: datetime, end: datetime) -> None:
        for value, name in ((start, "start"), (end, "end")):
            if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
                raise ReplayGuardrailError(
                    400,
                    "invalid_replay_interval",
                    f"{name} must be UTC",
                    False,
                )
        if start > end:
            raise ReplayGuardrailError(
                400,
                "invalid_replay_interval",
                "replay interval must be inclusive [start,end] with start <= end",
                False,
            )
        elapsed_days = (end - start).total_seconds() / 86_400.0
        if elapsed_days > self.max_range_days:
            raise ReplayGuardrailError(
                413,
                "replay_range_too_large",
                "replay interval exceeds configured range limit",
                False,
            )

    def validate_estimates(
        self,
        *,
        response_rows: int,
        internal_rows: int,
        estimated_response_bytes: int,
    ) -> None:
        values = (response_rows, internal_rows, estimated_response_bytes)
        if any(value < 0 for value in values):
            raise ValueError("replay estimates cannot be negative")
        if response_rows > self.max_rows:
            raise ReplayGuardrailError(413, "replay_row_limit", "replay row limit exceeded", False)
        if internal_rows > self.max_internal_rows:
            raise ReplayGuardrailError(
                413, "replay_internal_row_limit", "replay internal-row limit exceeded", False
            )
        if estimated_response_bytes > self.max_response_bytes:
            raise ReplayGuardrailError(
                413, "replay_response_too_large", "replay response-size limit exceeded", False
            )

    def validate_serialized_size(self, size_bytes: int) -> None:
        if size_bytes < 0:
            raise ValueError("serialized size cannot be negative")
        if size_bytes > self.max_response_bytes:
            raise ReplayGuardrailError(
                413, "replay_response_too_large", "replay response-size limit exceeded", False
            )


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(env.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
