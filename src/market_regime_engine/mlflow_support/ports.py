"""Injectable tracking and registry ports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ResolvedModelVersion:
    model_name: str
    alias: str
    exact_version: str
    resolved_at_utc: datetime

    def __post_init__(self) -> None:
        if not self.model_name or not self.alias or not self.exact_version:
            raise ValueError("resolved model identity fields cannot be empty")
        if self.resolved_at_utc.tzinfo is None:
            raise ValueError("resolved_at_utc must be timezone-aware")


@dataclass(frozen=True, slots=True)
class MetricPoint:
    key: str
    value: float
    step: int
    timestamp_ms: int

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("metric key cannot be empty")
        if self.step < 0 or self.timestamp_ms < 0:
            raise ValueError("metric step/timestamp cannot be negative")


class TrackingPort(Protocol):
    def start_run(self, *, run_name: str, parent_run_id: str | None = None) -> str: ...

    def log_params(self, run_id: str, params: dict[str, str]) -> None: ...

    def log_metric_points(self, run_id: str, points: tuple[MetricPoint, ...]) -> None: ...

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str) -> None: ...


class RegistryPort(Protocol):
    def resolve_alias(self, model_name: str, alias: str) -> ResolvedModelVersion: ...

    def get_model_package_uri(self, model_name: str, exact_version: str) -> str: ...

    def compare_and_swap_alias(
        self,
        *,
        model_name: str,
        alias: str,
        expected_current_version: str | None,
        new_version: str,
        reason: str,
    ) -> bool: ...
