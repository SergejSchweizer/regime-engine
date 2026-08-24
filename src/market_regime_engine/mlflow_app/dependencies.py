"""Injected dependency boundary for the custom regime-engine MLflow application."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from market_regime_engine.contracts import (
    LatestInvocation,
    RegimeInvocationResponse,
    ReplayInvocation,
)


class LatestHandlerPort(Protocol):
    def handle(
        self,
        *,
        request_id: str,
        profile_id: str,
        invocation: LatestInvocation,
        request_time_utc: datetime,
    ) -> RegimeInvocationResponse: ...


class ReplayHandlerPort(Protocol):
    def handle(
        self,
        *,
        request_id: str,
        profile_id: str,
        invocation: ReplayInvocation,
        request_time_utc: datetime,
    ) -> RegimeInvocationResponse: ...


class OOSHandlerPort(Protocol):
    def handle(
        self,
        *,
        profile_id: str,
        build_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class ReadinessSnapshot:
    """Non-secret readiness state exposed by the custom health route."""

    status: str
    ready: bool

    def __post_init__(self) -> None:
        if self.status not in {"healthy", "degraded", "not_ready"}:
            raise ValueError("readiness status must be healthy, degraded, or not_ready")
        if self.ready and self.status == "not_ready":
            raise ValueError("not_ready readiness state cannot be ready")


def _request_id() -> str:
    return uuid4().hex


def _request_time() -> datetime:
    return datetime.now(UTC)


def _ready() -> ReadinessSnapshot:
    return ReadinessSnapshot("healthy", True)


@dataclass(frozen=True, slots=True)
class ServiceDependencies:
    """All route dependencies are injected; route code owns no model/source math."""

    latest_handler: LatestHandlerPort
    replay_handler: ReplayHandlerPort
    oos_handler: OOSHandlerPort
    readiness: Callable[[], ReadinessSnapshot] = _ready
    request_id_factory: Callable[[], str] = _request_id
    request_time_factory: Callable[[], datetime] = _request_time

    def new_request_id(self) -> str:
        request_id = self.request_id_factory()
        if not request_id or request_id.strip() != request_id:
            raise ValueError("request ID factory must return a non-empty trimmed string")
        return request_id

    def request_time_utc(self) -> datetime:
        value = self.request_time_factory()
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("request time factory must return timezone-aware UTC")
        return value


_default_dependencies: ServiceDependencies | None = None


def configure_default_dependencies(dependencies: ServiceDependencies | None) -> None:
    """Install process-local dependencies during worker composition or clear them in tests."""

    global _default_dependencies
    _default_dependencies = dependencies


def resolve_dependencies(
    explicit: ServiceDependencies | None,
) -> ServiceDependencies | None:
    """Prefer app-local injection and otherwise use the process-local configured default."""

    return explicit if explicit is not None else _default_dependencies
