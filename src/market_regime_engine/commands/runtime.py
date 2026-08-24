"""Lazy production runtime composition for the installed operator CLI."""

from __future__ import annotations

from importlib import import_module
from typing import cast

from market_regime_engine.commands.contracts import OperatorService
from market_regime_engine.commands.errors import OperatorCommandError


def load_default_operator_service() -> OperatorService:
    """Load the lifecycle composition lazily once that later backlog layer is installed."""

    try:
        module = import_module("market_regime_engine.commands.lifecycle")
    except ModuleNotFoundError as exc:
        if exc.name != "market_regime_engine.commands.lifecycle":
            raise
        raise OperatorCommandError(
            "runtime_not_configured",
            "operator runtime composition is unavailable until lifecycle wiring is installed",
        ) from exc
    factory = getattr(module, "build_operator_service", None)
    if factory is None or not callable(factory):
        raise OperatorCommandError(
            "runtime_not_configured",
            "lifecycle module does not expose build_operator_service",
        )
    return cast(OperatorService, factory())
