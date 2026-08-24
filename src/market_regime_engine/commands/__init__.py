"""Thin operator-command boundary for the regime-engine CLI."""

from market_regime_engine.commands.contracts import (
    OperatorAction,
    OperatorRequest,
    OperatorResult,
    OperatorService,
)
from market_regime_engine.commands.errors import OperatorCommandError

__all__ = [
    "OperatorAction",
    "OperatorCommandError",
    "OperatorRequest",
    "OperatorResult",
    "OperatorService",
]
