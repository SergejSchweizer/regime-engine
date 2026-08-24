"""Thin adapters from CLI action arguments to the operator service."""

from __future__ import annotations

from market_regime_engine.commands.contracts import (
    OperatorAction,
    OperatorRequest,
    OperatorResult,
    OperatorService,
)


def _execute(
    service: OperatorService,
    *,
    action: OperatorAction,
    profile_id: str,
    parameters: tuple[tuple[str, str], ...] = (),
) -> OperatorResult:
    request = OperatorRequest(
        action=action,
        profile_id=profile_id,
        parameters=tuple(sorted(parameters)),
    )
    result = service.execute(request)
    if result.action is not action or result.profile_id != profile_id:
        raise ValueError("operator service returned a mismatched command identity")
    return result


def evaluate(service: OperatorService, *, profile_id: str) -> OperatorResult:
    return _execute(service, action=OperatorAction.EVALUATE, profile_id=profile_id)


def final_refit(
    service: OperatorService,
    *,
    profile_id: str,
    evaluation_id: str,
) -> OperatorResult:
    return _execute(
        service,
        action=OperatorAction.FINAL_REFIT,
        profile_id=profile_id,
        parameters=(("evaluation_id", evaluation_id),),
    )


def register(
    service: OperatorService,
    *,
    profile_id: str,
    production_package: str,
    oos_build_id: str,
) -> OperatorResult:
    """Register only after the caller supplies explicit immutable OOS publication evidence."""

    return _execute(
        service,
        action=OperatorAction.REGISTER,
        profile_id=profile_id,
        parameters=(
            ("oos_build_id", oos_build_id),
            ("production_package", production_package),
        ),
    )


def publish_oos(
    service: OperatorService,
    *,
    profile_id: str,
    evaluation_id: str,
) -> OperatorResult:
    return _execute(
        service,
        action=OperatorAction.PUBLISH_OOS,
        profile_id=profile_id,
        parameters=(("evaluation_id", evaluation_id),),
    )


def status(service: OperatorService, *, profile_id: str) -> OperatorResult:
    return _execute(service, action=OperatorAction.STATUS, profile_id=profile_id)
