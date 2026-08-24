"""Deterministic model-cycle, promotion, rollback, and freshness lifecycle operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from market_regime_engine.commands.contracts import (
    OperatorAction,
    OperatorRequest,
    OperatorResult,
    OperatorService,
)
from market_regime_engine.commands.errors import OperatorCommandError
from market_regime_engine.mlflow_support.ports import RegistryPort

RECOMMENDED_MODEL_CYCLE_DAYS = 7
SOURCE_STALE_WARN_DAYS = 4
SOURCE_STALE_FAIL_DAYS = 7
MODEL_STALE_WARN_DAYS = 14
MODEL_STALE_FAIL_DAYS = 35
REGISTERED_MODEL_NAME = "regime-xetra"
CHALLENGER_ALIAS = "challenger"
CHAMPION_ALIAS = "champion"


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    evaluation_id: str
    source_build_id: str
    statistical_champion_candidate_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "evaluation_id",
            "source_build_id",
            "statistical_champion_candidate_id",
        ):
            value = getattr(self, field_name)
            if not value or value.strip() != value:
                raise ValueError(f"{field_name} must be a non-empty trimmed string")


@dataclass(frozen=True, slots=True)
class FinalRefitOutcome:
    production_package: str

    def __post_init__(self) -> None:
        if not self.production_package or self.production_package.strip() != self.production_package:
            raise ValueError("production_package must be a non-empty trimmed string")


@dataclass(frozen=True, slots=True)
class OOSPublicationOutcome:
    oos_build_id: str

    def __post_init__(self) -> None:
        if not self.oos_build_id or self.oos_build_id.strip() != self.oos_build_id:
            raise ValueError("oos_build_id must be a non-empty trimmed string")


@dataclass(frozen=True, slots=True)
class RegistrationOutcome:
    exact_version: str

    def __post_init__(self) -> None:
        if not self.exact_version or self.exact_version.strip() != self.exact_version:
            raise ValueError("exact_version must be a non-empty trimmed string")


@dataclass(frozen=True, slots=True)
class LifecycleStatus:
    current_source_build_id: str
    completed_source_build_id: str | None
    champion_version: str | None
    challenger_version: str | None

    def __post_init__(self) -> None:
        if not self.current_source_build_id or self.current_source_build_id.strip() != (
            self.current_source_build_id
        ):
            raise ValueError("current_source_build_id must be a non-empty trimmed string")
        for value in (
            self.completed_source_build_id,
            self.champion_version,
            self.challenger_version,
        ):
            if value is not None and (not value or value.strip() != value):
                raise ValueError("optional lifecycle status identities must be trimmed when present")

    @property
    def source_changed(self) -> bool:
        return self.current_source_build_id != self.completed_source_build_id


@dataclass(frozen=True, slots=True)
class ModelCycleOutcome:
    changed: bool
    source_build_id: str
    evaluation_id: str | None = None
    statistical_champion_candidate_id: str | None = None
    production_package: str | None = None
    oos_build_id: str | None = None
    challenger_version: str | None = None

    def __post_init__(self) -> None:
        if not self.source_build_id or self.source_build_id.strip() != self.source_build_id:
            raise ValueError("source_build_id must be a non-empty trimmed string")
        evidence = (
            self.evaluation_id,
            self.statistical_champion_candidate_id,
            self.production_package,
            self.oos_build_id,
            self.challenger_version,
        )
        if self.changed and any(value is None for value in evidence):
            raise ValueError("changed model cycle requires complete lifecycle evidence")
        if not self.changed and any(value is not None for value in evidence):
            raise ValueError("unchanged model cycle must be a deterministic evidence-free no-op")


class LifecycleBackend(Protocol):
    """Engine-specific operations kept behind the lifecycle orchestration boundary."""

    def status(self, profile_id: str) -> LifecycleStatus: ...

    def evaluate(self, profile_id: str, source_build_id: str) -> EvaluationOutcome: ...

    def final_refit(self, profile_id: str, evaluation_id: str) -> FinalRefitOutcome: ...

    def publish_oos(self, profile_id: str, evaluation_id: str) -> OOSPublicationOutcome: ...

    def register_challenger(
        self,
        profile_id: str,
        production_package: str,
        oos_build_id: str,
    ) -> RegistrationOutcome: ...


class LifecycleRegistry(Protocol):
    def resolve_alias(self, model_name: str, alias: str) -> object: ...

    def compare_and_swap_alias(
        self,
        *,
        model_name: str,
        alias: str,
        expected_current_version: str | None,
        new_version: str,
        reason: str,
    ) -> bool: ...


class ModelLifecycleOperations:
    """Own alias mutations and deterministic changed-source model cycles."""

    def __init__(self, backend: LifecycleBackend, registry: RegistryPort) -> None:
        self._backend = backend
        self._registry = registry

    @staticmethod
    def _require_xetra(profile_id: str) -> None:
        if profile_id != "xetra":
            raise ValueError("model lifecycle currently supports exactly the xetra profile")

    @staticmethod
    def _require_reason(reason: str) -> None:
        if not reason or reason.strip() != reason:
            raise ValueError("alias mutation reason must be a non-empty trimmed string")

    def run_model_cycle(self, profile_id: str = "xetra") -> ModelCycleOutcome:
        """Evaluate a new source build exactly once; never promote champion automatically."""

        self._require_xetra(profile_id)
        status = self._backend.status(profile_id)
        if not status.source_changed:
            return ModelCycleOutcome(changed=False, source_build_id=status.current_source_build_id)

        evaluation = self._backend.evaluate(profile_id, status.current_source_build_id)
        if evaluation.source_build_id != status.current_source_build_id:
            raise ValueError("evaluation source build differs from the cycle-pinned source build")
        refit = self._backend.final_refit(profile_id, evaluation.evaluation_id)
        publication = self._backend.publish_oos(profile_id, evaluation.evaluation_id)
        registration = self._backend.register_challenger(
            profile_id,
            refit.production_package,
            publication.oos_build_id,
        )
        return ModelCycleOutcome(
            changed=True,
            source_build_id=status.current_source_build_id,
            evaluation_id=evaluation.evaluation_id,
            statistical_champion_candidate_id=evaluation.statistical_champion_candidate_id,
            production_package=refit.production_package,
            oos_build_id=publication.oos_build_id,
            challenger_version=registration.exact_version,
        )

    def promote(
        self,
        *,
        expected_current_version: str | None,
        new_version: str,
        reason: str,
    ) -> bool:
        self._require_reason(reason)
        if not new_version:
            raise ValueError("promotion target version cannot be empty")
        return self._registry.compare_and_swap_alias(
            model_name=REGISTERED_MODEL_NAME,
            alias=CHAMPION_ALIAS,
            expected_current_version=expected_current_version,
            new_version=new_version,
            reason=reason,
        )

    def rollback(
        self,
        *,
        expected_current_version: str,
        target_version: str,
        reason: str,
    ) -> bool:
        self._require_reason(reason)
        if not expected_current_version or not target_version:
            raise ValueError("rollback requires expected-current and target versions")
        return self._registry.compare_and_swap_alias(
            model_name=REGISTERED_MODEL_NAME,
            alias=CHAMPION_ALIAS,
            expected_current_version=expected_current_version,
            new_version=target_version,
            reason=reason,
        )


class LifecycleOperatorService(OperatorService):
    """Translate the PR-031 command contract into lifecycle backend operations."""

    def __init__(self, backend: LifecycleBackend) -> None:
        self._backend = backend

    @staticmethod
    def _required(request: OperatorRequest, name: str) -> str:
        value = request.parameter(name)
        if value is None:
            raise OperatorCommandError("missing_parameter", f"required parameter is missing: {name}")
        return value

    def execute(self, request: OperatorRequest) -> OperatorResult:
        if request.profile_id != "xetra":
            raise OperatorCommandError("unknown_profile", f"unsupported profile: {request.profile_id}")
        if request.action is OperatorAction.STATUS:
            status = self._backend.status(request.profile_id)
            fields = [
                ("current_source_build_id", status.current_source_build_id),
                ("source_changed", "true" if status.source_changed else "false"),
            ]
            for key, value in (
                ("challenger_version", status.challenger_version),
                ("champion_version", status.champion_version),
                ("completed_source_build_id", status.completed_source_build_id),
            ):
                if value is not None:
                    fields.append((key, value))
            return OperatorResult(request.action, request.profile_id, tuple(sorted(fields)))

        if request.action is OperatorAction.EVALUATE:
            status = self._backend.status(request.profile_id)
            outcome = self._backend.evaluate(request.profile_id, status.current_source_build_id)
            return OperatorResult(
                request.action,
                request.profile_id,
                (
                    ("evaluation_id", outcome.evaluation_id),
                    ("source_build_id", outcome.source_build_id),
                    (
                        "statistical_champion_candidate_id",
                        outcome.statistical_champion_candidate_id,
                    ),
                ),
            )

        evaluation_id = self._required(request, "evaluation_id") if request.action in {
            OperatorAction.FINAL_REFIT,
            OperatorAction.PUBLISH_OOS,
        } else None
        if request.action is OperatorAction.FINAL_REFIT:
            if evaluation_id is None:
                raise AssertionError("evaluation_id required above")
            refit = self._backend.final_refit(request.profile_id, evaluation_id)
            return OperatorResult(
                request.action,
                request.profile_id,
                (("production_package", refit.production_package),),
            )
        if request.action is OperatorAction.PUBLISH_OOS:
            if evaluation_id is None:
                raise AssertionError("evaluation_id required above")
            publication = self._backend.publish_oos(request.profile_id, evaluation_id)
            return OperatorResult(
                request.action,
                request.profile_id,
                (("oos_build_id", publication.oos_build_id),),
            )
        if request.action is OperatorAction.REGISTER:
            package = self._required(request, "production_package")
            oos_build_id = self._required(request, "oos_build_id")
            registration = self._backend.register_challenger(
                request.profile_id,
                package,
                oos_build_id,
            )
            return OperatorResult(
                request.action,
                request.profile_id,
                (("exact_version", registration.exact_version),),
            )
        raise AssertionError(f"unsupported operator action: {request.action}")


_backend_factory: callable | None = None  # type: ignore[valid-type]


def configure_operator_backend_factory(factory: object) -> None:
    """Configure process composition without importing model/source work at module import time."""

    global _backend_factory
    if not callable(factory):
        raise TypeError("operator backend factory must be callable")
    _backend_factory = factory  # type: ignore[assignment]


def build_operator_service() -> OperatorService:
    """Build the CLI service lazily after deployment composition installs a backend factory."""

    if _backend_factory is None:
        raise OperatorCommandError(
            "runtime_not_configured",
            "lifecycle backend factory has not been configured by deployment composition",
        )
    backend = _backend_factory()
    return LifecycleOperatorService(backend)  # type: ignore[arg-type]
