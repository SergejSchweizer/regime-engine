"""Deterministic model-cycle, promotion, rollback, and lifecycle CLI operations."""

from __future__ import annotations

from collections.abc import Callable
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


def _trimmed(value: str, field_name: str) -> str:
    if not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a non-empty trimmed string")
    return value


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    evaluation_id: str
    source_build_id: str
    statistical_champion_candidate_id: str

    def __post_init__(self) -> None:
        _trimmed(self.evaluation_id, "evaluation_id")
        _trimmed(self.source_build_id, "source_build_id")
        _trimmed(self.statistical_champion_candidate_id, "statistical_champion_candidate_id")


@dataclass(frozen=True, slots=True)
class FinalRefitOutcome:
    production_package: str

    def __post_init__(self) -> None:
        _trimmed(self.production_package, "production_package")


@dataclass(frozen=True, slots=True)
class OOSPublicationOutcome:
    oos_build_id: str

    def __post_init__(self) -> None:
        _trimmed(self.oos_build_id, "oos_build_id")


@dataclass(frozen=True, slots=True)
class RegistrationOutcome:
    exact_version: str

    def __post_init__(self) -> None:
        _trimmed(self.exact_version, "exact_version")


@dataclass(frozen=True, slots=True)
class LifecycleStatus:
    current_source_build_id: str
    completed_source_build_id: str | None
    champion_version: str | None
    challenger_version: str | None

    def __post_init__(self) -> None:
        _trimmed(self.current_source_build_id, "current_source_build_id")
        for field_name in (
            "completed_source_build_id",
            "champion_version",
            "challenger_version",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _trimmed(value, field_name)

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
        _trimmed(self.source_build_id, "source_build_id")
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
            raise ValueError("unchanged model cycle must be an evidence-free no-op")
        for value in evidence:
            if value is not None:
                _trimmed(value, "model-cycle evidence")


class LifecycleBackend(Protocol):
    """Engine-specific lifecycle primitives kept behind the orchestration boundary."""

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


class ModelLifecycleOperations:
    """Own changed-source model cycles and explicit champion alias mutations."""

    def __init__(self, backend: LifecycleBackend, registry: RegistryPort) -> None:
        self._backend = backend
        self._registry = registry

    @staticmethod
    def _require_xetra(profile_id: str) -> None:
        if profile_id != "xetra":
            raise ValueError("model lifecycle currently supports exactly the xetra profile")

    def run_model_cycle(self, profile_id: str = "xetra") -> ModelCycleOutcome:
        """Run evaluate -> select -> final refit -> OOS publish -> challenger register once."""

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
        _trimmed(new_version, "promotion target version")
        _trimmed(reason, "alias mutation reason")
        if expected_current_version is not None:
            _trimmed(expected_current_version, "expected current version")
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
        _trimmed(expected_current_version, "expected current version")
        _trimmed(target_version, "rollback target version")
        _trimmed(reason, "alias mutation reason")
        return self._registry.compare_and_swap_alias(
            model_name=REGISTERED_MODEL_NAME,
            alias=CHAMPION_ALIAS,
            expected_current_version=expected_current_version,
            new_version=target_version,
            reason=reason,
        )


class LifecycleOperatorService(OperatorService):
    """Translate the PR-031 operator command contract into lifecycle primitives."""

    def __init__(self, backend: LifecycleBackend) -> None:
        self._backend = backend

    @staticmethod
    def _required(request: OperatorRequest, name: str) -> str:
        value = request.parameter(name)
        if value is None:
            raise OperatorCommandError(
                "missing_parameter",
                f"required parameter is missing: {name}",
            )
        return value

    def execute(self, request: OperatorRequest) -> OperatorResult:
        if request.profile_id != "xetra":
            raise OperatorCommandError(
                "unknown_profile",
                f"unsupported profile: {request.profile_id}",
            )
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
            if outcome.source_build_id != status.current_source_build_id:
                raise OperatorCommandError(
                    "source_build_changed",
                    "evaluation source build differs from the status-pinned source build",
                )
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

        if request.action is OperatorAction.FINAL_REFIT:
            evaluation_id = self._required(request, "evaluation_id")
            refit = self._backend.final_refit(request.profile_id, evaluation_id)
            return OperatorResult(
                request.action,
                request.profile_id,
                (("production_package", refit.production_package),),
            )
        if request.action is OperatorAction.PUBLISH_OOS:
            evaluation_id = self._required(request, "evaluation_id")
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


OperatorBackendFactory = Callable[[], LifecycleBackend]
_backend_factory: OperatorBackendFactory | None = None


def configure_operator_backend_factory(factory: OperatorBackendFactory | None) -> None:
    """Install or clear the lazy backend factory used by the installed CLI."""

    global _backend_factory
    _backend_factory = factory


def build_operator_service() -> OperatorService:
    """Build the CLI service lazily after deployment composition installs a backend factory."""

    if _backend_factory is None:
        raise OperatorCommandError(
            "runtime_not_configured",
            "lifecycle backend factory has not been configured by deployment composition",
        )
    return LifecycleOperatorService(_backend_factory())
