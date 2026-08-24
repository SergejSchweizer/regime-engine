from __future__ import annotations

from datetime import UTC, datetime

import pytest

from market_regime_engine.commands.contracts import OperatorAction, OperatorRequest
from market_regime_engine.commands.errors import OperatorCommandError
from market_regime_engine.commands.lifecycle import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    MODEL_STALE_FAIL_DAYS,
    MODEL_STALE_WARN_DAYS,
    RECOMMENDED_MODEL_CYCLE_DAYS,
    REGISTERED_MODEL_NAME,
    SOURCE_STALE_FAIL_DAYS,
    SOURCE_STALE_WARN_DAYS,
    EvaluationOutcome,
    FinalRefitOutcome,
    LifecycleOperatorService,
    LifecycleStatus,
    ModelCycleOutcome,
    ModelLifecycleOperations,
    OOSPublicationOutcome,
    RegistrationOutcome,
    build_operator_service,
    configure_operator_backend_factory,
)
from market_regime_engine.mlflow_support.ports import ResolvedModelVersion


class FakeBackend:
    def __init__(self, status: LifecycleStatus | None = None) -> None:
        self.status_value = status or LifecycleStatus("source-2", "source-1", "5", "6")
        self.events: list[tuple[object, ...]] = []
        self.evaluation = EvaluationOutcome("eval-1", "source-2", "gaussian_hmm_k3_full")
        self.refit = FinalRefitOutcome("/packages/final")
        self.publication = OOSPublicationOutcome("oos-1")
        self.registration = RegistrationOutcome("7")

    def status(self, profile_id: str) -> LifecycleStatus:
        self.events.append(("status", profile_id))
        return self.status_value

    def evaluate(self, profile_id: str, source_build_id: str) -> EvaluationOutcome:
        self.events.append(("evaluate", profile_id, source_build_id))
        return self.evaluation

    def final_refit(self, profile_id: str, evaluation_id: str) -> FinalRefitOutcome:
        self.events.append(("final_refit", profile_id, evaluation_id))
        return self.refit

    def publish_oos(self, profile_id: str, evaluation_id: str) -> OOSPublicationOutcome:
        self.events.append(("publish_oos", profile_id, evaluation_id))
        return self.publication

    def register_challenger(
        self,
        profile_id: str,
        production_package: str,
        oos_build_id: str,
    ) -> RegistrationOutcome:
        self.events.append(("register", profile_id, production_package, oos_build_id))
        return self.registration


class FakeRegistry:
    def __init__(self, *, cas_result: bool = True) -> None:
        self.cas_result = cas_result
        self.calls: list[dict[str, object]] = []

    def resolve_alias(self, model_name: str, alias: str) -> ResolvedModelVersion:
        return ResolvedModelVersion(model_name, alias, "1", datetime(2026, 1, 1, tzinfo=UTC))

    def get_model_package_uri(self, model_name: str, exact_version: str) -> str:
        return f"file:///{model_name}/{exact_version}"

    def compare_and_swap_alias(
        self,
        *,
        model_name: str,
        alias: str,
        expected_current_version: str | None,
        new_version: str,
        reason: str,
    ) -> bool:
        self.calls.append(
            {
                "model_name": model_name,
                "alias": alias,
                "expected_current_version": expected_current_version,
                "new_version": new_version,
                "reason": reason,
            }
        )
        return self.cas_result


def request(
    action: OperatorAction,
    *parameters: tuple[str, str],
    profile_id: str = "xetra",
) -> OperatorRequest:
    return OperatorRequest(action, profile_id, tuple(sorted(parameters)))


def test_lifecycle_constants_are_exact_contract_values() -> None:
    assert RECOMMENDED_MODEL_CYCLE_DAYS == 7
    assert SOURCE_STALE_WARN_DAYS == 4
    assert SOURCE_STALE_FAIL_DAYS == 7
    assert MODEL_STALE_WARN_DAYS == 14
    assert MODEL_STALE_FAIL_DAYS == 35
    assert REGISTERED_MODEL_NAME == "regime-xetra"
    assert CHALLENGER_ALIAS == "challenger"
    assert CHAMPION_ALIAS == "champion"


def test_outcome_contracts_validate_trimmed_identifiers() -> None:
    assert EvaluationOutcome("e", "s", "c").evaluation_id == "e"
    assert FinalRefitOutcome("package").production_package == "package"
    assert OOSPublicationOutcome("oos").oos_build_id == "oos"
    assert RegistrationOutcome("8").exact_version == "8"
    for constructor, args in (
        (EvaluationOutcome, ("", "s", "c")),
        (EvaluationOutcome, ("e", " s", "c")),
        (EvaluationOutcome, ("e", "s", "c ")),
        (FinalRefitOutcome, ("",)),
        (OOSPublicationOutcome, (" oos",)),
        (RegistrationOutcome, ("8 ",)),
    ):
        with pytest.raises(ValueError, match="trimmed"):
            constructor(*args)  # type: ignore[misc]


def test_status_detects_changed_source_and_validates_optional_fields() -> None:
    changed = LifecycleStatus("source-2", "source-1", "5", "6")
    same = LifecycleStatus("source-2", "source-2", None, None)
    assert changed.source_changed
    assert not same.source_changed
    with pytest.raises(ValueError, match="current_source_build_id"):
        LifecycleStatus("", None, None, None)
    with pytest.raises(ValueError, match="champion_version"):
        LifecycleStatus("source", None, " bad", None)


def test_model_cycle_contract_requires_all_or_no_evidence() -> None:
    noop = ModelCycleOutcome(False, "source-1")
    assert not noop.changed
    complete = ModelCycleOutcome(
        True,
        "source-2",
        "eval-1",
        "gaussian_hmm_k3_full",
        "/package",
        "oos-1",
        "7",
    )
    assert complete.challenger_version == "7"
    with pytest.raises(ValueError, match="complete lifecycle evidence"):
        ModelCycleOutcome(True, "source-2")
    with pytest.raises(ValueError, match="evidence-free"):
        ModelCycleOutcome(False, "source-2", evaluation_id="eval")
    with pytest.raises(ValueError, match="trimmed"):
        ModelCycleOutcome(True, "source", "eval", "candidate", "pkg", "oos", " 7")


def test_unchanged_source_is_deterministic_noop() -> None:
    backend = FakeBackend(LifecycleStatus("source-1", "source-1", "5", "6"))
    operations = ModelLifecycleOperations(backend, FakeRegistry())  # type: ignore[arg-type]
    result = operations.run_model_cycle()
    assert result == ModelCycleOutcome(False, "source-1")
    assert backend.events == [("status", "xetra")]


def test_changed_source_runs_exact_statistical_lifecycle_without_promotion() -> None:
    backend = FakeBackend()
    registry = FakeRegistry()
    operations = ModelLifecycleOperations(backend, registry)  # type: ignore[arg-type]
    result = operations.run_model_cycle("xetra")
    assert result == ModelCycleOutcome(
        True,
        "source-2",
        "eval-1",
        "gaussian_hmm_k3_full",
        "/packages/final",
        "oos-1",
        "7",
    )
    assert backend.events == [
        ("status", "xetra"),
        ("evaluate", "xetra", "source-2"),
        ("final_refit", "xetra", "eval-1"),
        ("publish_oos", "xetra", "eval-1"),
        ("register", "xetra", "/packages/final", "oos-1"),
    ]
    assert registry.calls == []


def test_model_cycle_rejects_profile_and_source_build_race() -> None:
    operations = ModelLifecycleOperations(FakeBackend(), FakeRegistry())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly the xetra"):
        operations.run_model_cycle("crypto")

    backend = FakeBackend()
    backend.evaluation = EvaluationOutcome("eval", "other-source", "gaussian_hmm_k2_full")
    operations = ModelLifecycleOperations(backend, FakeRegistry())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cycle-pinned"):
        operations.run_model_cycle()
    assert backend.events == [
        ("status", "xetra"),
        ("evaluate", "xetra", "source-2"),
    ]


def test_promote_and_rollback_use_champion_alias_cas_with_reason() -> None:
    registry = FakeRegistry(cas_result=True)
    operations = ModelLifecycleOperations(FakeBackend(), registry)  # type: ignore[arg-type]
    assert operations.promote(
        expected_current_version="5",
        new_version="7",
        reason="validated challenger",
    )
    assert operations.rollback(
        expected_current_version="7",
        target_version="5",
        reason="operator rollback",
    )
    assert registry.calls == [
        {
            "model_name": "regime-xetra",
            "alias": "champion",
            "expected_current_version": "5",
            "new_version": "7",
            "reason": "validated challenger",
        },
        {
            "model_name": "regime-xetra",
            "alias": "champion",
            "expected_current_version": "7",
            "new_version": "5",
            "reason": "operator rollback",
        },
    ]

    false_registry = FakeRegistry(cas_result=False)
    false_ops = ModelLifecycleOperations(FakeBackend(), false_registry)  # type: ignore[arg-type]
    assert not false_ops.promote(
        expected_current_version=None,
        new_version="1",
        reason="initial explicit promotion",
    )


def test_alias_mutations_reject_empty_or_untrimmed_fields() -> None:
    operations = ModelLifecycleOperations(FakeBackend(), FakeRegistry())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="promotion target"):
        operations.promote(expected_current_version=None, new_version="", reason="r")
    with pytest.raises(ValueError, match="expected current"):
        operations.promote(expected_current_version=" 1", new_version="2", reason="r")
    with pytest.raises(ValueError, match="alias mutation reason"):
        operations.promote(expected_current_version="1", new_version="2", reason="")
    with pytest.raises(ValueError, match="expected current"):
        operations.rollback(expected_current_version="", target_version="1", reason="r")
    with pytest.raises(ValueError, match="rollback target"):
        operations.rollback(expected_current_version="1", target_version=" 2", reason="r")
    with pytest.raises(ValueError, match="alias mutation reason"):
        operations.rollback(expected_current_version="1", target_version="2", reason=" r")


def test_operator_service_status_and_evaluate_are_source_pinned() -> None:
    backend = FakeBackend()
    service = LifecycleOperatorService(backend)
    status = service.execute(request(OperatorAction.STATUS))
    assert status.fields == (
        ("challenger_version", "6"),
        ("champion_version", "5"),
        ("completed_source_build_id", "source-1"),
        ("current_source_build_id", "source-2"),
        ("source_changed", "true"),
    )
    evaluated = service.execute(request(OperatorAction.EVALUATE))
    assert evaluated.fields == (
        ("evaluation_id", "eval-1"),
        ("source_build_id", "source-2"),
        ("statistical_champion_candidate_id", "gaussian_hmm_k3_full"),
    )

    backend.evaluation = EvaluationOutcome("eval-2", "race", "gaussian_hmm_k2_full")
    with pytest.raises(OperatorCommandError) as exc:
        service.execute(request(OperatorAction.EVALUATE))
    assert exc.value.code == "source_build_changed"


def test_operator_service_dispatches_refit_publish_and_register() -> None:
    backend = FakeBackend()
    service = LifecycleOperatorService(backend)
    refit = service.execute(request(OperatorAction.FINAL_REFIT, ("evaluation_id", "eval-1")))
    assert refit.fields == (("production_package", "/packages/final"),)
    publication = service.execute(
        request(OperatorAction.PUBLISH_OOS, ("evaluation_id", "eval-1"))
    )
    assert publication.fields == (("oos_build_id", "oos-1"),)
    registration = service.execute(
        request(
            OperatorAction.REGISTER,
            ("oos_build_id", "oos-1"),
            ("production_package", "/packages/final"),
        )
    )
    assert registration.fields == (("exact_version", "7"),)
    assert backend.events[-3:] == [
        ("final_refit", "xetra", "eval-1"),
        ("publish_oos", "xetra", "eval-1"),
        ("register", "xetra", "/packages/final", "oos-1"),
    ]


def test_operator_service_rejects_unknown_profile_and_missing_parameters() -> None:
    service = LifecycleOperatorService(FakeBackend())
    with pytest.raises(OperatorCommandError) as exc:
        service.execute(request(OperatorAction.STATUS, profile_id="crypto"))
    assert exc.value.code == "unknown_profile"

    for action, expected in (
        (OperatorAction.FINAL_REFIT, "evaluation_id"),
        (OperatorAction.PUBLISH_OOS, "evaluation_id"),
        (OperatorAction.REGISTER, "production_package"),
    ):
        with pytest.raises(OperatorCommandError) as exc:
            service.execute(request(action))
        assert exc.value.code == "missing_parameter"
        assert expected in exc.value.message

    with pytest.raises(OperatorCommandError, match="oos_build_id"):
        service.execute(
            request(OperatorAction.REGISTER, ("production_package", "/packages/final"))
        )


def test_lazy_operator_backend_factory_can_be_installed_and_cleared() -> None:
    configure_operator_backend_factory(None)
    with pytest.raises(OperatorCommandError) as exc:
        build_operator_service()
    assert exc.value.code == "runtime_not_configured"

    backend = FakeBackend()
    configure_operator_backend_factory(lambda: backend)
    service = build_operator_service()
    assert service.execute(request(OperatorAction.STATUS)).profile_id == "xetra"
    configure_operator_backend_factory(None)
