from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

from market_regime_engine.contracts.core import K_CHAMPION_ALIASES
from market_regime_engine.evaluations.k_champion_contract import (
    KChampionPromotionCandidate,
    KChampionPromotionEvidence,
    KChampionSelection,
    feature_order_hash,
    rank_k_slot_candidates,
)
from market_regime_engine.mlflow_support.k_champion_contract import (
    build_promotion_instruction,
)
from market_regime_engine.mlflow_support.registry import (
    K_SLOT_ALIASES,
    MlflowModelRegistry,
)


@dataclass(frozen=True)
class _Version:
    version: str
    source: str
    tags: dict[str, str]


class _LocalRegistryClient:
    """Hermetic MLflow registry double with immutable model-version records."""

    def __init__(self) -> None:
        self.models: set[str] = set()
        self.versions: dict[tuple[str, str], _Version] = {}
        self.aliases: dict[tuple[str, str], str] = {}
        self.model_tags: dict[tuple[str, str], str] = {}
        self.create_count = 0

    def _missing(self) -> MlflowException:
        return MlflowException("missing", error_code=RESOURCE_DOES_NOT_EXIST)

    def get_registered_model(self, name: str) -> object:
        if name not in self.models:
            raise self._missing()
        return object()

    def create_registered_model(self, name: str) -> object:
        self.models.add(name)
        return object()

    def create_model_version(
        self,
        *,
        name: str,
        source: str,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> _Version:
        del description
        self.create_count += 1
        version = _Version(
            version=str(self.create_count),
            source=source,
            tags=dict(tags or {}),
        )
        self.versions[(name, version.version)] = version
        return version

    def search_model_versions(self, filter_string: str) -> tuple[_Version, ...]:
        assert filter_string == "name='regime-xetra'"
        return tuple(self.versions.values())

    def get_model_version(self, name: str, version: str) -> _Version:
        try:
            return self.versions[(name, version)]
        except KeyError as exc:
            raise self._missing() from exc

    def get_model_version_by_alias(self, name: str, alias: str) -> _Version:
        try:
            version = self.aliases[(name, alias)]
        except KeyError as exc:
            raise self._missing() from exc
        return self.get_model_version(name, version)

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[(name, alias)] = version

    def set_registered_model_tag(self, name: str, key: str, value: str) -> None:
        self.model_tags[(name, key)] = value


def _selection(
    *,
    slot_id: str = "k2",
    feature_order: tuple[str, ...] = ("f0", "f1"),
    artifact_byte: str = "a",
    candidate_identity: str = "gaussian_hmm_k2_full",
) -> KChampionSelection:
    state_count = int(slot_id.removeprefix("k"))
    cutoff = datetime(2026, 8, 20, tzinfo=UTC)
    return KChampionSelection(
        slot_id=slot_id,
        state_count=state_count,
        model_family="gaussian_hmm",
        candidate_identity=candidate_identity,
        feature_order=feature_order,
        feature_order_hash=feature_order_hash(feature_order),
        source_snapshot_id="snapshot-qa",
        profile_id="xetra",
        profile_config_version=4,
        policy_id="k_champion_portfolio",
        policy_version="k_champion_portfolio.v1",
        validation_cutoff=cutoff,
        deployment_cutoff=datetime(2026, 8, 21, tzinfo=UTC),
        comparison_domain_id="k_slot_promotion.v1",
        promotion_score_version="k_slot_promotion.v1",
        reference_teacher_id=f"teacher-{state_count}",
        artifact_hash=artifact_byte * 64,
    )


def _evidence(
    selection: KChampionSelection,
    *,
    mean_nmi: float | None = 0.8,
    rejection_reasons: tuple[str, ...] = (),
    valid_fold_count: int = 4,
    latest_complete_fold_valid: bool = True,
) -> KChampionPromotionEvidence:
    return KChampionPromotionEvidence(
        slot_id=selection.slot_id,
        source_snapshot_id=selection.source_snapshot_id,
        profile_id=selection.profile_id,
        profile_config_version=selection.profile_config_version,
        policy_version=selection.policy_version,
        validation_cutoff=selection.validation_cutoff,
        reference_teacher_id=selection.reference_teacher_id,
        feature_order_hash=selection.feature_order_hash,
        planned_fold_count=4,
        valid_fold_count=valid_fold_count,
        latest_complete_fold_valid=latest_complete_fold_valid,
        mean_soft_regime_nmi=mean_nmi,
        worst_fold_soft_regime_nmi=mean_nmi,
        common_support=0.95 if mean_nmi is not None else None,
        stability=0.9 if mean_nmi is not None else None,
        rejection_reasons=rejection_reasons,
    )


def _register(
    registry: MlflowModelRegistry,
    selection: KChampionSelection,
    *,
    source: str = "runs:/qa/package",
) -> str:
    return registry.register_k_slot_package(
        selection,
        package_source_uri=source,
        artifact_hash=selection.artifact_hash,
    ).exact_version


def test_k_slot_registry_uses_exactly_four_aliases() -> None:
    expected = {"champion-k2", "champion-k3", "champion-k4", "champion-k5"}

    assert set(K_CHAMPION_ALIASES) == expected
    assert expected == K_SLOT_ALIASES
    assert len(K_CHAMPION_ALIASES) == 4


def test_registration_is_immutable_and_idempotent_when_search_is_supported() -> None:
    client = _LocalRegistryClient()
    registry = MlflowModelRegistry(client)
    selection = _selection()

    first = registry.register_k_slot_package(
        selection,
        package_source_uri="runs:/qa/first-package",
        artifact_hash=selection.artifact_hash,
    )
    second = registry.register_k_slot_package(
        selection,
        package_source_uri="runs:/qa/replayed-package",
        artifact_hash=selection.artifact_hash,
    )

    assert first == second
    assert client.create_count == 1
    stored = client.versions[("regime-xetra", first.exact_version)]
    assert stored.source == "runs:/qa/first-package"
    assert stored.tags["regime_engine.slot_id"] == "k2"
    assert stored.tags["regime_engine.alias"] == "champion-k2"
    assert stored.tags["regime_engine.state_count"] == "2"
    assert stored.tags["regime_engine.idempotency_key"] == selection.idempotency_key


def test_slot_k_mismatch_is_rejected_and_stale_cas_is_a_no_op() -> None:
    client = _LocalRegistryClient()
    registry = MlflowModelRegistry(client)
    k2_version = _register(registry, _selection())

    with pytest.raises(ValueError, match="champion-k3 requires a matching K=3"):
        registry.compare_and_swap_alias(
            model_name="regime-xetra",
            alias="champion-k3",
            expected_current_version=None,
            new_version=k2_version,
            reason="invalid cross-slot move",
        )

    assert registry.compare_and_swap_alias(
        model_name="regime-xetra",
        alias="champion-k2",
        expected_current_version=None,
        new_version=k2_version,
        reason="initial slot move",
    )
    stale = registry.compare_and_swap_alias_with_audit(
        model_name="regime-xetra",
        alias="champion-k2",
        expected_current_version="stale-version",
        new_version=k2_version,
        reason="stale operator view",
    )

    assert stale.changed is False
    assert stale.observed_current_version == k2_version
    assert client.aliases[("regime-xetra", "champion-k2")] == k2_version


def test_successful_cas_and_k_promotion_leave_legacy_champion_untouched() -> None:
    client = _LocalRegistryClient()
    registry = MlflowModelRegistry(client)
    first_selection = _selection(feature_order=("f0", "f1"), artifact_byte="a")
    second_selection = _selection(
        feature_order=("f0", "f1", "pca0"),
        artifact_byte="b",
        candidate_identity="gaussian_hmm_k2_pca",
    )
    first_version = _register(registry, first_selection, source="runs:/qa/first")
    second_version = _register(registry, second_selection, source="runs:/qa/second")

    assert registry.compare_and_swap_alias(
        model_name="regime-xetra",
        alias="champion-k2",
        expected_current_version=None,
        new_version=first_version,
        reason="initial slot move",
    )
    assert registry.compare_and_swap_alias(
        model_name="regime-xetra",
        alias="champion",
        expected_current_version=None,
        new_version=first_version,
        reason="preserve legacy baseline",
    )
    decision = rank_k_slot_candidates(
        (
            KChampionPromotionCandidate(first_selection, _evidence(first_selection, mean_nmi=0.80)),
            KChampionPromotionCandidate(
                second_selection, _evidence(second_selection, mean_nmi=0.90)
            ),
        )
    )
    instruction = build_promotion_instruction(
        decision,
        exact_model_version=second_version,
        expected_current_version=first_version,
        reason="validated K=2 replacement",
    )
    audit = registry.apply_k_slot_promotion(instruction)

    assert audit.changed is True
    assert client.aliases[("regime-xetra", "champion-k2")] == second_version
    assert client.aliases[("regime-xetra", "champion")] == first_version


def test_ineligible_slot_cannot_create_a_promotion_instruction() -> None:
    selection = _selection()
    decision = rank_k_slot_candidates(
        (
            KChampionPromotionCandidate(
                selection,
                _evidence(
                    selection,
                    mean_nmi=None,
                    valid_fold_count=2,
                    latest_complete_fold_valid=False,
                    rejection_reasons=("insufficient valid folds",),
                ),
            ),
        )
    )

    assert decision.winner is None
    with pytest.raises(ValueError, match="ineligible K slot"):
        build_promotion_instruction(decision, exact_model_version="1")
