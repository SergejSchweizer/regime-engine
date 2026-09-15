from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

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
    assert stored.tags["regime_engine.selection_sha256"] == selection.selection_hash
    assert stored.tags["regime_engine.candidate_identity"] == selection.candidate_identity
    assert stored.tags["regime_engine.validation_cutoff"] == selection.validation_cutoff.isoformat()
    assert stored.tags["regime_engine.deployment_cutoff"] == selection.deployment_cutoff.isoformat()


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


class _ThreadSafeFileBackedRegistryClient(_LocalRegistryClient):
    """Process-local registry double with an atomic persisted state boundary.

    MLflow's public client methods used by ``MlflowModelRegistry`` are separate
    calls, so the double makes the version-creation operation idempotent at the
    registry boundary and serializes alias mutations.  This models the
    guarantees that the current adapter can actually rely on without adding a
    production-only synchronization API.
    """

    def __init__(self, state_path: Path) -> None:
        super().__init__()
        self._lock = RLock()
        self._state_path = state_path

    def _persist(self) -> None:
        payload = {
            "models": sorted(self.models),
            "versions": {
                f"{name}:{version}": {
                    "source": item.source,
                    "tags": item.tags,
                }
                for (name, version), item in sorted(self.versions.items())
            },
            "aliases": {
                f"{name}:{alias}": version
                for (name, alias), version in sorted(self.aliases.items())
            },
            "model_tags": {
                f"{name}:{key}": value for (name, key), value in sorted(self.model_tags.items())
            },
            "create_count": self.create_count,
        }
        self._state_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    def get_registered_model(self, name: str) -> object:
        with self._lock:
            return super().get_registered_model(name)

    def create_registered_model(self, name: str) -> object:
        with self._lock:
            result = super().create_registered_model(name)
            self._persist()
            return result

    def create_model_version(
        self,
        *,
        name: str,
        source: str,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> _Version:
        with self._lock:
            requested_tags = dict(tags or {})
            for existing in self.versions.values():
                if existing.tags == requested_tags:
                    return existing
            version = super().create_model_version(
                name=name,
                source=source,
                description=description,
                tags=requested_tags,
            )
            self._persist()
            return version

    def search_model_versions(self, filter_string: str) -> tuple[_Version, ...]:
        with self._lock:
            return super().search_model_versions(filter_string)

    def get_model_version(self, name: str, version: str) -> _Version:
        with self._lock:
            return super().get_model_version(name, version)

    def get_model_version_by_alias(self, name: str, alias: str) -> _Version:
        with self._lock:
            return super().get_model_version_by_alias(name, alias)

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        with self._lock:
            super().set_registered_model_alias(name, alias, version)
            self._persist()

    def set_registered_model_tag(self, name: str, key: str, value: str) -> None:
        with self._lock:
            super().set_registered_model_tag(name, key, value)
            self._persist()


@pytest.mark.parametrize("slot_id", ("k2", "k3", "k4", "k5"))
def test_all_four_slots_register_and_promote_only_to_their_matching_alias(
    slot_id: str,
) -> None:
    client = _LocalRegistryClient()
    registry = MlflowModelRegistry(client)
    selection = _selection(slot_id=slot_id, artifact_byte=slot_id[-1])
    version = _register(registry, selection)

    audit = registry.compare_and_swap_alias_with_audit(
        model_name="regime-xetra",
        alias=selection.alias,
        expected_current_version=None,
        new_version=version,
        reason=f"first promotion for {slot_id}",
    )

    assert audit.changed is True
    assert client.aliases["regime-xetra", selection.alias] == version
    assert client.versions["regime-xetra", version].tags["regime_engine.slot_id"] == slot_id


@pytest.mark.parametrize("slot_id", ("k2", "k3", "k4", "k5"))
def test_first_better_worse_and_tie_outcomes_are_deterministic(slot_id: str) -> None:
    client = _LocalRegistryClient()
    registry = MlflowModelRegistry(client)
    first_selection = _selection(slot_id=slot_id, feature_order=("f0", "f1"), artifact_byte="a")
    better_selection = _selection(
        slot_id=slot_id,
        feature_order=("f0", "f1", "pca0"),
        artifact_byte="b",
        candidate_identity="gaussian_hmm_k2_pca0",
    )
    worse_selection = _selection(
        slot_id=slot_id,
        feature_order=("f0", "f1", "pca1"),
        artifact_byte="c",
        candidate_identity="gaussian_hmm_k2_pca1",
    )
    tie_selection = _selection(
        slot_id=slot_id,
        feature_order=("f0", "f1", "pca2"),
        artifact_byte="d",
        candidate_identity="gaussian_hmm_k2_pca2",
    )
    first_version = _register(registry, first_selection)
    better_version = _register(registry, better_selection)
    worse_version = _register(registry, worse_selection)
    tie_version = _register(registry, tie_selection)

    first = registry.compare_and_swap_alias_with_audit(
        model_name="regime-xetra",
        alias=first_selection.alias,
        expected_current_version=None,
        new_version=first_version,
        reason="first candidate",
    )
    assert first.changed is True

    better_decision = rank_k_slot_candidates(
        (
            KChampionPromotionCandidate(first_selection, _evidence(first_selection, mean_nmi=0.80)),
            KChampionPromotionCandidate(
                better_selection,
                _evidence(better_selection, mean_nmi=0.90),
            ),
        )
    )
    assert better_decision.winner is not None
    assert better_decision.winner.selection == better_selection
    better = registry.apply_k_slot_promotion(
        build_promotion_instruction(
            better_decision,
            exact_model_version=better_version,
            expected_current_version=first_version,
            reason="better candidate",
        )
    )
    assert better.changed is True

    worse_decision = rank_k_slot_candidates(
        (
            KChampionPromotionCandidate(
                better_selection, _evidence(better_selection, mean_nmi=0.90)
            ),
            KChampionPromotionCandidate(worse_selection, _evidence(worse_selection, mean_nmi=0.70)),
        )
    )
    assert worse_decision.winner is not None
    assert worse_decision.winner.selection == better_selection
    assert client.aliases["regime-xetra", first_selection.alias] == better_version
    assert worse_version != better_version

    tie_decision = rank_k_slot_candidates(
        (
            KChampionPromotionCandidate(
                better_selection, _evidence(better_selection, mean_nmi=0.90)
            ),
            KChampionPromotionCandidate(tie_selection, _evidence(tie_selection, mean_nmi=0.90)),
        )
    )
    assert tie_decision.winner is not None
    expected_tie_winner = min(
        (better_selection, tie_selection),
        key=lambda item: (item.feature_order_hash, item.candidate_identity),
    )
    assert tie_decision.winner.selection == expected_tie_winner
    tie_version_for_winner = (
        better_version if expected_tie_winner == better_selection else tie_version
    )
    tie = registry.apply_k_slot_promotion(
        build_promotion_instruction(
            tie_decision,
            exact_model_version=tie_version_for_winner,
            expected_current_version=better_version,
            reason="deterministic tie winner",
        )
    )
    assert tie.changed is True
    assert client.aliases["regime-xetra", first_selection.alias] == tie_version_for_winner


@pytest.mark.parametrize("slot_id", ("k2", "k3", "k4", "k5"))
def test_stale_cas_and_cas_based_rollback_restore_an_earlier_version(slot_id: str) -> None:
    client = _LocalRegistryClient()
    registry = MlflowModelRegistry(client)
    original = _selection(slot_id=slot_id, artifact_byte="a")
    replacement = _selection(
        slot_id=slot_id,
        feature_order=("f0", "f1", "pca0"),
        artifact_byte="b",
        candidate_identity="gaussian_hmm_k2_pca0",
    )
    original_version = _register(registry, original)
    replacement_version = _register(registry, replacement)

    assert registry.compare_and_swap_alias(
        model_name="regime-xetra",
        alias=original.alias,
        expected_current_version=None,
        new_version=original_version,
        reason="establish original",
    )
    assert registry.compare_and_swap_alias(
        model_name="regime-xetra",
        alias=original.alias,
        expected_current_version=original_version,
        new_version=replacement_version,
        reason="promote replacement",
    )

    stale = registry.compare_and_swap_alias_with_audit(
        model_name="regime-xetra",
        alias=original.alias,
        expected_current_version=original_version,
        new_version=original_version,
        reason="stale rollback operator view",
    )
    assert stale.changed is False
    assert stale.observed_current_version == replacement_version
    assert client.aliases["regime-xetra", original.alias] == replacement_version

    other_slot = "k3" if slot_id != "k3" else "k2"
    other_selection = _selection(slot_id=other_slot, artifact_byte="c")
    other_version = _register(registry, other_selection)
    assert registry.compare_and_swap_alias(
        model_name="regime-xetra",
        alias=other_selection.alias,
        expected_current_version=None,
        new_version=other_version,
        reason="preserve another K slot",
    )
    assert registry.compare_and_swap_alias(
        model_name="regime-xetra",
        alias="champion",
        expected_current_version=None,
        new_version=other_version,
        reason="preserve legacy alias",
    )

    restored = registry.rollback_k_slot(
        model_name="regime-xetra",
        alias=original.alias,
        expected_current_version=replacement_version,
        rollback_version=original_version,
        reason="restore original through CAS",
    )
    assert restored.changed is True
    assert client.aliases["regime-xetra", original.alias] == original_version
    assert client.aliases["regime-xetra", other_selection.alias] == other_version
    assert client.aliases["regime-xetra", "champion"] == other_version


def test_incompatible_provenance_and_registry_targets_are_rejected() -> None:
    selection = _selection()
    mismatched_source = replace(_evidence(selection), source_snapshot_id="other-snapshot")
    with pytest.raises(ValueError, match="selection/evidence source snapshot mismatch"):
        KChampionPromotionCandidate(selection, mismatched_source)

    k3_selection = _selection(slot_id="k3", artifact_byte="b")
    with pytest.raises(ValueError, match="cross-K promotion comparison"):
        rank_k_slot_candidates(
            (
                KChampionPromotionCandidate(selection, _evidence(selection)),
                KChampionPromotionCandidate(k3_selection, _evidence(k3_selection)),
            )
        )

    client = _LocalRegistryClient()
    registry = MlflowModelRegistry(client)
    version = _register(registry, selection)
    with pytest.raises(ValueError, match="artifact hash differs"):
        registry.register_k_slot_package(
            selection,
            package_source_uri="runs:/qa/incompatible-artifact",
            artifact_hash="b" * 64,
        )
    client.versions["regime-xetra", version] = replace(
        client.versions["regime-xetra", version],
        tags={**client.versions["regime-xetra", version].tags, "regime_engine.state_count": "3"},
    )
    with pytest.raises(ValueError, match="champion-k2 requires a matching K=2"):
        registry.compare_and_swap_alias(
            model_name="regime-xetra",
            alias="champion-k2",
            expected_current_version=None,
            new_version=version,
            reason="incompatible target provenance",
        )


@pytest.mark.parametrize("slot_id", ("k2", "k3", "k4", "k5"))
def test_concurrent_registration_and_promotion_are_idempotent_in_a_file_backed_double(
    tmp_path: Path, slot_id: str
) -> None:
    client = _ThreadSafeFileBackedRegistryClient(tmp_path / "registry-state.json")
    registry = MlflowModelRegistry(client)
    selection = _selection(slot_id=slot_id, artifact_byte=slot_id[-1])

    def register() -> str:
        return _register(registry, selection, source="runs:/qa/concurrent")

    with ThreadPoolExecutor(max_workers=8) as executor:
        versions = tuple(executor.map(lambda _: register(), range(16)))

    assert set(versions) == {versions[0]}
    assert client.create_count == 1

    def promote() -> object:
        return registry.compare_and_swap_alias_with_audit(
            model_name="regime-xetra",
            alias=selection.alias,
            expected_current_version=None,
            new_version=versions[0],
            reason="concurrent idempotent promotion",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        audits = tuple(executor.map(lambda _: promote(), range(16)))

    assert 1 <= sum(audit.changed for audit in audits) <= len(audits)
    assert all(audit.new_version == versions[0] for audit in audits)
    assert all(audit.observed_current_version in {None, versions[0]} for audit in audits)
    assert client.aliases["regime-xetra", selection.alias] == versions[0]
    persisted = json.loads((tmp_path / "registry-state.json").read_text(encoding="utf-8"))
    assert persisted["aliases"][f"regime-xetra:{selection.alias}"] == versions[0]
    assert persisted["create_count"] == 1
