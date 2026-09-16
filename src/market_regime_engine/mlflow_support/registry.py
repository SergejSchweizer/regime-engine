"""MLflow model-registry service for immutable final-refit production packages."""

from __future__ import annotations

import fcntl
import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Protocol, cast

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.tracking import MlflowClient

from market_regime_engine.contracts.core import K_CHAMPION_ALIASES
from market_regime_engine.evaluations.k_champion_contract import KChampionSelection
from market_regime_engine.mlflow_support.k_champion_contract import (
    KChampionPromotionInstruction,
)
from market_regime_engine.mlflow_support.model_package import load_production_package
from market_regime_engine.mlflow_support.ports import ResolvedModelVersion
from market_regime_engine.mlflow_support.settings import MLflowSettings
from market_regime_engine.models.production_artifact import ProductionModelArtifact

REGISTERED_MODEL_NAME = "regime-xetra"
K_SLOT_ALIASES = frozenset(K_CHAMPION_ALIASES)
ALLOWED_ALIASES = frozenset({"challenger", "champion", *K_SLOT_ALIASES})
_K_SLOT_BY_ALIAS = {alias: int(alias.rsplit("-k", 1)[1]) for alias in K_SLOT_ALIASES}
_REGISTRY_THREAD_LOCK = threading.RLock()


@contextmanager
def _registry_mutation_lock() -> Iterator[None]:
    """Serialize registry side effects across local worker processes."""

    with _REGISTRY_THREAD_LOCK:
        configured = os.environ.get(
            "REGIME_MLFLOW_REGISTRY_LOCK_FILE",
            os.path.join(
                os.environ.get("XDG_RUNTIME_DIR", "/tmp"),
                "regime-engine-mlflow-registry.lock",
            ),
        )
        lock_path = Path(configured).expanduser().resolve()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class _ModelVersion(Protocol):
    version: str
    source: str


class _RegistryClient(Protocol):
    def get_registered_model(self, name: str) -> object: ...

    def create_registered_model(self, name: str) -> object: ...

    def create_model_version(
        self,
        *,
        name: str,
        source: str,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> _ModelVersion: ...

    def get_model_version(self, name: str, version: str) -> _ModelVersion: ...

    def get_model_version_by_alias(self, name: str, alias: str) -> _ModelVersion: ...

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None: ...

    def set_registered_model_tag(self, name: str, key: str, value: str) -> None: ...


@dataclass(frozen=True, slots=True)
class RegisteredProductionModel:
    model_name: str
    exact_version: str
    package_uri: str

    def __post_init__(self) -> None:
        if self.model_name != REGISTERED_MODEL_NAME:
            raise ValueError("registered production model must be exactly regime-xetra")
        if not self.exact_version or not self.package_uri:
            raise ValueError("registered model version/package URI cannot be empty")


@dataclass(frozen=True, slots=True)
class RegisteredKSlotModel:
    model_name: str
    slot_id: str
    exact_version: str
    package_uri: str
    idempotency_key: str

    def __post_init__(self) -> None:
        if self.model_name != REGISTERED_MODEL_NAME:
            raise ValueError("registered K-slot model must be exactly regime-xetra")
        if self.slot_id not in {"k2", "k3", "k4", "k5"}:
            raise ValueError("registered K-slot model has an invalid slot")
        if not self.exact_version or not self.package_uri or not self.idempotency_key:
            raise ValueError("registered K-slot identity/package fields cannot be empty")


@dataclass(frozen=True, slots=True)
class AliasMutationAudit:
    model_name: str
    alias: str
    expected_current_version: str | None
    observed_current_version: str | None
    new_version: str
    reason: str
    changed: bool
    observed_at_utc: datetime

    def __post_init__(self) -> None:
        if self.model_name != REGISTERED_MODEL_NAME:
            raise ValueError("alias audit model name must be exactly regime-xetra")
        if self.alias not in ALLOWED_ALIASES:
            raise ValueError(
                "only challenger/champion or champion-k2..champion-k5 aliases are permitted"
            )
        if not self.new_version:
            raise ValueError("alias mutation requires a non-empty target version")
        if not self.reason or self.reason.strip() != self.reason:
            raise ValueError("alias mutation requires a non-empty trimmed reason")
        if self.observed_at_utc.tzinfo is None or self.observed_at_utc.utcoffset() != UTC.utcoffset(
            self.observed_at_utc
        ):
            raise ValueError("alias audit timestamp must be timezone-aware UTC")
        if self.changed != (self.observed_current_version == self.expected_current_version):
            raise ValueError("alias audit changed flag must match the CAS comparison")

    def canonical_json(self) -> str:
        payload = asdict(self)
        payload["observed_at_utc"] = self.observed_at_utc.isoformat().replace("+00:00", "Z")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _is_missing(exc: MlflowException) -> bool:
    code = getattr(exc, "error_code", None)
    if code in {RESOURCE_DOES_NOT_EXIST, "RESOURCE_DOES_NOT_EXIST"}:
        return True
    message = str(exc)
    return code in {INVALID_PARAMETER_VALUE, "INVALID_PARAMETER_VALUE"} and (
        message.startswith("Registered model alias ") and message.endswith(" not found.")
    )


def _require_model_name(model_name: str) -> None:
    if model_name != REGISTERED_MODEL_NAME:
        raise ValueError("Xetra registry model name must be exactly regime-xetra")


def _require_alias(alias: str) -> None:
    if alias not in ALLOWED_ALIASES:
        raise ValueError(
            "only challenger/champion or champion-k2..champion-k5 aliases are permitted"
        )


def _require_slot_target(alias: str, target: object) -> None:
    """Ensure a K-slot alias can only point to the matching K model version."""

    expected_state_count = _K_SLOT_BY_ALIAS.get(alias)
    if expected_state_count is None:
        return
    tags = getattr(target, "tags", {})
    observed = tags.get("regime_engine.state_count") if isinstance(tags, dict) else None
    if observed is None:
        raise ValueError(f"{alias} target is missing regime_engine.state_count")
    try:
        state_count = int(observed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{alias} target has an invalid state count") from exc
    if state_count != expected_state_count:
        raise ValueError(f"{alias} requires a matching K={expected_state_count} model version")


class MlflowModelRegistry:
    """Concrete registry boundary with production-package validation and audited CAS."""

    def __init__(self, client: _RegistryClient | None = None) -> None:
        if client is not None:
            self._client = client
            return
        settings = MLflowSettings.from_environment()
        self._client = cast(
            _RegistryClient,
            MlflowClient(
                tracking_uri=settings.tracking_uri,
                registry_uri=settings.registry_uri,
            ),
        )

    def register_production_model(
        self,
        artifact: ProductionModelArtifact,
        package_directory: str | Path,
        *,
        description: str | None = None,
        package_source_uri: str | None = None,
    ) -> RegisteredProductionModel:
        if type(artifact) is not ProductionModelArtifact:
            raise TypeError("only v4 ProductionModelArtifact objects can be registered")
        _require_model_name(artifact.registered_model)
        package_path = Path(package_directory).resolve()
        packaged = load_production_package(package_path)
        if packaged != artifact:
            raise ValueError(
                "production package payload differs from supplied final-refit artifact"
            )

        try:
            self._client.get_registered_model(REGISTERED_MODEL_NAME)
        except MlflowException as exc:
            if not _is_missing(exc):
                raise
            self._client.create_registered_model(REGISTERED_MODEL_NAME)

        if package_source_uri is None or not package_source_uri.startswith("runs:/"):
            raise ValueError(
                "production registration requires a remote MLflow runs:/ package source URI"
            )
        source = package_source_uri
        package_digest = sha256((package_path / "production_model.json").read_bytes()).hexdigest()
        tags = {
            "regime_engine.package_schema": "RegimeEngineProductionModel.v4",
            "regime_engine.package_sha256": package_digest,
            "regime_engine.profile_id": artifact.profile_id,
            "regime_engine.profile_config_version": str(artifact.profile_config_version),
            "regime_engine.candidate_id": artifact.candidate_id,
            "regime_engine.state_count": str(artifact.state_count),
            "regime_engine.source_build_id": artifact.source_build_id,
            "regime_engine.source_data_sha256": artifact.source_data_sha256,
            "regime_engine.source_catalog_hash": artifact.source_catalog_hash,
            "regime_engine.validation_evidence_hash": artifact.validation_evidence_hash,
            "regime_engine.feature_selection_definition_hash": (
                artifact.feature_selection_definition_hash
            ),
            "regime_engine.feature_selection_execution_hash": (
                artifact.feature_selection_execution_hash
            ),
            "regime_engine.evaluation_plan_hash": artifact.evaluation_plan_hash,
            "regime_engine.validation_evaluation_cutoff": (
                artifact.validation_evaluation_cutoff.isoformat().replace("+00:00", "Z")
            ),
            "regime_engine.deployment_selection_cutoff": (
                artifact.deployment_selection_cutoff.isoformat().replace("+00:00", "Z")
            ),
            "regime_engine.state_identity_scope": artifact.state_identity_scope,
            "regime_engine.trained_through_timestamp": (
                artifact.trained_through_timestamp.isoformat().replace("+00:00", "Z")
            ),
        }
        search = getattr(self._client, "search_model_versions", None)
        if callable(search):
            for existing in search("name='regime-xetra'"):
                existing_tags = getattr(existing, "tags", {})
                if all(existing_tags.get(key) == value for key, value in tags.items()):
                    return RegisteredProductionModel(
                        model_name=REGISTERED_MODEL_NAME,
                        exact_version=str(existing.version),
                        package_uri=str(existing.source),
                    )
        version = self._client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=source,
            description=description,
            tags=tags,
        )
        return RegisteredProductionModel(
            model_name=REGISTERED_MODEL_NAME,
            exact_version=str(version.version),
            package_uri=source,
        )

    def resolve_alias(self, model_name: str, alias: str) -> ResolvedModelVersion:
        _require_model_name(model_name)
        _require_alias(alias)
        version = self._client.get_model_version_by_alias(model_name, alias)
        return ResolvedModelVersion(
            model_name=model_name,
            alias=alias,
            exact_version=str(version.version),
            resolved_at_utc=datetime.now(UTC),
        )

    def register_k_slot_package(
        self,
        selection: KChampionSelection,
        *,
        package_source_uri: str,
        artifact_hash: str,
        description: str | None = None,
    ) -> RegisteredKSlotModel:
        with _registry_mutation_lock():
            return self._register_k_slot_package_unlocked(
                selection,
                package_source_uri=package_source_uri,
                artifact_hash=artifact_hash,
                description=description,
            )

    def _register_k_slot_package_unlocked(
        self,
        selection: KChampionSelection,
        *,
        package_source_uri: str,
        artifact_hash: str,
        description: str | None,
    ) -> RegisteredKSlotModel:
        """Register one immutable K-slot package by its remote ``runs:/`` URI."""

        _require_model_name(REGISTERED_MODEL_NAME)
        if not package_source_uri.startswith("runs:/"):
            raise ValueError("K-slot registration requires a remote MLflow runs:/ URI")
        if len(artifact_hash) != 64 or artifact_hash != artifact_hash.lower():
            raise ValueError("K-slot artifact_hash must be a lowercase SHA-256")
        if artifact_hash != selection.artifact_hash:
            raise ValueError("K-slot artifact hash differs from the immutable selection record")
        try:
            self._client.get_registered_model(REGISTERED_MODEL_NAME)
        except MlflowException as exc:
            if not _is_missing(exc):
                raise
            self._client.create_registered_model(REGISTERED_MODEL_NAME)
        tags = {
            "regime_engine.slot_id": selection.slot_id,
            "regime_engine.alias": selection.alias,
            "regime_engine.state_count": str(selection.state_count),
            "regime_engine.model_family": selection.model_family,
            "regime_engine.candidate_identity": selection.candidate_identity,
            "regime_engine.policy_id": selection.policy_id,
            "regime_engine.policy_version": selection.policy_version,
            "regime_engine.profile_id": selection.profile_id,
            "regime_engine.profile_config_version": str(selection.profile_config_version),
            "regime_engine.source_snapshot_id": selection.source_snapshot_id,
            "regime_engine.feature_order_sha256": selection.feature_order_hash,
            "regime_engine.comparison_domain_id": selection.comparison_domain_id,
            "regime_engine.promotion_score_version": selection.promotion_score_version,
            "regime_engine.reference_teacher_id": selection.reference_teacher_id,
            "regime_engine.validation_cutoff": selection.validation_cutoff.isoformat(),
            "regime_engine.deployment_cutoff": selection.deployment_cutoff.isoformat(),
            "regime_engine.selection_sha256": selection.selection_hash,
            "regime_engine.artifact_sha256": artifact_hash,
            "regime_engine.idempotency_key": selection.idempotency_key,
        }
        search = getattr(self._client, "search_model_versions", None)
        if callable(search):
            for existing in search("name='regime-xetra'"):
                existing_tags = getattr(existing, "tags", {})
                if all(existing_tags.get(key) == value for key, value in tags.items()):
                    return RegisteredKSlotModel(
                        REGISTERED_MODEL_NAME,
                        selection.slot_id,
                        str(existing.version),
                        str(existing.source),
                        selection.idempotency_key,
                    )
        version = self._client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=package_source_uri,
            description=description,
            tags=tags,
        )
        return RegisteredKSlotModel(
            REGISTERED_MODEL_NAME,
            selection.slot_id,
            str(version.version),
            package_source_uri,
            selection.idempotency_key,
        )

    def apply_k_slot_promotion(
        self,
        instruction: KChampionPromotionInstruction,
    ) -> AliasMutationAudit:
        """Apply an already validated K-slot instruction using audited CAS."""

        return self.compare_and_swap_alias_with_audit(
            model_name=instruction.model_name,
            alias=instruction.alias,
            expected_current_version=instruction.expected_current_version,
            new_version=instruction.exact_model_version,
            reason=instruction.reason,
        )

    def get_model_package_uri(self, model_name: str, exact_version: str) -> str:
        _require_model_name(model_name)
        if not exact_version:
            raise ValueError("exact model version cannot be empty")
        version = self._client.get_model_version(model_name, exact_version)
        if not version.source:
            raise ValueError("registered model version has no package source URI")
        return str(version.source)

    def _current_alias_version(self, alias: str) -> str | None:
        try:
            version = self._client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
        except MlflowException as exc:
            if _is_missing(exc):
                return None
            raise
        return str(version.version)

    def compare_and_swap_alias_with_audit(
        self,
        *,
        model_name: str,
        alias: str,
        expected_current_version: str | None,
        new_version: str,
        reason: str,
    ) -> AliasMutationAudit:
        with _registry_mutation_lock():
            return self._compare_and_swap_alias_with_audit_unlocked(
                model_name=model_name,
                alias=alias,
                expected_current_version=expected_current_version,
                new_version=new_version,
                reason=reason,
            )

    def _compare_and_swap_alias_with_audit_unlocked(
        self,
        *,
        model_name: str,
        alias: str,
        expected_current_version: str | None,
        new_version: str,
        reason: str,
    ) -> AliasMutationAudit:
        _require_model_name(model_name)
        _require_alias(alias)
        if not new_version:
            raise ValueError("new alias version cannot be empty")
        if not reason or reason.strip() != reason:
            raise ValueError("alias mutation reason must be a non-empty trimmed string")

        target = self._client.get_model_version(model_name, new_version)
        if str(target.version) != new_version:
            raise ValueError("registry returned a mismatched target model version")
        _require_slot_target(alias, target)
        observed = self._current_alias_version(alias)
        changed = observed == expected_current_version
        audit = AliasMutationAudit(
            model_name=model_name,
            alias=alias,
            expected_current_version=expected_current_version,
            observed_current_version=observed,
            new_version=new_version,
            reason=reason,
            changed=changed,
            observed_at_utc=datetime.now(UTC),
        )
        if changed:
            self._client.set_registered_model_alias(model_name, alias, new_version)
        key = f"regime_engine.alias_audit.{audit.observed_at_utc.timestamp():.6f}"
        self._client.set_registered_model_tag(model_name, key, audit.canonical_json())
        return audit

    def rollback_k_slot(
        self,
        *,
        model_name: str,
        alias: str,
        expected_current_version: str,
        rollback_version: str,
        reason: str = "K-slot rollback",
    ) -> AliasMutationAudit:
        """Restore one retained K-slot version through the same audited CAS."""

        if alias not in K_SLOT_ALIASES:
            raise ValueError("rollback is permitted only for champion-k2..champion-k5")
        if not expected_current_version:
            raise ValueError("rollback requires the currently observed alias version")
        return self.compare_and_swap_alias_with_audit(
            model_name=model_name,
            alias=alias,
            expected_current_version=expected_current_version,
            new_version=rollback_version,
            reason=reason,
        )

    def compare_and_swap_alias(
        self,
        *,
        model_name: str,
        alias: str,
        expected_current_version: str | None,
        new_version: str,
        reason: str,
    ) -> bool:
        return self.compare_and_swap_alias_with_audit(
            model_name=model_name,
            alias=alias,
            expected_current_version=expected_current_version,
            new_version=new_version,
            reason=reason,
        ).changed
