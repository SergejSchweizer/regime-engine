"""MLflow model-registry service for immutable final-refit production packages."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.tracking import MlflowClient

from market_regime_engine.mlflow_support.model_package import load_production_package
from market_regime_engine.mlflow_support.ports import ResolvedModelVersion
from market_regime_engine.models.production_artifact import ProductionModelArtifact

REGISTERED_MODEL_NAME = "regime-xetra"
ALLOWED_ALIASES = frozenset({"challenger", "champion"})


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
            raise ValueError("only challenger/champion aliases are permitted")
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
        raise ValueError("only challenger/champion aliases are permitted")


def _package_uri(package_directory: str | Path) -> str:
    package_path = Path(package_directory).resolve()
    if not package_path.is_dir():
        raise ValueError("production package directory does not exist")
    return package_path.as_uri()


class MlflowModelRegistry:
    """Concrete registry boundary with production-package validation and audited CAS."""

    def __init__(self, client: _RegistryClient | None = None) -> None:
        self._client = client if client is not None else cast(_RegistryClient, MlflowClient())

    def register_production_model(
        self,
        artifact: ProductionModelArtifact,
        package_directory: str | Path,
        *,
        description: str | None = None,
    ) -> RegisteredProductionModel:
        if type(artifact) is not ProductionModelArtifact:
            raise TypeError("only PR-063 ProductionModelArtifact objects can be registered")
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

        source = _package_uri(package_path)
        version = self._client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=source,
            description=description,
            tags={
                "regime_engine.package_schema": "RegimeEngineProductionModel.v1",
                "regime_engine.profile_id": artifact.profile_id,
                "regime_engine.profile_config_version": str(artifact.profile_config_version),
                "regime_engine.candidate_id": artifact.candidate_id,
                "regime_engine.source_build_id": artifact.source_build_id,
                "regime_engine.source_data_sha256": artifact.source_data_sha256,
                "regime_engine.feature_selection_definition_hash": (
                    artifact.feature_selection_definition_hash
                ),
                "regime_engine.feature_selection_execution_hash": (
                    artifact.feature_selection_execution_hash
                ),
                "regime_engine.evaluation_plan_hash": artifact.evaluation_plan_hash,
                "regime_engine.trained_through_timestamp": (
                    artifact.trained_through_timestamp.isoformat().replace("+00:00", "Z")
                ),
            },
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
        _require_model_name(model_name)
        _require_alias(alias)
        if not new_version:
            raise ValueError("new alias version cannot be empty")
        if not reason or reason.strip() != reason:
            raise ValueError("alias mutation reason must be a non-empty trimmed string")

        target = self._client.get_model_version(model_name, new_version)
        if str(target.version) != new_version:
            raise ValueError("registry returned a mismatched target model version")
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
