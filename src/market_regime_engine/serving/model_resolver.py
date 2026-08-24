"""Profile-aware exact-version resolver layered over registry aliases and the model cache."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from urllib.parse import unquote, urlparse

from mlflow.artifacts import download_artifacts

from market_regime_engine.mlflow_support.model_package import load_production_package
from market_regime_engine.mlflow_support.ports import RegistryPort
from market_regime_engine.models.production_artifact import ProductionModelArtifact
from market_regime_engine.serving.model_cache import ModelCache, ModelLease
from market_regime_engine.serving.profile_registry import ProfileModelTarget, ProfileRegistry

DEFAULT_ALIAS_TTL_SECONDS = 30.0
PackageLoader = Callable[[str], ProductionModelArtifact]
Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class _AliasCacheEntry:
    exact_version: str
    expires_at: float


@dataclass(slots=True)
class ResolvedModelLease:
    target: ProfileModelTarget
    exact_version: str
    resolved_via_alias: str | None
    _lease: ModelLease

    @property
    def artifact(self) -> ProductionModelArtifact:
        return self._lease.artifact

    def release(self) -> None:
        self._lease.release()

    def __enter__(self) -> ResolvedModelLease:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.release()


def _load_mlflow_package(
    package_uri: str,
    *,
    tracking_uri: str | None = None,
) -> ProductionModelArtifact:
    parsed = urlparse(package_uri)
    if parsed.scheme == "file" and parsed.netloc in {"", "localhost"}:
        path = Path(unquote(parsed.path))
    elif parsed.scheme == "runs":
        path = Path(
            download_artifacts(
                artifact_uri=package_uri,
                tracking_uri=tracking_uri,
                registry_uri=tracking_uri,
            )
        )
    else:
        raise ValueError("production package URI must be a local file URI or MLflow run URI")
    return load_production_package(path)


def mlflow_package_loader(*, tracking_uri: str) -> PackageLoader:
    """Bind run-artifact downloads to the public MLflow HTTP endpoint.

    The MLflow server process changes its global tracking URI to its SQL backend.
    Custom serving routes must therefore not rely on that process-global value.
    """

    if not tracking_uri:
        raise ValueError("tracking URI must be non-empty")

    def load(package_uri: str) -> ProductionModelArtifact:
        return _load_mlflow_package(package_uri, tracking_uri=tracking_uri)

    return load


class ModelResolver:
    """Resolve profile->model versions without silently relabelling stale/invalid aliases."""

    def __init__(
        self,
        registry: RegistryPort,
        *,
        profiles: ProfileRegistry | None = None,
        cache: ModelCache | None = None,
        alias_ttl_seconds: float = DEFAULT_ALIAS_TTL_SECONDS,
        package_loader: PackageLoader = _load_mlflow_package,
        clock: Clock = monotonic,
    ) -> None:
        if alias_ttl_seconds <= 0.0:
            raise ValueError("alias TTL must be positive")
        self._registry = registry
        self._profiles = profiles or ProfileRegistry()
        self._cache = cache or ModelCache()
        self._alias_ttl_seconds = alias_ttl_seconds
        self._package_loader = package_loader
        self._clock = clock
        self._alias_cache: dict[tuple[str, str], _AliasCacheEntry] = {}

    def _resolve_exact_version(
        self,
        target: ProfileModelTarget,
        explicit_version: str | None,
    ) -> tuple[str, str | None]:
        if explicit_version is not None:
            if not explicit_version:
                raise ValueError("explicit model version cannot be empty")
            return explicit_version, None

        key = (target.model_name, target.production_alias)
        now = self._clock()
        cached = self._alias_cache.get(key)
        if cached is not None and now < cached.expires_at:
            return cached.exact_version, target.production_alias

        resolved = self._registry.resolve_alias(target.model_name, target.production_alias)
        if resolved.model_name != target.model_name or resolved.alias != target.production_alias:
            raise ValueError("registry alias resolution returned mismatched model identity")
        self._alias_cache[key] = _AliasCacheEntry(
            exact_version=resolved.exact_version,
            expires_at=now + self._alias_ttl_seconds,
        )
        return resolved.exact_version, target.production_alias

    def _load_and_validate(
        self,
        target: ProfileModelTarget,
        exact_version: str,
    ) -> ProductionModelArtifact:
        package_uri = self._registry.get_model_package_uri(target.model_name, exact_version)
        artifact = self._package_loader(package_uri)
        if type(artifact) is not ProductionModelArtifact:
            raise TypeError("production package loader returned an unsupported artifact type")
        if artifact.profile_id != target.profile_id:
            raise ValueError("production package profile differs from requested public profile")
        if artifact.profile_config_version != target.profile_config_version:
            raise ValueError(
                "production package profile configuration version differs from routing target"
            )
        if artifact.registered_model != target.model_name:
            raise ValueError("production package registered model differs from routing target")
        return artifact

    def resolve(
        self,
        profile_id: str,
        *,
        profile_config_version: int | None = None,
        exact_version: str | None = None,
    ) -> ResolvedModelLease:
        target = self._profiles.resolve(profile_id, profile_config_version)
        version, alias = self._resolve_exact_version(target, exact_version)
        cache_key = f"{target.model_name}:{version}"
        lease = self._cache.acquire(
            cache_key,
            lambda: self._load_and_validate(target, version),
        )
        return ResolvedModelLease(
            target=target,
            exact_version=version,
            resolved_via_alias=alias,
            _lease=lease,
        )
