from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

from market_regime_engine.mlflow_support.model_package import save_production_package
from market_regime_engine.mlflow_support.registry import MlflowModelRegistry
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.production_artifact import ProductionModelArtifact
from market_regime_engine.preprocessing.scaling import StandardScalerArtifact


@dataclass
class Version:
    version: str
    source: str


class FakeRegistryClient:
    def __init__(self) -> None:
        self.models: set[str] = set()
        self.versions: dict[tuple[str, str], Version] = {}
        self.aliases: dict[tuple[str, str], str] = {}
        self.tags: dict[tuple[str, str], str] = {}
        self.next_version = 1

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
    ) -> Version:
        del description, tags
        version = Version(str(self.next_version), source)
        self.next_version += 1
        self.versions[(name, version.version)] = version
        return version

    def get_model_version(self, name: str, version: str) -> Version:
        try:
            return self.versions[(name, version)]
        except KeyError as exc:
            raise self._missing() from exc

    def get_model_version_by_alias(self, name: str, alias: str) -> Version:
        try:
            version = self.aliases[(name, alias)]
        except KeyError as exc:
            raise self._missing() from exc
        return self.get_model_version(name, version)

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[(name, alias)] = version

    def set_registered_model_tag(self, name: str, key: str, value: str) -> None:
        self.tags[(name, key)] = value


def artifact() -> ProductionModelArtifact:
    features = ("f0", "f1")
    return ProductionModelArtifact(
        profile_id="xetra",
        profile_config_version=1,
        registered_model="regime-xetra",
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        source_build_id="build-1",
        source_data_sha256="d" * 64,
        source_schema_version=1,
        source_feature_version=1,
        data_time_semantics="current_vintage_observation_day",
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        evaluation_plan_hash="c" * 64,
        evaluation_cutoff=datetime(2026, 8, 20, tzinfo=UTC),
        feature_order=features,
        scaler=StandardScalerArtifact(
            feature_order=features,
            means=(0.0, 0.0),
            variances=(1.0, 1.5),
            scales=(1.0, 1.5**0.5),
        ),
        hmm=GaussianHMMArtifact(
            state_count=2,
            feature_order=features,
            start_probabilities=(0.5, 0.5),
            transition_matrix=((0.9, 0.1), (0.2, 0.8)),
            means=((-1.0, 0.5), (1.0, -0.5)),
            full_covariances=(
                ((1.0, 0.2), (0.2, 1.5)),
                ((1.2, -0.1), (-0.1, 1.0)),
            ),
        ),
        winning_seed=11,
        inference_origin_timestamp=datetime(2020, 1, 1, tzinfo=UTC),
        trained_through_timestamp=datetime(2026, 8, 19, tzinfo=UTC),
        terminal_filtered_probabilities=(0.45, 0.55),
        retained_observation_count=1500,
        skipped_incomplete_observation_count=2,
    )


def test_registers_only_matching_final_refit_package(tmp_path) -> None:
    client = FakeRegistryClient()
    registry = MlflowModelRegistry(client)
    package = save_production_package(artifact(), tmp_path / "package")
    registered = registry.register_production_model(artifact(), package)
    assert registered.model_name == "regime-xetra"
    assert registered.exact_version == "1"
    assert registry.get_model_package_uri("regime-xetra", "1") == package.resolve().as_uri()
    assert "regime-xetra" in client.models

    with pytest.raises(ValueError, match="differs"):
        registry.register_production_model(replace(artifact(), winning_seed=23), package)


def test_compare_and_swap_alias_is_fail_closed_and_audited(tmp_path) -> None:
    client = FakeRegistryClient()
    registry = MlflowModelRegistry(client)
    first_package = save_production_package(artifact(), tmp_path / "one")
    first = registry.register_production_model(artifact(), first_package)
    second_artifact = replace(artifact(), source_build_id="build-2")
    second_package = save_production_package(second_artifact, tmp_path / "two")
    second = registry.register_production_model(second_artifact, second_package)

    assert registry.compare_and_swap_alias(
        model_name="regime-xetra",
        alias="challenger",
        expected_current_version=None,
        new_version=first.exact_version,
        reason="initial challenger",
    )
    resolved = registry.resolve_alias("regime-xetra", "challenger")
    assert resolved.exact_version == first.exact_version

    mismatch = registry.compare_and_swap_alias_with_audit(
        model_name="regime-xetra",
        alias="challenger",
        expected_current_version="999",
        new_version=second.exact_version,
        reason="stale operator view",
    )
    assert mismatch.changed is False
    assert mismatch.observed_current_version == first.exact_version
    current = registry.resolve_alias("regime-xetra", "challenger")
    assert current.exact_version == first.exact_version

    changed = registry.compare_and_swap_alias_with_audit(
        model_name="regime-xetra",
        alias="challenger",
        expected_current_version=first.exact_version,
        new_version=second.exact_version,
        reason="validated replacement",
    )
    assert changed.changed is True
    current = registry.resolve_alias("regime-xetra", "challenger")
    assert current.exact_version == second.exact_version
    assert any("alias_audit" in key for _, key in client.tags)


def test_registry_rejects_unknown_model_alias_and_empty_reason() -> None:
    registry = MlflowModelRegistry(FakeRegistryClient())
    with pytest.raises(ValueError, match="regime-xetra"):
        registry.resolve_alias("other", "champion")
    with pytest.raises(ValueError, match="challenger/champion"):
        registry.resolve_alias("regime-xetra", "production")
    with pytest.raises(ValueError, match="reason"):
        registry.compare_and_swap_alias(
            model_name="regime-xetra",
            alias="champion",
            expected_current_version=None,
            new_version="1",
            reason="",
        )
