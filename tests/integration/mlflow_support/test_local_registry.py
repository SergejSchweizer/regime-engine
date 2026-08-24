from __future__ import annotations

from datetime import UTC, datetime

import pytest
from mlflow.tracking import MlflowClient

from market_regime_engine.mlflow_support.model_package import save_production_package
from market_regime_engine.mlflow_support.registry import MlflowModelRegistry
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.production_artifact import ProductionModelArtifact
from market_regime_engine.preprocessing.scaling import StandardScalerArtifact


def artifact() -> ProductionModelArtifact:
    features = ("f0",)
    return ProductionModelArtifact(
        profile_id="xetra",
        profile_config_version=1,
        registered_model="regime-xetra",
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        source_build_id="local-build",
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
            means=(0.0,),
            variances=(1.0,),
            scales=(1.0,),
        ),
        hmm=GaussianHMMArtifact(
            state_count=2,
            feature_order=features,
            start_probabilities=(0.5, 0.5),
            transition_matrix=((0.9, 0.1), (0.1, 0.9)),
            means=((-1.0,), (1.0,)),
            full_covariances=(((1.0,),), ((1.5,),)),
        ),
        winning_seed=11,
        inference_origin_timestamp=datetime(2020, 1, 1, tzinfo=UTC),
        trained_through_timestamp=datetime(2026, 8, 19, tzinfo=UTC),
        terminal_filtered_probabilities=(0.4, 0.6),
        retained_observation_count=1500,
        skipped_incomplete_observation_count=0,
    )


@pytest.mark.integration
def test_local_sqlite_mlflow_registry_package_and_alias_roundtrip(tmp_path) -> None:
    database_uri = f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}"
    client = MlflowClient(tracking_uri=database_uri, registry_uri=database_uri)
    registry = MlflowModelRegistry(client)
    package = save_production_package(artifact(), tmp_path / "package")

    version = registry.register_production_model(artifact(), package)
    assert version.exact_version == "1"
    assert registry.compare_and_swap_alias(
        model_name="regime-xetra",
        alias="champion",
        expected_current_version=None,
        new_version=version.exact_version,
        reason="local integration promotion",
    )
    resolved = registry.resolve_alias("regime-xetra", "champion")
    assert resolved.exact_version == version.exact_version
    assert (
        registry.get_model_package_uri("regime-xetra", version.exact_version)
        == package.resolve().as_uri()
    )
