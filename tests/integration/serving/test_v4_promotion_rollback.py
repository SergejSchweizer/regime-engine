from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from market_regime_engine.commands.lifecycle import ModelLifecycleOperations
from market_regime_engine.contracts import (
    DATA_TIME_SEMANTICS,
    LatestInvocation,
    ReplayInvocation,
    SourceLineage,
)
from market_regime_engine.features.ports import FeatureRow, FeatureSnapshot
from market_regime_engine.mlflow_support.model_package import (
    load_production_package,
    save_production_package,
)
from market_regime_engine.mlflow_support.registry import MlflowModelRegistry
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.production_artifact import ProductionModelArtifact
from market_regime_engine.preprocessing.scaling import StandardScalerArtifact
from market_regime_engine.serving.latest_handler import LatestHandler
from market_regime_engine.serving.model_resolver import ModelResolver
from market_regime_engine.serving.replay_admission import ReplayAdmission
from market_regime_engine.serving.replay_handler import ReplayHandler
from market_regime_engine.serving.replay_limits import ReplayLimits

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def artifact(build: str) -> ProductionModelArtifact:
    features = ("f0",)
    return ProductionModelArtifact(
        profile_id="xetra",
        profile_config_version=4,
        registered_model="regime-xetra",
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        source_build_id=build,
        source_data_sha256=(build.encode("utf-8").hex() * 64)[:64],
        source_schema_version=1,
        source_feature_version=1,
        data_time_semantics=DATA_TIME_SEMANTICS,
        feature_selection_definition_hash="b" * 64,
        feature_selection_execution_hash="c" * 64,
        evaluation_plan_hash="d" * 64,
        validation_evaluation_cutoff=BASE + timedelta(days=1),
        deployment_selection_cutoff=BASE + timedelta(days=2),
        validation_evidence_hash="e" * 64,
        source_catalog_hash="f" * 64,
        state_identity_scope="model_version_local",
        feature_order=features,
        scaler=StandardScalerArtifact(features, (0.0,), (1.0,), (1.0,)),
        hmm=GaussianHMMArtifact(
            state_count=2,
            feature_order=features,
            start_probabilities=(0.6, 0.4),
            transition_matrix=((0.9, 0.1), (0.1, 0.9)),
            means=((-1.0,), (1.0,)),
            full_covariances=(((1.0,),), ((1.5,),)),
        ),
        winning_seed=11,
        inference_origin_timestamp=BASE,
        trained_through_timestamp=BASE + timedelta(days=2),
        terminal_filtered_probabilities=(0.7, 0.3),
        retained_observation_count=504,
        skipped_incomplete_observation_count=0,
    )


class Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


class FeatureSource:
    def __init__(self, snapshot: FeatureSnapshot) -> None:
        self.snapshot = snapshot

    def read(self, request: object) -> FeatureSnapshot:
        del request
        return self.snapshot


class NoopBackend:
    """Alias-only lifecycle operations do not need an evaluation backend."""


def source_snapshot() -> FeatureSnapshot:
    maximum = BASE + timedelta(days=6)
    lineage = SourceLineage(
        source_dataset="regime_loader.regime_features_daily",
        source_build_id="serving-build",
        data_sha256="a" * 64,
        schema_version=1,
        feature_version=1,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=maximum,
        data_time_semantics=DATA_TIME_SEMANTICS,
        row_count=7,
        min_timestamp=BASE,
        max_timestamp=maximum,
    )
    return FeatureSnapshot(
        lineage=lineage,
        feature_names=("f0",),
        rows=tuple(
            FeatureRow(BASE + timedelta(days=index), (float(index) / 10.0,)) for index in range(7)
        ),
        skipped_incomplete_row_count=0,
    )


@pytest.mark.integration
def test_file_backed_v4_promotion_and_rollback_are_serving_safe(tmp_path: Path) -> None:
    database_uri = f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}"
    client = MlflowClient(tracking_uri=database_uri, registry_uri=database_uri)
    registry = MlflowModelRegistry(client)
    model_a = artifact("v4-build-a")
    model_b = artifact("v4-build-b")
    package_a = save_production_package(model_a, tmp_path / "package-a")
    package_b = save_production_package(model_b, tmp_path / "package-b")
    registered_a = registry.register_production_model(
        model_a,
        package_a,
        package_source_uri="runs:/v4-a/production-package",
    )
    registered_b = registry.register_production_model(
        model_b,
        package_b,
        package_source_uri="runs:/v4-b/production-package",
    )
    assert (registered_a.exact_version, registered_b.exact_version) == ("1", "2")

    assert registry.compare_and_swap_alias(
        model_name="regime-xetra",
        alias="champion",
        expected_current_version=None,
        new_version="1",
        reason="seed legacy-compatible v4 champion",
    )
    assert registry.compare_and_swap_alias(
        model_name="regime-xetra",
        alias="challenger",
        expected_current_version=None,
        new_version="2",
        reason="register audited v4 challenger",
    )

    package_paths = {
        "runs:/v4-a/production-package": package_a,
        "runs:/v4-b/production-package": package_b,
    }

    def exact_package_loader(uri: str) -> ProductionModelArtifact:
        return load_production_package(package_paths[uri])

    clock = Clock()
    resolver = ModelResolver(
        registry,
        alias_ttl_seconds=1.0,
        package_loader=exact_package_loader,
        clock=clock,
    )
    source = FeatureSource(source_snapshot())
    latest = LatestHandler(resolver, source)  # type: ignore[arg-type]
    replay = ReplayHandler(
        resolver,
        source,  # type: ignore[arg-type]
        ReplayLimits(),
        ReplayAdmission(ReplayLimits()),
    )
    operations = ModelLifecycleOperations(NoopBackend(), registry)  # type: ignore[arg-type]

    assert operations.promote(
        expected_current_version="1",
        new_version="2",
        reason="explicit v4 challenger promotion",
    )
    clock.value = 102.0
    latest_response = latest.handle(
        request_id="promotion-latest",
        profile_id="xetra",
        invocation=LatestInvocation(),
        request_time_utc=BASE + timedelta(days=6),
    )
    replay_response = replay.handle(
        request_id="promotion-replay",
        profile_id="xetra",
        invocation=ReplayInvocation(BASE + timedelta(days=4), BASE + timedelta(days=6)),
        request_time_utc=BASE + timedelta(days=6),
    )
    for response in (latest_response, replay_response):
        assert response.model.model_version == "2"
        assert response.model.profile_config_version == 4
        assert response.model.model_alias == "champion"
        assert response.model.feature_order == ("f0",)
        assert response.predictions[0].state_ids == ("state_0", "state_1")

    assert operations.rollback(
        expected_current_version="2",
        target_version="1",
        reason="explicit v4 rollback to prior champion",
    )
    clock.value = 104.0
    rolled_back = latest.handle(
        request_id="rollback-latest",
        profile_id="xetra",
        invocation=LatestInvocation(),
        request_time_utc=BASE + timedelta(days=6),
    )
    assert rolled_back.model.model_version == "1"
    assert rolled_back.model.profile_config_version == 4
    assert rolled_back.model.model_alias == "champion"

    before_failed_cas = registry.resolve_alias("regime-xetra", "champion").exact_version
    assert not operations.rollback(
        expected_current_version="stale-version",
        target_version="2",
        reason="stale operator rollback must not mutate alias",
    )
    after_failed_cas = registry.resolve_alias("regime-xetra", "champion").exact_version
    assert before_failed_cas == after_failed_cas == "1"
