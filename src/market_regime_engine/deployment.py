"""Production composition for the deployed MLflow application."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import psycopg
import yaml
from mlflow.tracking import MlflowClient

from market_regime_engine.feature_selection.contracts import FeatureBlock, FeatureSelectionPolicy
from market_regime_engine.features.postgres_settings import FeaturePostgresSettings
from market_regime_engine.features.postgres_source import ConnectionLike, PostgresFeatureSource
from market_regime_engine.mlflow_app.dependencies import (
    ReadinessSnapshot,
    ServiceDependencies,
    configure_default_dependencies,
)
from market_regime_engine.mlflow_support.registry import MlflowModelRegistry
from market_regime_engine.predictions.store import PredictionStore
from market_regime_engine.serving.latest_handler import LatestHandler
from market_regime_engine.serving.model_resolver import ModelResolver
from market_regime_engine.serving.oos_handler import OOSPredictionHandler
from market_regime_engine.serving.profile_registry import ProfileModelTarget, ProfileRegistry
from market_regime_engine.serving.replay_admission import ReplayAdmission
from market_regime_engine.serving.replay_handler import ReplayHandler
from market_regime_engine.serving.replay_limits import ReplayLimits

_ROOT = Path("/opt/regime-engine")
_ARTIFACT_ROOT = Path("/mlflow/artifacts") / "regime-engine"


def _policy() -> FeatureSelectionPolicy:
    path = _ROOT / "configs/feature_selection/xetra_semantic_medoid_v2.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("production feature policy must be a mapping")
    blocks = tuple(
        FeatureBlock(str(item["block_id"]), tuple(str(value) for value in item["features"]))
        for item in raw["blocks"]
    )
    return FeatureSelectionPolicy(
        policy_id=str(raw["policy_id"]),
        blocks=blocks,
        within_block_method=str(raw["within_block_method"]),
        cross_block_method=str(raw["cross_block_method"]),
        minimum_feature_coverage=float(raw["minimum_feature_coverage"]),
        minimum_nonzero_variance=float(raw["minimum_nonzero_variance"]),
        minimum_block_complete_observations=int(raw["minimum_block_complete_observations"]),
        maximum_cross_block_abs_spearman=float(raw["maximum_cross_block_abs_spearman"]),
        numeric_tie_abs_tolerance=float(raw["numeric_tie_abs_tolerance"]),
    )


def compose_serving_dependencies() -> ServiceDependencies:
    """Compose read-only feature serving against the local MLflow registry."""

    settings = FeaturePostgresSettings.from_env(os.environ)
    policy = _policy()

    def connect() -> ConnectionLike:
        return cast(ConnectionLike, psycopg.connect(**cast(Any, settings.connection_kwargs())))

    source = PostgresFeatureSource(
        connect,
        policy.feature_universe,
    )
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    registry = MlflowModelRegistry(cast(Any, client))
    profiles = ProfileRegistry(
        (
            ProfileModelTarget(
                profile_id="xetra",
                profile_config_version=2,
                model_name="regime-xetra",
                production_alias="champion",
            ),
        )
    )
    resolver = ModelResolver(registry, profiles=profiles)
    limits = ReplayLimits.from_env(os.environ)
    return ServiceDependencies(
        latest_handler=LatestHandler(resolver, source),
        replay_handler=ReplayHandler(resolver, source, limits, ReplayAdmission(limits)),
        oos_handler=OOSPredictionHandler(PredictionStore(_ARTIFACT_ROOT / "oos")),
        readiness=lambda: ReadinessSnapshot("healthy", True),
    )


def configure_serving_defaults() -> None:
    """Install dependencies once per Gunicorn worker during Flask app construction."""

    configure_default_dependencies(compose_serving_dependencies())
