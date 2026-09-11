"""Production composition for the deployed MLflow application."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import mlflow
import psycopg
from mlflow.tracking import MlflowClient

from market_regime_engine.features.postgres_settings import FeaturePostgresSettings
from market_regime_engine.features.postgres_source import ConnectionLike, PostgresFeatureSource
from market_regime_engine.mlflow_app.dependencies import (
    ReadinessSnapshot,
    ServiceDependencies,
    configure_default_dependencies,
)
from market_regime_engine.mlflow_support.registry import MlflowModelRegistry
from market_regime_engine.mlflow_support.settings import MLflowSettings
from market_regime_engine.predictions.store import PredictionStore
from market_regime_engine.serving.latest_handler import LatestHandler
from market_regime_engine.serving.model_resolver import ModelResolver, mlflow_package_loader
from market_regime_engine.serving.oos_handler import OOSPredictionHandler
from market_regime_engine.serving.profile_registry import ProfileModelTarget, ProfileRegistry
from market_regime_engine.serving.replay_admission import ReplayAdmission
from market_regime_engine.serving.replay_handler import ReplayHandler
from market_regime_engine.serving.replay_limits import ReplayLimits

_ROOT = Path(os.environ.get("REGIME_ENGINE_ROOT", Path(__file__).resolve().parents[2]))
_ARTIFACT_ROOT = (
    Path(os.environ.get("MLFLOW_ARTIFACT_ROOT", str(_ROOT / ".state/mlflow-artifacts")))
    / "regime-engine"
)


def compose_serving_dependencies() -> ServiceDependencies:
    """Compose read-only feature serving against the external MLflow registry."""

    settings = FeaturePostgresSettings.from_env(os.environ)

    def connect() -> ConnectionLike:
        return cast(ConnectionLike, psycopg.connect(**cast(Any, settings.connection_kwargs())))

    source = PostgresFeatureSource(connect)
    mlflow_settings = MLflowSettings.from_environment()
    tracking_uri = mlflow_settings.tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(mlflow_settings.registry_uri)
    client = MlflowClient(
        tracking_uri=mlflow_settings.tracking_uri,
        registry_uri=mlflow_settings.registry_uri,
    )
    registry = MlflowModelRegistry(cast(Any, client))
    profiles = ProfileRegistry(
        (
            ProfileModelTarget(
                profile_id="xetra",
                profile_config_version=4,
                model_name="regime-xetra",
                production_alias="champion",
            ),
        )
    )
    resolver = ModelResolver(
        registry,
        profiles=profiles,
        package_loader=mlflow_package_loader(tracking_uri=tracking_uri),
    )
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
