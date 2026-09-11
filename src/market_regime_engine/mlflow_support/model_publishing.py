"""Publish immutable HMM packages to the external NAS MLflow registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mlflow.tracking import MlflowClient

from market_regime_engine.mlflow_support.model_package import load_production_package
from market_regime_engine.mlflow_support.registry import (
    MlflowModelRegistry,
    RegisteredProductionModel,
)
from market_regime_engine.mlflow_support.settings import MLflowSettings
from market_regime_engine.models.production_artifact import ProductionModelArtifact


@dataclass(frozen=True, slots=True)
class PublishedProductionModel:
    """Remote MLflow identities for one uploaded final-refit HMM package."""

    tracking_uri: str
    run_id: str
    registered: RegisteredProductionModel


def _experiment_id(client: Any, experiment_name: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return str(experiment.experiment_id)
    return str(client.create_experiment(experiment_name))


def publish_production_package(
    artifact: ProductionModelArtifact,
    package_directory: str | Path,
    *,
    experiment_name: str = "regime-engine-production",
    run_name: str = "regime-xetra-final-refit",
) -> PublishedProductionModel:
    """Upload a final-refit package and register it by a remote ``runs:/`` URI.

    The package is uploaded to the NAS MLflow artifact store before its model
    version is created.  A local ``file:`` package URI is never registered,
    because the NAS MLflow server must be able to resolve the model package.
    """

    if type(artifact) is not ProductionModelArtifact:
        raise TypeError("only v4 ProductionModelArtifact objects can be published")
    package_path = Path(package_directory).resolve()
    if load_production_package(package_path) != artifact:
        raise ValueError("production package payload differs from supplied artifact")

    settings = MLflowSettings.from_environment()
    client = cast(
        Any,
        MlflowClient(
            tracking_uri=settings.tracking_uri,
            registry_uri=settings.registry_uri,
        ),
    )
    experiment_id = _experiment_id(client, experiment_name)
    run = client.create_run(
        experiment_id,
        tags={
            "mlflow.runName": run_name,
            "regime_engine.package_schema": "RegimeEngineProductionModel.v4",
            "regime_engine.source_build_id": artifact.source_build_id,
            "regime_engine.source_catalog_hash": artifact.source_catalog_hash,
            "regime_engine.validation_evidence_hash": artifact.validation_evidence_hash,
        },
    )
    run_id = str(run.info.run_id)
    source = f"runs:/{run_id}/production-package"
    try:
        client.log_artifacts(run_id, str(package_path), "production-package")
        client.set_tag(run_id, "regime_engine.model_name", artifact.registered_model)
        registered = MlflowModelRegistry(client).register_production_model(
            artifact,
            package_path,
            package_source_uri=source,
        )
        client.set_terminated(run_id, status="FINISHED")
    except BaseException:
        client.set_terminated(run_id, status="FAILED")
        raise
    return PublishedProductionModel(settings.tracking_uri, run_id, registered)


__all__ = ["PublishedProductionModel", "publish_production_package"]
