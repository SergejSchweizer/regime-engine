"""MLflow client boundary settings without import-time network work."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import environ

PRODUCTION_MLFLOW_URI = "http://10.10.1.3:5000"


@dataclass(frozen=True, slots=True)
class MLflowSettings:
    tracking_uri: str = PRODUCTION_MLFLOW_URI
    registry_uri: str = PRODUCTION_MLFLOW_URI

    def __post_init__(self) -> None:
        if self.tracking_uri != PRODUCTION_MLFLOW_URI:
            raise ValueError("production tracking URI must be exactly http://10.10.1.3:5000")
        if self.registry_uri != self.tracking_uri:
            raise ValueError("tracking and registry must use the same one-port MLflow service")

    @classmethod
    def from_environment(cls, values: Mapping[str, str] | None = None) -> MLflowSettings:
        """Load the single external MLflow endpoint used by production jobs."""

        source = environ if values is None else values
        tracking_uri = source.get("MLFLOW_TRACKING_URI", PRODUCTION_MLFLOW_URI)
        registry_uri = source.get("MLFLOW_REGISTRY_URI", tracking_uri)
        return cls(tracking_uri=tracking_uri, registry_uri=registry_uri)
