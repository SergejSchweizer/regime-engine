"""MLflow client boundary settings without import-time network work."""

from __future__ import annotations

from dataclasses import dataclass

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
