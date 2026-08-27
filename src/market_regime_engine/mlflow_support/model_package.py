"""Canonical filesystem package for final-refit production model artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.production_artifact import ProductionModelArtifact
from market_regime_engine.preprocessing.scaling import StandardScalerArtifact

PACKAGE_SCHEMA_VERSION = "RegimeEngineProductionModel.v1"
PACKAGE_DATA_FILE = "production_model.json"
MLMODEL_FILE = "MLmodel"


def _hex(value: float) -> str:
    return value.hex()


def _from_hex(value: str) -> float:
    return float.fromhex(value)


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _scaler_payload(scaler: StandardScalerArtifact) -> dict[str, Any]:
    return {
        "feature_order": list(scaler.feature_order),
        "means_hex": [_hex(value) for value in scaler.means],
        "scales_hex": [_hex(value) for value in scaler.scales],
        "variances_hex": [_hex(value) for value in scaler.variances],
    }


def _hmm_payload(hmm: GaussianHMMArtifact) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "covariance_type": hmm.covariance_type,
        "feature_order": list(hmm.feature_order),
        "full_covariances_hex": [
            [[_hex(value) for value in row] for row in matrix] for matrix in hmm.full_covariances
        ],
        "means_hex": [[_hex(value) for value in row] for row in hmm.means],
        "model_family": hmm.model_family,
        "start_probabilities_hex": [_hex(value) for value in hmm.start_probabilities],
        "state_count": hmm.state_count,
        "transition_matrix_hex": [[_hex(value) for value in row] for row in hmm.transition_matrix],
    }
    if hmm.model_family == "gmm_hmm":
        assert hmm.mixture_weights is not None
        assert hmm.mixture_means is not None
        assert hmm.mixture_full_covariances is not None
        payload["mixture_weights_hex"] = [
            [_hex(value) for value in row] for row in hmm.mixture_weights
        ]
        payload["mixture_means_hex"] = [
            [[_hex(value) for value in mixture] for mixture in state] for state in hmm.mixture_means
        ]
        payload["mixture_full_covariances_hex"] = [
            [[[_hex(value) for value in row] for row in mixture] for mixture in state]
            for state in hmm.mixture_full_covariances
        ]
    return payload


def production_artifact_payload(artifact: ProductionModelArtifact) -> dict[str, Any]:
    """Return the complete JSON-safe production contract without secret/source-table data."""

    if type(artifact) is not ProductionModelArtifact:
        raise TypeError("only PR-063 ProductionModelArtifact objects can be packaged")
    return {
        "candidate_id": artifact.candidate_id,
        "data_time_semantics": artifact.data_time_semantics,
        "evaluation_cutoff": _timestamp(artifact.evaluation_cutoff),
        "evaluation_plan_hash": artifact.evaluation_plan_hash,
        "feature_order": list(artifact.feature_order),
        "feature_selection_definition_hash": artifact.feature_selection_definition_hash,
        "feature_selection_execution_hash": artifact.feature_selection_execution_hash,
        "hmm": _hmm_payload(artifact.hmm),
        "inference_origin_timestamp": _timestamp(artifact.inference_origin_timestamp),
        "profile_config_version": artifact.profile_config_version,
        "profile_id": artifact.profile_id,
        "registered_model": artifact.registered_model,
        "retained_observation_count": artifact.retained_observation_count,
        "scaler": _scaler_payload(artifact.scaler),
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "skipped_incomplete_observation_count": artifact.skipped_incomplete_observation_count,
        "source_build_id": artifact.source_build_id,
        "source_data_sha256": artifact.source_data_sha256,
        "source_feature_version": artifact.source_feature_version,
        "source_schema_version": artifact.source_schema_version,
        "state_count": artifact.state_count,
        "terminal_filtered_probabilities_hex": [
            _hex(value) for value in artifact.terminal_filtered_probabilities
        ],
        "trained_through_timestamp": _timestamp(artifact.trained_through_timestamp),
        "winning_seed": artifact.winning_seed,
    }


def production_artifact_json(artifact: ProductionModelArtifact) -> str:
    """Serialize the package data deterministically and losslessly for float values."""

    return json.dumps(
        production_artifact_payload(artifact),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def production_artifact_from_payload(payload: dict[str, Any]) -> ProductionModelArtifact:
    expected = {
        "candidate_id",
        "data_time_semantics",
        "evaluation_cutoff",
        "evaluation_plan_hash",
        "feature_order",
        "feature_selection_definition_hash",
        "feature_selection_execution_hash",
        "hmm",
        "inference_origin_timestamp",
        "profile_config_version",
        "profile_id",
        "registered_model",
        "retained_observation_count",
        "scaler",
        "schema_version",
        "skipped_incomplete_observation_count",
        "source_build_id",
        "source_data_sha256",
        "source_feature_version",
        "source_schema_version",
        "state_count",
        "terminal_filtered_probabilities_hex",
        "trained_through_timestamp",
        "winning_seed",
    }
    if set(payload) != expected:
        raise ValueError("unknown or missing production package fields")
    if payload["schema_version"] != PACKAGE_SCHEMA_VERSION:
        raise ValueError("unsupported production package schema version")

    scaler_payload = payload["scaler"]
    hmm_payload = payload["hmm"]
    if not isinstance(scaler_payload, dict) or not isinstance(hmm_payload, dict):
        raise ValueError("production scaler/HMM payloads must be mappings")

    scaler = StandardScalerArtifact(
        feature_order=tuple(scaler_payload["feature_order"]),
        means=tuple(_from_hex(value) for value in scaler_payload["means_hex"]),
        variances=tuple(_from_hex(value) for value in scaler_payload["variances_hex"]),
        scales=tuple(_from_hex(value) for value in scaler_payload["scales_hex"]),
    )
    model_family = str(hmm_payload.get("model_family", "gaussian_hmm"))
    mixture_weights = (
        None
        if model_family == "gaussian_hmm"
        else tuple(
            tuple(_from_hex(value) for value in row) for row in hmm_payload["mixture_weights_hex"]
        )
    )
    mixture_means = (
        None
        if model_family == "gaussian_hmm"
        else tuple(
            tuple(tuple(_from_hex(value) for value in mixture) for mixture in state)
            for state in hmm_payload["mixture_means_hex"]
        )
    )
    mixture_covariances = (
        None
        if model_family == "gaussian_hmm"
        else tuple(
            tuple(
                tuple(tuple(_from_hex(value) for value in row) for row in mixture)
                for mixture in state
            )
            for state in hmm_payload["mixture_full_covariances_hex"]
        )
    )
    hmm = GaussianHMMArtifact(
        state_count=int(hmm_payload["state_count"]),
        feature_order=tuple(hmm_payload["feature_order"]),
        start_probabilities=tuple(
            _from_hex(value) for value in hmm_payload["start_probabilities_hex"]
        ),
        transition_matrix=tuple(
            tuple(_from_hex(value) for value in row) for row in hmm_payload["transition_matrix_hex"]
        ),
        means=tuple(tuple(_from_hex(value) for value in row) for row in hmm_payload["means_hex"]),
        full_covariances=tuple(
            tuple(tuple(_from_hex(value) for value in row) for row in matrix)
            for matrix in hmm_payload["full_covariances_hex"]
        ),
        covariance_type=str(hmm_payload["covariance_type"]),
        model_family=model_family,
        mixture_weights=mixture_weights,
        mixture_means=mixture_means,
        mixture_full_covariances=mixture_covariances,
    )
    return ProductionModelArtifact(
        profile_id=str(payload["profile_id"]),
        profile_config_version=int(payload["profile_config_version"]),
        registered_model=str(payload["registered_model"]),
        candidate_id=str(payload["candidate_id"]),
        state_count=int(payload["state_count"]),
        source_build_id=str(payload["source_build_id"]),
        source_data_sha256=str(payload["source_data_sha256"]),
        source_schema_version=int(payload["source_schema_version"]),
        source_feature_version=int(payload["source_feature_version"]),
        data_time_semantics=str(payload["data_time_semantics"]),
        feature_selection_definition_hash=str(payload["feature_selection_definition_hash"]),
        feature_selection_execution_hash=str(payload["feature_selection_execution_hash"]),
        evaluation_plan_hash=str(payload["evaluation_plan_hash"]),
        evaluation_cutoff=_parse_timestamp(str(payload["evaluation_cutoff"])),
        feature_order=tuple(payload["feature_order"]),
        scaler=scaler,
        hmm=hmm,
        winning_seed=int(payload["winning_seed"]),
        inference_origin_timestamp=_parse_timestamp(str(payload["inference_origin_timestamp"])),
        trained_through_timestamp=_parse_timestamp(str(payload["trained_through_timestamp"])),
        terminal_filtered_probabilities=tuple(
            _from_hex(value) for value in payload["terminal_filtered_probabilities_hex"]
        ),
        retained_observation_count=int(payload["retained_observation_count"]),
        skipped_incomplete_observation_count=int(payload["skipped_incomplete_observation_count"]),
    )


def production_artifact_from_json(payload: str) -> ProductionModelArtifact:
    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise ValueError("production package root must be a mapping")
    return production_artifact_from_payload(raw)


def save_production_package(
    artifact: ProductionModelArtifact,
    package_directory: str | Path,
) -> Path:
    """Write one immutable local model package that MLflow can register by URI."""

    package_path = Path(package_directory)
    package_path.mkdir(parents=True, exist_ok=True)
    data_path = package_path / PACKAGE_DATA_FILE
    mlmodel_path = package_path / MLMODEL_FILE
    if data_path.exists() or mlmodel_path.exists():
        raise FileExistsError("production model package is immutable once written")

    data_path.write_text(production_artifact_json(artifact) + "\n", encoding="utf-8")
    mlmodel = {
        "flavors": {
            "regime_engine": {
                "data": PACKAGE_DATA_FILE,
                "schema_version": PACKAGE_SCHEMA_VERSION,
            }
        },
        "mlflow_version": "3.15.1",
        "model_name": artifact.registered_model,
        "python_version": "3.14.7",
    }
    mlmodel_path.write_text(
        json.dumps(mlmodel, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return package_path


def load_production_package(package_directory: str | Path) -> ProductionModelArtifact:
    package_path = Path(package_directory)
    data_path = package_path / PACKAGE_DATA_FILE
    mlmodel_path = package_path / MLMODEL_FILE
    if not data_path.is_file() or not mlmodel_path.is_file():
        raise ValueError("production package requires MLmodel and production_model.json")
    metadata = json.loads(mlmodel_path.read_text(encoding="utf-8"))
    flavor = metadata.get("flavors", {}).get("regime_engine", {})
    if flavor.get("data") != PACKAGE_DATA_FILE:
        raise ValueError("MLmodel does not reference the canonical production data file")
    if flavor.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise ValueError("MLmodel production package schema is incompatible")
    artifact = production_artifact_from_json(data_path.read_text(encoding="utf-8"))
    if metadata.get("model_name") != artifact.registered_model:
        raise ValueError("MLmodel registered model identity differs from package payload")
    if metadata.get("mlflow_version") != "3.15.1":
        raise ValueError("production package requires MLflow 3.15.1")
    if metadata.get("python_version") != "3.14.7":
        raise ValueError("production package requires Python 3.14.7")
    return artifact
