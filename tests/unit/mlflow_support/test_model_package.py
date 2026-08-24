from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from market_regime_engine.mlflow_support.model_package import (
    MLMODEL_FILE,
    PACKAGE_DATA_FILE,
    load_production_package,
    production_artifact_from_json,
    production_artifact_json,
    save_production_package,
)
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.production_artifact import ProductionModelArtifact
from market_regime_engine.preprocessing.scaling import StandardScalerArtifact


def artifact() -> ProductionModelArtifact:
    feature_order = ("f0", "f1")
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
        feature_order=feature_order,
        scaler=StandardScalerArtifact(
            feature_order=feature_order,
            means=(0.25, -0.5),
            variances=(1.5, 2.0),
            scales=(1.5**0.5, 2.0**0.5),
        ),
        hmm=GaussianHMMArtifact(
            state_count=2,
            feature_order=feature_order,
            start_probabilities=(0.4, 0.6),
            transition_matrix=((0.8, 0.2), (0.1, 0.9)),
            means=((-1.0, 0.5), (1.5, -0.25)),
            full_covariances=(
                ((1.0, 0.2), (0.2, 1.5)),
                ((2.0, -0.3), (-0.3, 1.25)),
            ),
        ),
        winning_seed=23,
        inference_origin_timestamp=datetime(2020, 1, 2, tzinfo=UTC),
        trained_through_timestamp=datetime(2026, 8, 19, tzinfo=UTC),
        terminal_filtered_probabilities=(0.3, 0.7),
        retained_observation_count=1500,
        skipped_incomplete_observation_count=12,
    )


def test_json_roundtrip_is_lossless_and_deterministic() -> None:
    original = artifact()
    payload = production_artifact_json(original)
    assert production_artifact_from_json(payload) == original
    assert production_artifact_json(original) == payload
    raw = json.loads(payload)
    assert "source_table" not in raw
    assert "password" not in payload.lower()
    assert raw["hmm"]["full_covariances_hex"][0][0][1] == (0.2).hex()


def test_save_load_package_roundtrip_and_immutability(tmp_path) -> None:
    original = artifact()
    package = save_production_package(original, tmp_path / "model")
    assert (package / MLMODEL_FILE).is_file()
    assert (package / PACKAGE_DATA_FILE).is_file()
    assert load_production_package(package) == original
    with pytest.raises(FileExistsError, match="immutable"):
        save_production_package(original, package)


def test_package_fails_closed_on_metadata_and_payload_drift(tmp_path) -> None:
    package = save_production_package(artifact(), tmp_path / "model")
    mlmodel_path = package / MLMODEL_FILE
    metadata = json.loads(mlmodel_path.read_text(encoding="utf-8"))
    metadata["model_name"] = "other"
    mlmodel_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="identity differs"):
        load_production_package(package)

    data_path = package / PACKAGE_DATA_FILE
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    raw["unknown"] = True
    data_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown or missing"):
        production_artifact_from_json(data_path.read_text(encoding="utf-8"))


def test_json_loader_rejects_root_shape_fields_and_schema() -> None:
    with pytest.raises(ValueError, match="root must be a mapping"):
        production_artifact_from_json("[]")

    raw = json.loads(production_artifact_json(artifact()))
    raw.pop("candidate_id")
    with pytest.raises(ValueError, match="unknown or missing"):
        production_artifact_from_json(json.dumps(raw))

    raw = json.loads(production_artifact_json(artifact()))
    raw["schema_version"] = "future.v2"
    with pytest.raises(ValueError, match="unsupported production package schema"):
        production_artifact_from_json(json.dumps(raw))

    raw = json.loads(production_artifact_json(artifact()))
    raw["scaler"] = []
    with pytest.raises(ValueError, match="scaler/HMM payloads must be mappings"):
        production_artifact_from_json(json.dumps(raw))


def test_package_loader_rejects_missing_or_incompatible_mlmodel(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires MLmodel"):
        load_production_package(tmp_path / "missing")

    package = save_production_package(artifact(), tmp_path / "bad-data-ref")
    path = package / MLMODEL_FILE
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["flavors"]["regime_engine"]["data"] = "wrong.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical production data file"):
        load_production_package(package)

    package = save_production_package(artifact(), tmp_path / "bad-schema")
    path = package / MLMODEL_FILE
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["flavors"]["regime_engine"]["schema_version"] = "future.v2"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="schema is incompatible"):
        load_production_package(package)


def test_package_loader_rejects_runtime_version_drift(tmp_path) -> None:
    package = save_production_package(artifact(), tmp_path / "bad-mlflow")
    path = package / MLMODEL_FILE
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["mlflow_version"] = "3.16.0"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match=r"MLflow 3\.15\.1"):
        load_production_package(package)

    package = save_production_package(artifact(), tmp_path / "bad-python")
    path = package / MLMODEL_FILE
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["python_version"] = "3.14.8"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match=r"Python 3\.14\.7"):
        load_production_package(package)


def test_only_final_refit_artifact_type_is_accepted() -> None:
    with pytest.raises(TypeError, match="PR-063"):
        production_artifact_json(object())  # type: ignore[arg-type]
    assert replace(artifact(), winning_seed=131).winning_seed == 131
