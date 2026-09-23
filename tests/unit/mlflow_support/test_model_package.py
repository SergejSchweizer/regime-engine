from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from market_regime_engine.mlflow_support import model_publishing
from market_regime_engine.mlflow_support.model_package import (
    MLMODEL_FILE,
    PACKAGE_DATA_FILE,
    load_production_package,
    production_artifact_from_json,
    production_artifact_from_payload,
    production_artifact_json,
    save_production_package,
)
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.production_artifact import ProductionModelArtifact
from market_regime_engine.preprocessing.two_stage import fit_family_pca_hmm_scaler


def artifact() -> ProductionModelArtifact:
    feature_order = ("f0", "f1")
    index = np.arange(120, dtype=np.float64)
    pca_scaler = fit_family_pca_hmm_scaler(
        np.column_stack((np.sin(index / 5.0), np.cos(index / 7.0))),
        raw_feature_order=feature_order,
        family_pca=(),
        model_feature_order=feature_order,
    )
    return ProductionModelArtifact(
        profile_id="xetra",
        profile_config_version=4,
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
        validation_evaluation_cutoff=datetime(2026, 8, 20, tzinfo=UTC),
        deployment_selection_cutoff=datetime(2026, 8, 21, tzinfo=UTC),
        validation_evidence_hash="e" * 64,
        source_catalog_hash="f" * 64,
        state_identity_scope="model_version_local",
        feature_order=feature_order,
        scaler=pca_scaler.hmm_scaler,
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
        pca_scaler=pca_scaler,
    )


def pca_artifact() -> ProductionModelArtifact:
    base = artifact()
    index = np.arange(120, dtype=np.float64)
    raw_rows = np.column_stack((np.sin(index / 5.0), np.cos(index / 7.0)))
    pca_scaler = fit_family_pca_hmm_scaler(
        raw_rows,
        raw_feature_order=("f0", "f1"),
        family_pca=(),
        model_feature_order=("f0", "f1"),
    )
    dimension = len(pca_scaler.model_feature_order)
    covariance = tuple(
        tuple(0.2 if row == column else 0.0 for column in range(dimension))
        for row in range(dimension)
    )
    hmm = replace(
        base.hmm,
        feature_order=pca_scaler.model_feature_order,
        means=tuple(tuple(-1.0 for _ in range(dimension)) for _ in range(2)),
        full_covariances=(covariance, covariance),
    )
    return replace(
        base,
        feature_order=pca_scaler.model_feature_order,
        scaler=pca_scaler.hmm_scaler,
        hmm=hmm,
        pca_scaler=pca_scaler,
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


def test_gmm_hmm_json_roundtrip_preserves_two_mixture_emissions() -> None:
    base = artifact()
    hmm = base.hmm
    gmm = replace(
        hmm,
        model_family="gmm_hmm",
        mixture_weights=((0.5, 0.5), (0.5, 0.5)),
        mixture_means=(
            ((-1.2, 0.4), (-0.8, 0.6)),
            ((1.3, -0.35), (1.7, -0.15)),
        ),
        mixture_full_covariances=(
            (hmm.full_covariances[0], hmm.full_covariances[0]),
            (hmm.full_covariances[1], hmm.full_covariances[1]),
        ),
    )
    original = replace(base, candidate_id="gmm_hmm_k2_m2_full", hmm=gmm)

    payload = production_artifact_json(original)
    restored = production_artifact_from_json(payload)

    assert restored == original
    assert json.loads(payload)["hmm"]["model_family"] == "gmm_hmm"


def test_pca_lineage_json_roundtrip_preserves_family_artifact() -> None:
    original = pca_artifact()
    payload = production_artifact_json(original)

    restored = production_artifact_from_json(payload)

    assert restored == original
    assert json.loads(payload)["pca_scaler"]["artifact_schema"] == (
        "RegimeEngineFamilyPCATwoStageScaler.v1"
    )


def test_student_t_hmm_json_roundtrip_preserves_state_degrees_of_freedom() -> None:
    base = artifact()
    student = replace(
        base.hmm,
        model_family="student_t_hmm",
        degrees_of_freedom=(4.25, 11.5),
    )
    original = replace(base, candidate_id="student_t_hmm_k2_full", hmm=student)
    payload = production_artifact_json(original)
    restored = production_artifact_from_json(payload)
    assert restored == original
    assert json.loads(payload)["hmm"]["degrees_of_freedom_hex"] == [
        (4.25).hex(),
        (11.5).hex(),
    ]


def test_save_load_package_roundtrip_and_immutability(tmp_path: Path) -> None:
    original = artifact()
    package = save_production_package(original, tmp_path / "model")
    assert (package / MLMODEL_FILE).is_file()
    assert (package / PACKAGE_DATA_FILE).is_file()
    assert load_production_package(package) == original
    with pytest.raises(FileExistsError, match="immutable"):
        save_production_package(original, package)


def test_package_fails_closed_on_metadata_and_payload_drift(tmp_path: Path) -> None:
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


def test_model_publishing_resolves_or_creates_experiment() -> None:
    class Client:
        def __init__(self, existing: object | None) -> None:
            self.existing = existing
            self.created: list[str] = []

        def get_experiment_by_name(self, name: str) -> object | None:
            return self.existing

        def create_experiment(self, name: str) -> str:
            self.created.append(name)
            return "42"

    existing = Client(type("Experiment", (), {"experiment_id": "7"})())
    assert model_publishing._experiment_id(existing, "exp") == "7"
    created = Client(None)
    assert model_publishing._experiment_id(created, "exp") == "42"
    assert created.created == ["exp"]


def test_model_publishing_rejects_wrong_artifact_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="ProductionModelArtifact"):
        model_publishing.publish_production_package(object(), tmp_path)


def test_model_publishing_uploads_package_and_registers_remote_uri(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = artifact()
    package = save_production_package(original, tmp_path / "model")
    calls: list[tuple[str, object]] = []

    class Client:
        def __init__(self, *, tracking_uri: str, registry_uri: str) -> None:
            assert tracking_uri == registry_uri == "http://10.10.1.3:5000"

        def get_experiment_by_name(self, name: str) -> None:
            assert name == "regime-engine-production"
            return None

        def create_experiment(self, name: str) -> str:
            calls.append(("experiment", name))
            return "experiment-1"

        def create_run(self, experiment_id: str, *, tags: dict[str, str]) -> object:
            calls.append(("run", (experiment_id, tags)))
            return SimpleNamespace(info=SimpleNamespace(run_id="run-1"))

        def log_artifacts(self, run_id: str, path: str, artifact_path: str) -> None:
            calls.append(("artifacts", (run_id, path, artifact_path)))

        def set_tag(self, run_id: str, key: str, value: str) -> None:
            calls.append(("tag", (run_id, key, value)))

        def set_terminated(self, run_id: str, *, status: str) -> None:
            calls.append(("terminated", (run_id, status)))

    registered = SimpleNamespace(version="1", source="runs:/run-1/production-package")
    monkeypatch.setattr(
        model_publishing,
        "MLflowSettings",
        SimpleNamespace(
            from_environment=lambda: SimpleNamespace(
                tracking_uri="http://10.10.1.3:5000", registry_uri="http://10.10.1.3:5000"
            )
        ),
    )
    monkeypatch.setattr(model_publishing, "MlflowClient", Client)
    monkeypatch.setattr(
        model_publishing.MlflowModelRegistry,
        "register_production_model",
        lambda self, artifact, path, *, package_source_uri: (
            calls.append(("register", package_source_uri)) or registered
        ),
    )

    result = model_publishing.publish_production_package(original, package)

    assert result.tracking_uri == "http://10.10.1.3:5000"
    assert result.run_id == "run-1"
    assert result.registered is registered
    assert ("terminated", ("run-1", "FINISHED")) in calls
    assert ("register", "runs:/run-1/production-package") in calls


def test_model_publishing_marks_failed_run_when_registry_rejects_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = artifact()
    package = save_production_package(original, tmp_path / "model")
    statuses: list[str] = []

    class Client:
        def __init__(self, *, tracking_uri: str, registry_uri: str) -> None:
            assert tracking_uri == registry_uri == "http://10.10.1.3:5000"

        def get_experiment_by_name(self, name: str) -> object:
            return SimpleNamespace(experiment_id="experiment-1")

        def create_run(self, experiment_id: str, *, tags: dict[str, str]) -> object:
            return SimpleNamespace(info=SimpleNamespace(run_id="run-1"))

        def log_artifacts(self, run_id: str, path: str, artifact_path: str) -> None:
            pass

        def set_tag(self, run_id: str, key: str, value: str) -> None:
            pass

        def set_terminated(self, run_id: str, *, status: str) -> None:
            statuses.append(status)

    monkeypatch.setattr(
        model_publishing,
        "MLflowSettings",
        SimpleNamespace(
            from_environment=lambda: SimpleNamespace(
                tracking_uri="http://10.10.1.3:5000", registry_uri="http://10.10.1.3:5000"
            )
        ),
    )
    monkeypatch.setattr(model_publishing, "MlflowClient", Client)

    def reject(*args: object, **kwargs: object) -> object:
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(model_publishing.MlflowModelRegistry, "register_production_model", reject)
    with pytest.raises(RuntimeError, match="registry unavailable"):
        model_publishing.publish_production_package(original, package)
    assert statuses == ["FAILED"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("profile_id", "other", "Xetra v4"),
        ("profile_config_version", 3, "Xetra v4"),
        ("registered_model", "other", "registered model"),
        ("state_count", 1, "K2/K3/K4/K5"),
        ("candidate_id", "unknown", "candidate identity"),
        ("source_build_id", "", "source identity"),
        ("source_data_sha256", "x", "SHA-256"),
        ("source_schema_version", 0, "versions must be positive"),
        ("data_time_semantics", "historical", "data_time_semantics"),
        ("validation_evaluation_cutoff", datetime(2026, 8, 20), "UTC"),
        ("state_identity_scope", "global", "model_version_local"),
        ("retained_observation_count", 503, "504 retained"),
        ("skipped_incomplete_observation_count", -1, "cannot be negative"),
        ("terminal_filtered_probabilities", (1.0,), "dimension"),
        ("winning_seed", 17, "pinned eight-seed"),
    ],
)
def test_production_artifact_rejects_invalid_identity_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(artifact(), **{field: value})


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

    raw = json.loads(production_artifact_json(artifact()))
    raw["pca_scaler"] = []
    with pytest.raises(ValueError, match="PCA scaler payload"):
        production_artifact_from_json(json.dumps(raw))

    raw = json.loads(production_artifact_json(artifact()))
    raw.pop("pca_scaler")
    with pytest.raises(ValueError, match="unknown or missing"):
        production_artifact_from_json(json.dumps(raw))


def test_json_loader_rejects_nested_schema_drift_and_cross_family_fields() -> None:
    mutations: tuple[tuple[str, Callable[[dict[str, Any]], Any]], ...] = (
        ("missing HMM field", lambda raw: raw["hmm"].pop("means_hex")),
        ("unknown HMM field", lambda raw: raw["hmm"].update({"unexpected": True})),
        ("missing scaler field", lambda raw: raw["scaler"].pop("scales_hex")),
        ("unknown scaler field", lambda raw: raw["scaler"].update({"unexpected": True})),
        (
            "Gaussian with Student-t field",
            lambda raw: raw["hmm"].update({"degrees_of_freedom_hex": [(4.0).hex()]}),
        ),
        (
            "Student-t without degrees of freedom",
            lambda raw: raw["hmm"].update({"model_family": "student_t_hmm"}),
        ),
        (
            "GMM without mixture payload",
            lambda raw: raw["hmm"].update({"model_family": "gmm_hmm"}),
        ),
    )

    for _description, mutate in mutations:
        raw = json.loads(production_artifact_json(artifact()))
        mutate(raw)
        with pytest.raises(ValueError, match="unknown or missing"):
            production_artifact_from_json(json.dumps(raw))


def test_old_package_schema_is_rejected_before_nested_payload_decoding() -> None:
    payload: dict[str, object] = {
        "schema_version": "RegimeEngineProductionModel.v3",
        "hmm": object(),
        "opaque_legacy_payload": object(),
    }

    with pytest.raises(ValueError, match="unsupported production package schema"):
        production_artifact_from_json(json.dumps({"schema_version": payload["schema_version"]}))
    with pytest.raises(ValueError, match="unsupported production package schema"):
        # The direct mapping path models a caller that already parsed an old
        # package whose nested fields are not compatible with the v4 schema.
        production_artifact_from_payload(payload)


def test_package_loader_rejects_missing_or_incompatible_mlmodel(tmp_path: Path) -> None:
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


def test_package_loader_rejects_runtime_version_drift(tmp_path: Path) -> None:
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
    with pytest.raises(TypeError, match="v4"):
        production_artifact_json(object())  # type: ignore[arg-type]
    assert replace(artifact(), winning_seed=131).winning_seed == 131
