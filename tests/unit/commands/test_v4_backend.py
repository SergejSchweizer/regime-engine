from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import market_regime_engine.commands.v4_backend as module


def test_lifecycle_backend_requires_configured_persistent_state_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REGIME_ENGINE_STATE_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="persistent deployment volume"):
        module._configured_state_root(Path.cwd())


def test_lifecycle_state_root_is_absolute_and_outside_checkout(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    repository.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("REGIME_ENGINE_STATE_ROOT", "relative/state")
        with pytest.raises(RuntimeError, match="absolute"):
            module._configured_state_root(repository)
    outside = tmp_path / "persistent"
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("REGIME_ENGINE_STATE_ROOT", str(outside))
        assert module._configured_state_root(repository) == outside


def test_lifecycle_state_root_rejects_repository_subdirectory(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    repository.mkdir()
    state = repository / ".state"
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("REGIME_ENGINE_STATE_ROOT", str(state))
        with pytest.raises(RuntimeError, match="outside the repository"):
            module._configured_state_root(repository)


def test_lifecycle_evaluation_does_not_create_a_resume_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object.__new__(module.V4LifecycleBackend)
    backend.root = Path(__file__).parents[3]
    backend.state_root = tmp_path / "lifecycle"
    backend.profile = SimpleNamespace(
        profile_hash="a" * 64,
        pca=SimpleNamespace(variance_threshold=0.90, component_count=8),
    )
    backend.mlflow_settings = SimpleNamespace(tracking_uri="file:///tmp/mlflow")
    catalog = SimpleNamespace(lineage=SimpleNamespace(source_build_id="build-1"))
    snapshot = SimpleNamespace(
        feature_names=("f0",),
        rows=(
            SimpleNamespace(
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                values=(1.0,),
            ),
        ),
    )

    class Source:
        def read_with_catalog(self, request):
            del request
            return catalog, snapshot

    backend._source = Source()
    package = SimpleNamespace(state_count=2)
    result = SimpleNamespace(
        source_build_id="build-1",
        production_eligible=True,
        latest_package=package,
    )
    captured: dict[str, object] = {}

    pca_calls: list[dict[str, object]] = []

    def fake_materialize(source_catalog, source_snapshot, **kwargs):
        pca_calls.append(kwargs)
        assert source_catalog is catalog
        assert source_snapshot is snapshot
        return SimpleNamespace(catalog=catalog, snapshot=snapshot)

    def fake_evaluate(rows, **kwargs):
        captured["rows"] = rows
        captured.update(kwargs)
        return result

    monkeypatch.setattr(module, "run_canonical_xetra_evaluation", fake_evaluate)
    monkeypatch.setattr(module, "fit_and_materialize_pca_source", fake_materialize)
    monkeypatch.setattr(module, "CanonicalFileMlflowTrackingPort", lambda *args, **kwargs: object())

    outcome = backend.evaluate("xetra", "build-1")

    assert outcome.statistical_champion_candidate_id == "gaussian_hmm_k2_full"
    assert captured["rows"].empty is False
    assert isinstance(captured["metadata_store"], module.FeatureSelectionMetadataStore)
    assert pca_calls == [{"variance_threshold": 0.90, "component_count": 8}]
    assert not (backend.state_root / "evaluation-runs").exists()


def test_lifecycle_backend_helpers_round_trip_and_validate_state(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "state.pkl"
    value = {"source_build_id": "build-1"}
    module._atomic_pickle(path, value)
    assert module._read_pickle(path, dict) == value
    with pytest.raises(FileNotFoundError, match="missing\\.pkl"):
        module._read_pickle(tmp_path / "missing.pkl", dict)
    module._atomic_pickle(path, ["wrong"])
    with pytest.raises(ValueError, match="incompatible type"):
        module._read_pickle(path, dict)

    with pytest.raises(ValueError, match="incompatible type"):
        module._read_pickle(path, dict)
    assert module._file_sha256(path)


def test_lifecycle_backend_helper_projection_and_profile_contracts() -> None:
    now = datetime(2024, 1, 1, tzinfo=UTC)
    snapshot = SimpleNamespace(
        feature_names=("f0", "f1"),
        rows=(SimpleNamespace(timestamp=now, values=(1.0, 2.0)),),
    )
    frame = module._rows(snapshot)
    assert list(frame.columns) == ["timestamp_m1", "f0", "f1"]
    assert frame.iloc[0].to_dict()["f0"] == 1.0

    catalog = SimpleNamespace(
        feature_names=("f0", "f1"),
        lineage=SimpleNamespace(source_build_id="source-build"),
    )
    configuration = SimpleNamespace(
        candidate_id="student_t_hmm_k3_full",
        state_count=3,
        feature_order=("f0",),
        source_build_id=None,
        selection_definition_hash="a" * 64,
        selection_execution_hash="b" * 64,
        model_family="student_t_hmm",
    )
    candidate = module._candidate(configuration, catalog)
    assert candidate.source_build_id == "source-build"
    assert candidate.mixture_count == 1


def test_lifecycle_backend_source_capture_and_saved_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = object.__new__(module.V4LifecycleBackend)
    backend.state_root = tmp_path / "lifecycle"
    backend.profile = SimpleNamespace()
    backend.profile = SimpleNamespace(
        pca=SimpleNamespace(variance_threshold=0.95, component_count=None),
    )
    catalog = SimpleNamespace(lineage=SimpleNamespace(source_build_id="build"))
    snapshot = SimpleNamespace()

    class Source:
        def read_with_catalog(self, request):
            del request
            return catalog, snapshot

    backend._source = Source()
    monkeypatch.setattr(
        module,
        "fit_and_materialize_pca_source",
        lambda source_catalog, source_snapshot, **kwargs: SimpleNamespace(
            catalog=source_catalog, snapshot=source_snapshot
        ),
    )
    assert backend._capture_source() == (catalog, snapshot)
    assert backend._saved_source() == (catalog, snapshot)
    backend._source_path.write_bytes(module.pickle.dumps(("bad",)))
    with pytest.raises(ValueError, match="malformed"):
        backend._saved_source()


def test_lifecycle_backend_cached_evaluation_and_registry_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = object.__new__(module.V4LifecycleBackend)
    backend.state_root = tmp_path / "lifecycle"
    backend.mlflow_settings = SimpleNamespace(tracking_uri="file:///tmp/mlflow")
    package = SimpleNamespace(state_count=3)
    monthly = SimpleNamespace(valid_folds=(SimpleNamespace(package=package),))
    result = module.CanonicalXetraEvaluation(
        monthly=monthly,
        source_build_id="build",
        feature_selection_profile_hash="a" * 64,
        selected_features_by_fold=(("vix_log_level", "move_log_level"),),
    )
    module._atomic_pickle(backend._evaluation_path, result)
    assert (
        backend.evaluate("xetra", "build").statistical_champion_candidate_id
        == "gaussian_hmm_k3_full"
    )
    with pytest.raises(ValueError, match="differs"):
        backend.evaluate("xetra", "other")
    with pytest.raises(ValueError, match="only xetra"):
        backend.evaluate("other", "build")

    class Registry:
        def resolve_alias(self, name: str, alias: str) -> object:
            assert name == "regime-xetra"
            return SimpleNamespace(exact_version="7")

    monkeypatch.setattr(module, "MlflowModelRegistry", Registry)
    assert backend._registry_version("champion") == "7"

    class MissingRegistry:
        def resolve_alias(self, name: str, alias: str) -> object:
            raise module.MlflowException("resource not found")

    monkeypatch.setattr(module, "MlflowModelRegistry", MissingRegistry)
    assert backend._registry_version("challenger") is None


def test_lifecycle_backend_rejects_invalid_stage_arguments(tmp_path: Path) -> None:
    backend = object.__new__(module.V4LifecycleBackend)
    backend.state_root = tmp_path / "lifecycle"
    with pytest.raises(ValueError, match="only xetra"):
        backend.status("other")
    with pytest.raises(ValueError, match="completed global_regime_v4"):
        backend.final_refit("xetra", "wrong")
    with pytest.raises(ValueError, match="completed global_regime_v4"):
        backend.publish_oos("xetra", "wrong")
    with pytest.raises(ValueError, match="only xetra"):
        backend.register_challenger("other", "package", "oos")


def test_lifecycle_status_captures_source_and_completion_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = object.__new__(module.V4LifecycleBackend)
    backend.root = tmp_path / "repository"
    backend.root.mkdir()
    backend.state_root = tmp_path / "lifecycle"
    backend.profile = SimpleNamespace(
        pca=SimpleNamespace(variance_threshold=0.9, component_count=1),
    )
    catalog = SimpleNamespace(lineage=SimpleNamespace(source_build_id="current-build"))
    snapshot = SimpleNamespace()

    class Source:
        def read_with_catalog(self, request):
            del request
            return catalog, snapshot

    backend._source = Source()
    monkeypatch.setattr(
        module,
        "fit_and_materialize_pca_source",
        lambda catalog, snapshot, **kwargs: SimpleNamespace(catalog=catalog, snapshot=snapshot),
    )
    backend.state_root.mkdir()
    (backend.state_root / "completed.json").write_text(
        '{"source_build_id":"completed-build"}', encoding="utf-8"
    )
    monkeypatch.setattr(
        backend, "_registry_version", lambda alias: {"champion": "4", "challenger": None}[alias]
    )
    status = backend.status("xetra")
    assert status.current_source_build_id == "current-build"
    assert status.completed_source_build_id == "completed-build"
    assert status.champion_version == "4"
    assert status.challenger_version is None


def test_lifecycle_backend_requires_current_state_root_and_reports_registry_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "checkout"
    repository.mkdir()
    monkeypatch.setenv("REGIME_ENGINE_STATE_ROOT", str(tmp_path / "state"))
    assert module._configured_state_root(repository) == tmp_path / "state"

    backend = object.__new__(module.V4LifecycleBackend)

    class Registry:
        def resolve_alias(self, name: str, alias: str) -> object:
            raise module.MlflowException("permission denied")

    monkeypatch.setattr(module, "MlflowModelRegistry", Registry)
    with pytest.raises(module.MlflowException, match="permission denied"):
        backend._registry_version("champion")


def test_lifecycle_cached_final_refit_does_not_recompute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = object.__new__(module.V4LifecycleBackend)
    backend.state_root = tmp_path / "lifecycle"
    package = backend._package_path
    package.mkdir(parents=True)
    (package / "MLmodel").write_text("cached", encoding="utf-8")
    monkeypatch.setattr(module, "load_production_package", lambda path: object())
    outcome = backend.final_refit("xetra", "global_regime_v4")
    assert outcome.production_package == str(package)


def test_lifecycle_oos_publication_builds_and_persists_canonical_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = object.__new__(module.V4LifecycleBackend)
    backend.state_root = tmp_path / "lifecycle"
    backend.profile = SimpleNamespace()
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    package = SimpleNamespace(
        selected_features=("vix_log_level", "move_log_level"),
        state_count=2,
        model_hashes=("a" * 64,),
        package_hash="b" * 64,
        feature_selection_profile_hash="c" * 64,
    )
    fold = SimpleNamespace(
        fold=SimpleNamespace(
            fold_id="outer_fold_001",
            train_source_observations=1,
            test_source_observations=1,
        ),
        package=package,
    )
    monthly = SimpleNamespace(
        result_hash="result-hash",
        plan=SimpleNamespace(plan_hash="policy-hash"),
        valid_folds=(fold,),
    )
    result = module.CanonicalXetraEvaluation(
        monthly=monthly,
        source_build_id="build-1",
        feature_selection_profile_hash="c" * 64,
        selected_features_by_fold=package.selected_features,
    )
    module._atomic_pickle(backend._evaluation_path, result)
    catalog = SimpleNamespace(
        feature_names=("vix_log_level", "move_log_level"),
        catalog_hash="catalog-hash",
        lineage=SimpleNamespace(
            data_sha256="data-hash",
            schema_version="schema-v1",
            feature_version="features-v1",
            synced_at_utc=timestamp,
        ),
    )
    snapshot = SimpleNamespace(
        feature_names=("vix_log_level", "move_log_level"),
        rows=(
            SimpleNamespace(timestamp=timestamp, values=(1.0, 2.0)),
            SimpleNamespace(timestamp=datetime(2024, 1, 2, tzinfo=UTC), values=(1.0, 2.0)),
        ),
    )
    backend._saved_source = lambda: (catalog, snapshot)
    monkeypatch.setattr(
        module.CanonicalModelCallbacks,
        "outer_test_probabilities",
        lambda self, features, state_count, model_hashes: ((timestamp,), ((0.25, 0.75),)),
    )
    published: dict[str, object] = {}

    class Store:
        def __init__(self, path: Path) -> None:
            published["path"] = path

        def load_manifest(self, profile_id: str, build_id: str) -> object:
            raise FileNotFoundError(build_id)

        def publish(self, **kwargs: object) -> None:
            published.update(kwargs)

    monkeypatch.setattr(module, "PredictionStore", Store)
    outcome = backend.publish_oos("xetra", "global_regime_v4")
    assert outcome.oos_build_id == module.sha256(b"walk_forward_oos:result-hash").hexdigest()
    assert published["profile_id"] == "xetra"
    assert published["source_build_id"] == "build-1"
    rows = published["rows"]
    assert isinstance(rows, list)
    assert rows[0]["dominant_state"] == "state_001"
    assert rows[0]["confidence"] == 0.75
