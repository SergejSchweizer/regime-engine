from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import market_regime_engine.commands.v4_backend as module


def test_lifecycle_backend_requires_configured_persistent_state_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REGIME_ENGINE_STATE_ROOT", raising=False)
    monkeypatch.delenv("REGIME_EVALUATION_CHECKPOINT_ROOT", raising=False)
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
    backend.profile = object()
    backend.mlflow_settings = SimpleNamespace(tracking_uri="file:///tmp/mlflow")
    catalog = SimpleNamespace(lineage=SimpleNamespace(source_build_id="build-1"))
    snapshot = SimpleNamespace()

    class Source:
        def read_schema_wide_with_catalog(self, request):
            del request
            return catalog, snapshot

    backend._source = Source()
    fold = SimpleNamespace(
        valid=True,
        final_configuration=SimpleNamespace(candidate_id="gaussian_hmm_k2_full"),
    )
    result = SimpleNamespace(
        source_build_id="build-1",
        production_eligible=True,
        outer_folds=(fold,),
    )
    captured: dict[str, object] = {}

    def fake_evaluate(source, **kwargs):
        captured["source"] = source
        captured.update(kwargs)
        return result

    monkeypatch.setattr(module, "evaluate_global_regime_v4_from_source", fake_evaluate)
    monkeypatch.setattr(module, "build_global_v4_evidence", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "track_global_v4_evaluation", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "FileMlflowTrackingPort", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "_commit", lambda root: "c" * 40)
    monkeypatch.setattr(module, "_file_sha256", lambda path: "d" * 64)

    outcome = backend.evaluate("xetra", "build-1")

    assert outcome.statistical_champion_candidate_id == "gaussian_hmm_k2_full"
    assert "run_store" not in captured
    assert isinstance(captured["snapshot_store"], module.ArrowDatasetSnapshotStore)
    assert not (backend.state_root / "evaluation-runs").exists()
