from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[2] / "scripts" / "run_xetra_v4_evaluation.py"
    spec = importlib.util.spec_from_file_location("run_xetra_v4_evaluation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load evaluation entrypoint")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_root_requires_explicit_absolute_external_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.delenv("REGIME_EVALUATION_CHECKPOINT_ROOT", raising=False)
    monkeypatch.delenv("REGIME_ENGINE_STATE_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="must be configured"):
        module._configured_checkpoint_root(tmp_path)

    monkeypatch.setenv("REGIME_EVALUATION_CHECKPOINT_ROOT", "relative/state")
    with pytest.raises(RuntimeError, match="absolute"):
        module._configured_checkpoint_root(tmp_path)


def test_checkpoint_root_cannot_be_repository_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setenv("REGIME_EVALUATION_CHECKPOINT_ROOT", str(tmp_path / "state"))
    with pytest.raises(RuntimeError, match="outside the repository"):
        module._configured_checkpoint_root(tmp_path)


def test_checkpoint_root_accepts_external_persistent_volume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    external = tmp_path.parent / "persistent-xetra-runs"
    monkeypatch.setenv("REGIME_EVALUATION_CHECKPOINT_ROOT", str(external))
    assert module._configured_checkpoint_root(tmp_path) == external.resolve()
