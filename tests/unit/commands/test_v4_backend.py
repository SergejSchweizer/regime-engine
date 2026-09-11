from __future__ import annotations

from pathlib import Path

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
