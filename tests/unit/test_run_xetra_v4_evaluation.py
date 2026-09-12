from __future__ import annotations

import importlib.util
import sys
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


def test_tracking_worker_count_uses_all_available_cpus_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.delenv("REGIME_TRACKING_WORKERS", raising=False)
    assert module._tracking_worker_count(10_000) == module.available_cpu_count()


def test_tracking_worker_count_respects_explicit_bounded_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setenv("REGIME_TRACKING_WORKERS", "2")
    assert module._tracking_worker_count(10_000) == min(2, module.available_cpu_count())


def test_full_evaluation_does_not_accept_a_resume_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(sys, "argv", ["run_xetra_v4_evaluation.py", "--run-key", "old-run"])
    with pytest.raises(SystemExit):
        module._run(None)
