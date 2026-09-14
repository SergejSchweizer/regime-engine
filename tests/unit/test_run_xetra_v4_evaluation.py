from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from market_regime_engine.features.ports import FeatureRequest


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


def test_full_audit_requires_all_valid_fold_selections_and_policy_eligibility() -> None:
    module = _module()
    eligible = SimpleNamespace(
        outer_folds=(
            SimpleNamespace(fold_index=2, valid=True),
            SimpleNamespace(fold_index=1, valid=True),
            SimpleNamespace(fold_index=3, valid=False),
        ),
        production_eligible=True,
    )
    module._require_full_audit_eligibility(eligible, {1: object(), 2: object()})

    with pytest.raises(RuntimeError, match="every valid outer fold"):
        module._require_full_audit_eligibility(eligible, {1: object()})

    ineligible = SimpleNamespace(
        outer_folds=(SimpleNamespace(fold_index=1, valid=True),),
        production_eligible=False,
    )
    with pytest.raises(RuntimeError, match="production-eligible"):
        module._require_full_audit_eligibility(ineligible, {1: object()})


def test_current_audit_source_request_rejects_allowlists_and_time_bounds() -> None:
    module = _module()
    complete = FeatureRequest.all_features()
    assert module._source_request_evidence(complete)["all_source_rows"] is True

    with pytest.raises(RuntimeError, match="allowlist"):
        module._source_request_evidence(FeatureRequest(("feature_a",), None, None, complete.mode))
    with pytest.raises(RuntimeError, match="timestamp bounds"):
        module._source_request_evidence(
            FeatureRequest(
                (),
                datetime(2026, 1, 1, tzinfo=UTC),
                None,
                complete.mode,
            )
        )


def test_current_audit_requires_durable_summary_and_resource_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.delenv("REGIME_EVALUATION_SUMMARY_PATH", raising=False)
    with pytest.raises(RuntimeError, match="REGIME_EVALUATION_SUMMARY_PATH"):
        module._durable_evidence_path(root, "REGIME_EVALUATION_SUMMARY_PATH")

    monkeypatch.setenv("REGIME_EVALUATION_SUMMARY_PATH", str(root / "summary.json"))
    with pytest.raises(RuntimeError, match="outside the repository"):
        module._durable_evidence_path(root, "REGIME_EVALUATION_SUMMARY_PATH")

    external = tmp_path.parent / "xetra-audit-evidence"
    monkeypatch.setenv("REGIME_EVALUATION_SUMMARY_PATH", str(external / "summary.json"))
    assert (
        module._durable_evidence_path(root, "REGIME_EVALUATION_SUMMARY_PATH")
        == (external / "summary.json").resolve()
    )
