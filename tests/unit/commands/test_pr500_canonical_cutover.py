from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd

import market_regime_engine.commands.canonical_xetra as module


def _callbacks() -> module.CanonicalStageCallbacks:
    def noop(*_args):
        return None

    return module.CanonicalStageCallbacks(
        evaluate_subset=noop,
        evaluate_hmm_subset=noop,
        hmm_selector_contract_hash="a" * 64,
        fit_final_hmm=lambda *_args: "b" * 64,
        evaluate_gaussian_subset_by_k=noop,
        evaluate_outer_test=lambda *_args: "c" * 64,
    )


def test_public_xetra_boundary_forwards_all_stages_to_monthly_runner(monkeypatch) -> None:
    captured: dict[str, object] = {}
    package = SimpleNamespace(
        feature_selection_profile_hash="d" * 64,
        selected_features=("vix_log_level",),
    )
    fold = SimpleNamespace(package=package)
    monthly = SimpleNamespace(
        source_build_id="source-build",
        result_hash="e" * 64,
        folds=(fold,),
        valid_folds=(fold,),
    )

    def fake_runner(rows, **kwargs):
        captured["rows"] = rows
        captured.update(kwargs)
        return monthly

    monkeypatch.setattr(module, "run_monthly_outer_refit", fake_runner)
    catalog = SimpleNamespace(lineage=SimpleNamespace(source_build_id="source-build"))
    callbacks = _callbacks()
    rows = pd.DataFrame(
        {
            "timestamp_m1": (datetime(2024, 1, 1, tzinfo=UTC),),
            "vix_log_level": (1.0,),
        }
    )

    result = module.run_canonical_xetra_evaluation(
        rows,
        catalog=catalog,
        profile=SimpleNamespace(),
        callbacks=callbacks,
        metadata_store=object(),
        max_workers=86,
    )

    assert captured["rows"] is rows
    assert captured["catalog"] is catalog
    assert captured["metadata_store"] is not None
    assert captured["evaluate_subset"] is callbacks.evaluate_subset
    assert captured["evaluate_hmm_subset"] is callbacks.evaluate_hmm_subset
    assert captured["evaluate_gaussian_subset_by_k"] is callbacks.evaluate_gaussian_subset_by_k
    assert captured["max_workers"] == 86
    assert result.feature_selection_profile_hash == "d" * 64
    assert result.selected_features_by_fold == (("vix_log_level",),)


def test_canonical_result_requires_the_source_identity_to_reconcile(monkeypatch) -> None:
    package = SimpleNamespace(
        feature_selection_profile_hash="d" * 64,
        selected_features=("vix_log_level",),
    )
    fold = SimpleNamespace(package=package)
    monthly = SimpleNamespace(
        source_build_id="different-source",
        result_hash="e" * 64,
        folds=(fold,),
        valid_folds=(fold,),
    )
    monkeypatch.setattr(module, "run_monthly_outer_refit", lambda *_args, **_kwargs: monthly)
    catalog = SimpleNamespace(lineage=SimpleNamespace(source_build_id="source-build"))
    rows = pd.DataFrame(
        {
            "timestamp_m1": (datetime(2024, 1, 1, tzinfo=UTC),),
            "vix_log_level": (1.0,),
        }
    )

    import pytest

    with pytest.raises(ValueError, match="source build"):
        module.run_canonical_xetra_evaluation(
            rows,
            catalog=catalog,
            profile=SimpleNamespace(),
            callbacks=_callbacks(),
            metadata_store=object(),
        )
