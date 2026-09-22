from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pandas as pd  # type: ignore[import-untyped]
import pytest

from market_regime_engine.feature_discovery import monthly_refit as module
from market_regime_engine.feature_discovery.feature_roles import CORE_FEATURES, TEMPORAL_KEY
from market_regime_engine.feature_discovery.metadata_store import FeatureSelectionMetadataStore
from market_regime_engine.profiles.loader import load_profile
from tests.unit.feature_discovery.test_pr498_monthly_refit import (
    _catalog,
    _fake_pipeline,
    _fake_quality,
    _source,
    _Tracking,
)


def _install_fake_stages(monkeypatch: pytest.MonkeyPatch, *, failure: str | None = None) -> None:
    names = CORE_FEATURES

    def quality(*_args: object, **_kwargs: object) -> SimpleNamespace:
        if failure == "quality":
            raise ValueError("quality stage failed")
        return _fake_quality(names)

    def pipeline(*_args: object, **_kwargs: object) -> SimpleNamespace:
        if failure == "pipeline":
            raise ValueError("pipeline stage failed")
        return _fake_pipeline(names)

    monkeypatch.setattr(module, "filter_outer_train_quality", quality)
    monkeypatch.setattr(module, "run_canonical_feature_selection", pipeline)


def _run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source: pd.DataFrame | None = None,
    max_workers: int | None = 1,
    failure: str | None = None,
    tracker: _Tracking | None = None,
    metadata_store: FeatureSelectionMetadataStore | None = None,
) -> module.MonthlyRefitResult:
    _install_fake_stages(monkeypatch, failure=failure)
    source = _source() if source is None else source
    profile = load_profile("configs/profiles/xetra_v4.yaml")

    def fit(frame: pd.DataFrame, selected: tuple[str, ...], _state_count: int) -> str:
        assert set(selected) <= set(CORE_FEATURES) | {
            name for name in selected if name.startswith("family_pc_")
        }
        if failure == "fit":
            raise ValueError("fit stage failed")
        assert frame[TEMPORAL_KEY].iloc[-1] < source[TEMPORAL_KEY].iloc[-1]
        return "1" * 64

    def outer_test(
        _train: pd.DataFrame,
        test: pd.DataFrame,
        _selected: tuple[str, ...],
        _state_count: int,
        _model_hashes: tuple[str, ...],
    ) -> str:
        if failure == "outer_test":
            raise ValueError("outer TEST stage failed")
        assert test[TEMPORAL_KEY].iloc[0] > _train[TEMPORAL_KEY].iloc[-1]
        return "2" * 64

    return module.run_monthly_outer_refit(
        source,
        catalog=_catalog(),
        profile=profile,
        evaluate_subset=lambda _: None,
        evaluate_hmm_subset=lambda _: None,
        hmm_selector_contract_hash="3" * 64,
        fit_final_hmm=fit,
        evaluate_gaussian_subset_by_k=lambda _state_count, _features: None,
        evaluate_outer_test=outer_test,
        max_workers=max_workers,
        tracking=tracker,  # type: ignore[arg-type]
        metadata_store=metadata_store,
    )


def test_two_consecutive_closed_months_have_complete_oos_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run(monkeypatch)
    assert len(result.folds) >= 2
    assert all(item.valid and item.package is not None for item in result.folds[:2])
    assert all(
        item.package.available_for_test_month == item.fold.test_calendar_month
        for item in result.folds[:2]
        if item.package is not None
    )


def test_mutating_month_m_plus_one_cannot_change_month_m_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _run(monkeypatch)
    first = original.folds[0]
    assert first.package is not None
    mutated = _source()
    mutated.loc[
        mutated[TEMPORAL_KEY] > pd.Timestamp(first.fold.test_last_timestamp), CORE_FEATURES[0]
    ] += 10000.0
    changed = _run(monkeypatch, source=mutated)
    assert changed.folds[0].package is not None
    assert changed.folds[0].package.package_hash == first.package.package_hash


def test_midmonth_resolution_uses_latest_frozen_package_without_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run(monkeypatch)
    first = result.folds[0]
    assert first.package is not None
    request = first.fold.test_first_timestamp + timedelta(days=10)
    request_month = request.strftime("%Y-%m")
    eligible = [
        item
        for item in result.folds
        if item.package is not None and item.package.available_for_test_month <= request_month
    ]
    assert eligible
    assert eligible[-1].package is first.package


@pytest.mark.parametrize("failure", ("quality", "pipeline", "fit", "outer_test"))
def test_failure_at_each_stage_has_no_later_package(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    result = _run(monkeypatch, failure=failure)
    assert not result.folds[0].valid
    assert result.folds[0].package is None
    assert failure.replace("_", " ") in (result.folds[0].failure_reason or "").lower()


def test_stage_runs_share_one_parent_and_serial_parallel_hashes_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serial = _run(monkeypatch, max_workers=1)
    parallel = _run(monkeypatch, max_workers=86)
    assert serial.result_hash == parallel.result_hash
    tracker = _Tracking()
    tracked = _run(monkeypatch, tracker=tracker)
    assert tracked.valid_folds
    assert tracker.starts[0] == ("run-0", None)
    assert all(parent == "run-0" for _, parent in tracker.starts[1:])
    assert len(tracker.artifacts) == len(tracker.starts) - 1


def test_duckdb_package_and_mlflow_stage_identities_reconcile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tracker = _Tracking()
    store = FeatureSelectionMetadataStore(tmp_path)
    result = _run(monkeypatch, tracker=tracker, metadata_store=store)
    assert result.valid_folds
    with duckdb.connect(str(store.database), read_only=True) as connection:
        rows = connection.execute(
            "SELECT fold_id, diagnostics_json, mlflow_run_id FROM fold_model_stats"
        ).fetchall()
    assert len(rows) == len(result.valid_folds)
    for fold_id, diagnostics_json, model_run_id in rows:
        fold = next(item for item in result.valid_folds if item.fold.fold_id == fold_id)
        assert fold.package is not None
        assert fold.package.package_hash in diagnostics_json
        assert model_run_id in {run_id for run_id, _parent in tracker.starts[1:]}


def test_hmm_boundary_never_receives_raw_transformation_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, ...]] = []
    _install_fake_stages(monkeypatch)
    source = _source()
    profile = load_profile("configs/profiles/xetra_v4.yaml")

    def fit(_frame: pd.DataFrame, selected: tuple[str, ...], _state_count: int) -> str:
        seen.append(selected)
        assert all(name in CORE_FEATURES or name.startswith("family_pc_") for name in selected)
        return "4" * 64

    module.run_monthly_outer_refit(
        source,
        catalog=_catalog(),
        profile=profile,
        evaluate_subset=lambda _: None,
        evaluate_hmm_subset=lambda _: None,
        hmm_selector_contract_hash="5" * 64,
        fit_final_hmm=fit,
        evaluate_gaussian_subset_by_k=lambda _state_count, _features: None,
        evaluate_outer_test=lambda *_args: "6" * 64,
    )
    assert seen
