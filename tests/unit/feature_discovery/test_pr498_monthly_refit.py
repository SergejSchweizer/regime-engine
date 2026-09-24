from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd  # type: ignore[import-untyped]
import pytest

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.feature_discovery import monthly_refit as module
from market_regime_engine.feature_discovery.feature_roles import CORE_FEATURES, TEMPORAL_KEY
from market_regime_engine.feature_discovery.sffs import SFFSResult
from market_regime_engine.features.ports import FeatureCatalogEntry, FeatureCatalogSnapshot
from market_regime_engine.profiles.loader import load_profile


def _catalog() -> FeatureCatalogSnapshot:
    return FeatureCatalogSnapshot.from_entries(
        SourceLineage(
            source_dataset="macro_features",
            source_build_id="build-monthly",
            data_sha256="a" * 64,
            schema_version=6,
            feature_version=5,
            source_table="macro_loader.macro_features",
            synced_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        TEMPORAL_KEY,
        tuple(FeatureCatalogEntry(name, index + 1) for index, name in enumerate(CORE_FEATURES)),
    )


def _source() -> pd.DataFrame:
    timestamps = pd.date_range("2018-01-01", periods=1500, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            TEMPORAL_KEY: timestamps,
            **{
                name: [float(index + offset) for index in range(len(timestamps))]
                for offset, name in enumerate(CORE_FEATURES)
            },
        }
    )


def _fake_quality(names: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(result_hash="b" * 64, eligible_features=names, features=())


def _fake_pipeline(names: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        profile_hash="c" * 64,
        role_contract_hash="d" * 64,
        family_pca=(),
        global_reduction=SimpleNamespace(
            result_hash="e" * 64,
            representatives=names[:3],
            evidence=(),
        ),
        sffs=SFFSResult(names[:3], names[0], ()),
        k_sffs=(),
        selected_features=names[:3],
        ablation=SimpleNamespace(fit_execution_hashes=("f" * 64,)),
    )


class _Tracking:
    def __init__(self) -> None:
        self.starts: list[tuple[str, str | None]] = []
        self.artifacts: list[tuple[str, str]] = []
        self.ended: list[str] = []

    def start_run(self, *, run_name: str, parent_run_id: str | None = None) -> str:
        del run_name
        run_id = f"run-{len(self.starts)}"
        self.starts.append((run_id, parent_run_id))
        return run_id

    def log_params(self, _run_id: str, _params: dict[str, str]) -> None:
        return None

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str) -> None:
        self.artifacts.append((run_id, f"{local_path}:{artifact_path}"))

    def end_run(self, run_id: str) -> None:
        self.ended.append(run_id)

    def fail_run(self, _run_id: str) -> None:
        raise AssertionError("synthetic tracking should not fail")


def test_monthly_refit_uses_only_closed_train_prefix_and_freezes_package_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    catalog = _catalog()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    names = CORE_FEATURES
    observed_cutoffs: list[datetime] = []
    tracking = _Tracking()
    monkeypatch.setattr(
        module, "filter_outer_train_quality", lambda *args, **kwargs: _fake_quality(names)
    )
    monkeypatch.setattr(
        module, "run_canonical_feature_selection", lambda *args, **kwargs: _fake_pipeline(names)
    )

    def fit(frame: pd.DataFrame, selected: tuple[str, ...], state_count: int) -> str:
        assert selected == names[:3]
        assert state_count == 2
        cutoff = frame[TEMPORAL_KEY].iloc[-1].to_pydatetime()
        observed_cutoffs.append(cutoff)
        return "1" * 64

    result = module.run_monthly_outer_refit(
        source,
        catalog=catalog,
        profile=profile,
        evaluate_subset=lambda _: None,
        evaluate_hmm_subset=lambda _: None,
        hmm_selector_contract_hash="2" * 64,
        fit_final_hmm=fit,
        evaluate_gaussian_subset_by_k=lambda _state_count, _features: None,
        evaluate_outer_test=lambda *_args: "5" * 64,
        max_workers=86,
        tracking=tracking,  # type: ignore[arg-type]
    )

    assert result.folds
    assert all(item.valid for item in result.folds), [item.failure_reason for item in result.folds]
    assert tuple(item.fold.train_cutoff_timestamp for item in result.folds) == tuple(
        observed_cutoffs
    )
    assert all(
        item.package is not None
        and item.package.available_for_test_month == item.fold.test_calendar_month
        and item.package.train_cutoff == item.fold.train_cutoff_timestamp
        for item in result.folds
    )
    assert len({item.package.package_hash for item in result.folds if item.package}) == len(
        result.folds
    )
    assert tracking.starts[0] == ("run-0", None)
    assert all(parent == "run-0" for _, parent in tracking.starts[1:])
    assert len(tracking.starts) == 1 + len(result.folds) * 8
    assert len(tracking.artifacts) == len(result.folds) * 8
    assert set(tracking.ended) == {f"run-{index}" for index in range(len(tracking.starts))}
    assert tracking.ended[-1] == "run-0"


def test_failed_monthly_fit_is_invalid_and_is_not_committed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    catalog = _catalog()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    names = CORE_FEATURES
    monkeypatch.setattr(
        module, "filter_outer_train_quality", lambda *args, **kwargs: _fake_quality(names)
    )
    monkeypatch.setattr(
        module, "run_canonical_feature_selection", lambda *args, **kwargs: _fake_pipeline(names)
    )
    monkeypatch.setattr(module, "_metadata_bundle", lambda **kwargs: object())

    class Store:
        def __init__(self) -> None:
            self.commits = 0

        def commit_fold(self, bundle: object) -> bool:
            self.commits += 1
            return True

    store = Store()
    calls = 0

    def fit(_frame: pd.DataFrame, _selected: tuple[str, ...], _state_count: int) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("synthetic final fit failure")
        return "3" * 64

    result = module.run_monthly_outer_refit(
        source,
        catalog=catalog,
        profile=profile,
        evaluate_subset=lambda _: None,
        evaluate_hmm_subset=lambda _: None,
        hmm_selector_contract_hash="4" * 64,
        fit_final_hmm=fit,
        evaluate_gaussian_subset_by_k=lambda _state_count, _features: None,
        evaluate_outer_test=lambda *_args: "6" * 64,
        metadata_store=store,  # type: ignore[arg-type]
    )

    assert not result.folds[0].valid
    assert "synthetic final fit failure" in (result.folds[0].failure_reason or "")
    assert all(item.valid for item in result.folds[1:]), [
        item.failure_reason for item in result.folds
    ]
    assert store.commits == len(result.folds) - 1


def test_completed_fold_checkpoint_is_reused_for_same_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    source = _source()
    catalog = _catalog()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    names = CORE_FEATURES
    pipeline = _fake_pipeline(names)
    pipeline.role_contract_hash = module.build_feature_role_contract_from_catalog(
        catalog
    ).contract_hash
    monkeypatch.setattr(
        module, "filter_outer_train_quality", lambda *args, **kwargs: _fake_quality(names)
    )
    monkeypatch.setattr(module, "run_canonical_feature_selection", lambda *args, **kwargs: pipeline)
    monkeypatch.setattr(module, "_metadata_bundle", lambda **kwargs: object())

    class Store:
        def __init__(self) -> None:
            self.database = tmp_path / "feature_selection.duckdb"
            self.commits = 0

        def commit_fold(self, _bundle: object) -> bool:
            self.commits += 1
            return True

    store = Store()
    calls = 0

    def fit(_frame: pd.DataFrame, _selected: tuple[str, ...], _state_count: int) -> str:
        nonlocal calls
        calls += 1
        return "7" * 64

    first = module.run_monthly_outer_refit(
        source,
        catalog=catalog,
        profile=profile,
        evaluate_subset=lambda _: None,
        evaluate_hmm_subset=lambda _: None,
        hmm_selector_contract_hash="8" * 64,
        fit_final_hmm=fit,
        evaluate_gaussian_subset_by_k=lambda _state_count, _features: None,
        evaluate_outer_test=lambda *_args: "9" * 64,
        metadata_store=store,  # type: ignore[arg-type]
    )
    first_calls = calls
    second = module.run_monthly_outer_refit(
        source,
        catalog=catalog,
        profile=profile,
        evaluate_subset=lambda _: None,
        evaluate_hmm_subset=lambda _: None,
        hmm_selector_contract_hash="8" * 64,
        fit_final_hmm=lambda *_args: (_ for _ in ()).throw(
            AssertionError("checkpointed folds must not refit")
        ),
        evaluate_gaussian_subset_by_k=lambda _state_count, _features: None,
        evaluate_outer_test=lambda *_args: "9" * 64,
        metadata_store=store,  # type: ignore[arg-type]
    )

    assert first.folds and second.folds
    assert all(item.valid for item in first.folds)
    assert all(item.valid for item in second.folds)
    assert calls == first_calls
    assert store.commits == len(first.folds)
