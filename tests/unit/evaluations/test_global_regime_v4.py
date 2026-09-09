from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

import market_regime_engine.evaluations.global_regime_v4 as global_v4
from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluation.walk_forward import AdapterFactory, WalkForwardEvaluation
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan
from market_regime_engine.evaluations.teacher_reference import FrozenTeacherRefit
from market_regime_engine.feature_discovery.contracts import ProvisionalTeacherReference
from market_regime_engine.features.ports import (
    FeatureCatalogEntry,
    FeatureCatalogSnapshot,
)
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile

START = datetime(2020, 1, 1, tzinfo=UTC)
HASH = "a" * 64


def _catalog() -> FeatureCatalogSnapshot:
    lineage = SourceLineage(
        source_dataset="xetra_gold",
        source_build_id="build-1",
        data_sha256=HASH,
        schema_version=2,
        feature_version=4,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=START,
        row_count=1449,
        min_timestamp=START,
        max_timestamp=START + timedelta(days=1448),
    )
    entries = tuple(
        FeatureCatalogEntry(name, index + 1) for index, name in enumerate(("f0", "f1", "f2"))
    )
    return FeatureCatalogSnapshot.from_entries(lineage, "timestamp_m1", entries)


def _rows(count: int = 1449) -> pd.DataFrame:
    index = np.arange(count, dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp_m1": tuple(START + timedelta(days=int(value)) for value in index),
            "f0": np.sin(index / 17.0),
            "f1": np.cos(index / 23.0),
            "f2": np.sin(index / 7.0) + index / 1000.0,
        }
    )


def _candidate(catalog: FeatureCatalogSnapshot) -> ResolvedCandidateProfile:
    return ResolvedCandidateProfile(
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        covariance_type="full",
        feature_order=("f0", "f1"),
        feature_dimension=2,
        source_build_id=catalog.lineage.source_build_id,
        feature_selection_definition_hash=HASH,
        feature_selection_execution_hash=HASH,
        original_feature_universe=catalog.feature_names,
        preliminary_medoids=(),
        feature_contract_version=4,
    )


def _teacher_reference() -> ProvisionalTeacherReference:
    return ProvisionalTeacherReference(
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        timestamps=(START,),
        filtered_probabilities=((0.5, 0.5),),
        dominant_states=(0,),
        valid_inner_fold_ids=("fold_001",),
        source_build_id="build-1",
        inner_plan_hash=HASH,
        prototype_features=("f0",),
    )


def _teacher_refit(timestamps: tuple[datetime, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        model_artifact=GaussianHMMArtifact(
            state_count=2,
            feature_order=("f0",),
            start_probabilities=(0.5, 0.5),
            transition_matrix=((0.8, 0.2), (0.2, 0.8)),
            means=((-1.0,), (1.0,)),
            full_covariances=(((1.0,),), ((1.0,),)),
        ),
        test_timestamps=timestamps,
        test_filtered_probabilities=((0.5, 0.5),) * len(timestamps),
    )


def _model_evaluation(timestamps: tuple[datetime, ...]) -> SimpleNamespace:
    fold = SimpleNamespace(
        oos_predictive_log_likelihood_per_observation=-1.0,
        oos_timestamps=timestamps,
        oos_filtered_probabilities=((0.5, 0.5),) * len(timestamps),
    )
    return SimpleNamespace(valid_folds=(fold,))


def test_outer_policy_passes_only_train_rows_to_each_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows()
    catalog = _catalog()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    candidate = _candidate(catalog)
    seen_lengths: list[int] = []

    def select(train_rows: pd.DataFrame, **kwargs: object) -> SimpleNamespace:
        del kwargs
        seen_lengths.append(len(train_rows))
        return SimpleNamespace(
            final_candidate=candidate,
            teacher_reference=_teacher_reference(),
            catalog_hash=catalog.catalog_hash,
            feature_discovery_hash=HASH,
        )

    def outer_runner(
        source_rows: pd.DataFrame,
        plan: WalkForwardPlan,
        profile: ModelProfile,
        candidate: ResolvedCandidateProfile,
        candidate_adapter_factory: AdapterFactory,
    ) -> WalkForwardEvaluation:
        del profile, candidate, candidate_adapter_factory
        fold = plan.folds[0]
        timestamps = tuple(source_rows["timestamp_m1"].iloc[-63:])
        assert timestamps[-1] == fold.test_end
        return cast(WalkForwardEvaluation, _model_evaluation(timestamps))

    monkeypatch.setattr(global_v4, "select_v4_configuration", select)
    result = global_v4.evaluate_global_regime_v4(
        rows,
        catalog=catalog,
        profile=profile,
        outer_runner=outer_runner,
        teacher_refitter=cast(
            Callable[..., FrozenTeacherRefit],
            lambda train, test, **kwargs: _teacher_refit(tuple(test["timestamp_m1"])),
        ),
    )

    assert seen_lengths == [1260, 1323, 1386]
    assert len(result.outer_folds) == 3
    assert result.valid_fold_count == 3
    assert result.production_eligible is True


def test_failed_outer_selection_does_not_reuse_a_previous_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows()
    catalog = _catalog()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    calls = 0

    def fail_selection(train_rows: pd.DataFrame, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise ValueError(f"synthetic selection failure {calls}")

    monkeypatch.setattr(global_v4, "select_v4_configuration", fail_selection)
    result = global_v4.evaluate_global_regime_v4(rows, catalog=catalog, profile=profile)

    assert calls == 3
    assert result.valid_fold_count == 0
    assert result.production_eligible is False
    assert all(not fold.valid for fold in result.outer_folds)
    assert (
        len({fold.final_configuration.feature_discovery_hash for fold in result.outer_folds}) == 3
    )
    assert all(
        "TRAIN-only v4 selection failed" in (fold.failure_reason or "")
        for fold in result.outer_folds
    )


def test_train_snapshot_rejects_test_only_column_and_keeps_catalog_order() -> None:
    catalog = _catalog()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    train = _rows(20).drop(columns=["f2"])

    with pytest.raises(ValueError, match="missing catalog columns"):
        global_v4.select_v4_configuration(train, catalog=catalog, profile=profile)

    snapshot = global_v4._as_feature_snapshot(_rows(20), catalog)
    assert snapshot.feature_names == ("f0", "f1", "f2")
    assert snapshot.rows[0].timestamp == START
