from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.evaluations.final_v4_grid import evaluate_final_v4_grid
from market_regime_engine.feature_discovery.contracts import FINAL_CANDIDATE_IDS
from market_regime_engine.profiles.loader import load_profile

HASH = "a" * 64
FEATURES = ("f0", "f1")


def source_rows(row_count: int = 1323) -> pd.DataFrame:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    index = np.arange(row_count, dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp_m1": tuple(start + timedelta(days=int(value)) for value in index),
            "f0": np.sin(index / 11.0),
            "f1": np.cos(index / 17.0),
            "unused": index,
        }
    )


def fake_runner(frame, plan, profile, candidate, candidate_adapter_factory):
    del frame, profile, candidate_adapter_factory
    fold = SimpleNamespace(
        fold_id=plan.folds[0].fold_id,
        oos_predictive_log_likelihood_per_observation=float(candidate.state_count),
        bic=float(candidate.state_count),
        aic=float(candidate.state_count),
    )
    return SimpleNamespace(
        candidate_id=candidate.candidate_id,
        state_count=candidate.state_count,
        source_build_id=candidate.source_build_id,
        feature_order=candidate.feature_order,
        feature_selection_definition_hash=candidate.feature_selection_definition_hash,
        feature_selection_execution_hash=candidate.feature_selection_execution_hash,
        evaluation_plan_hash=plan.plan_hash,
        folds=(fold,),
        valid_folds=(fold,),
    )


def invalid_runner(frame, plan, profile, candidate, candidate_adapter_factory):
    del frame, profile, candidate_adapter_factory
    fold = SimpleNamespace(fold_id=plan.folds[0].fold_id, valid=False)
    return SimpleNamespace(
        candidate_id=candidate.candidate_id,
        state_count=candidate.state_count,
        source_build_id=candidate.source_build_id,
        feature_order=candidate.feature_order,
        feature_selection_definition_hash=candidate.feature_selection_definition_hash,
        feature_selection_execution_hash=candidate.feature_selection_execution_hash,
        evaluation_plan_hash=plan.plan_hash,
        folds=(fold,),
        valid_folds=(),
    )


def inputs() -> tuple[pd.DataFrame, object, object]:
    rows = source_rows()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    return rows, profile, plan


def test_final_grid_runs_exact_ordered_12_candidates_on_one_feature_contract() -> None:
    rows, profile, plan = inputs()
    result = evaluate_final_v4_grid(
        rows,
        feature_order=FEATURES,
        original_feature_universe=("f0", "f1", "unused"),
        profile=profile,
        plan=plan,
        source_build_id="build-1",
        feature_selection_definition_hash=HASH,
        feature_selection_execution_hash=HASH,
        runner=fake_runner,
        max_workers=1,
    )

    assert tuple(item.candidate_id for item in result.grid.evaluations) == FINAL_CANDIDATE_IDS
    assert tuple(item.candidate_id for item in result.grid.aggregates) == FINAL_CANDIDATE_IDS
    assert result.selection is not None
    assert result.selection.champion_candidate_id == "gaussian_hmm_k5_full"
    assert result.no_champion_reason is None
    assert all(item.feature_order == FEATURES for item in result.grid.evaluations)


def test_final_grid_returns_explicit_no_champion_when_every_candidate_fails() -> None:
    rows, profile, plan = inputs()
    result = evaluate_final_v4_grid(
        rows,
        feature_order=FEATURES,
        profile=profile,
        plan=plan,
        source_build_id="build-1",
        runner=invalid_runner,
        max_workers=1,
    )

    assert result.selection is None
    assert result.no_champion_reason == "no candidate passes statistical hard gates"


@pytest.mark.parametrize("feature_order", [("f0",), ("f0", "missing")])
def test_final_grid_rejects_invalid_selected_feature_contract(
    feature_order: tuple[str, ...],
) -> None:
    rows, profile, plan = inputs()
    with pytest.raises(ValueError):
        evaluate_final_v4_grid(
            rows,
            feature_order=feature_order,
            profile=profile,
            plan=plan,
            source_build_id="build-1",
            runner=fake_runner,
            max_workers=1,
        )
