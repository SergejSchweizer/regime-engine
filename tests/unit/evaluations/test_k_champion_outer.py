from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from market_regime_engine.evaluation.walk_forward_splits import WalkForwardFold, WalkForwardPlan
from market_regime_engine.evaluations.k_champion_contract import (
    KChampionSelection,
    feature_order_hash,
)
from market_regime_engine.evaluations.k_champion_outer import (
    KChampionFoldEvaluation,
    run_k_champion_outer_policy,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def plan() -> WalkForwardPlan:
    folds = tuple(
        WalkForwardFold(
            fold_index=index,
            fold_id=f"fold_{index:03d}",
            train_start=BASE,
            train_end=BASE + timedelta(minutes=1260 + (index - 1) * 63 - 1),
            test_start=BASE + timedelta(minutes=1260 + (index - 1) * 63),
            test_end=BASE + timedelta(minutes=1260 + index * 63 - 1),
            train_source_observations=1260 + (index - 1) * 63,
            test_source_observations=63,
        )
        for index in (1, 2, 3)
    )
    return WalkForwardPlan(folds=folds, evaluation_cutoff=folds[-1].test_end, plan_hash="b" * 64)


def _selection(
    train_rows: pd.DataFrame, *, slot_id: str, fold: WalkForwardFold
) -> KChampionSelection:
    del train_rows
    state_count = int(slot_id[1:])
    order = ("f0", "f1")
    return KChampionSelection(
        slot_id=slot_id,
        state_count=state_count,
        model_family="gaussian_hmm",
        candidate_identity=f"gaussian_hmm_k{state_count}_full",
        feature_order=order,
        feature_order_hash=feature_order_hash(order),
        source_snapshot_id="snapshot-1",
        profile_id="xetra",
        profile_config_version=4,
        policy_id="k_champion_portfolio",
        policy_version="k_champion_portfolio.v1",
        validation_cutoff=BASE + timedelta(minutes=1448),
        deployment_cutoff=BASE + timedelta(minutes=1449),
        comparison_domain_id="k_slot_promotion.v1",
        promotion_score_version="k_slot_promotion.v1",
        reference_teacher_id=f"teacher-{state_count}",
        artifact_hash="a" * 64,
    )


def _evaluate(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    *,
    selection: KChampionSelection,
    slot_id: str,
    fold: WalkForwardFold,
) -> KChampionFoldEvaluation:
    del train_rows, test_rows, slot_id, fold
    k = selection.state_count
    rows = tuple(tuple(0.7 / (k - 1) if state else 0.3 for state in range(k)) for _ in range(3))
    teacher = tuple(tuple(1.0 / k for _ in range(k)) for _ in range(3))
    times = tuple(BASE + timedelta(hours=index) for index in range(3))
    return KChampionFoldEvaluation(times, rows, times, teacher, 0.8)


def test_four_slots_are_independent_and_canonically_ordered() -> None:
    timestamps = tuple(BASE + timedelta(minutes=index) for index in range(1449))
    rows = pd.DataFrame(
        {
            "timestamp_m1": timestamps,
            "f0": tuple(float(index) for index in range(1449)),
            "f1": tuple(float(1449 - index) for index in range(1449)),
        }
    )
    result = run_k_champion_outer_policy(
        rows,
        plan=plan(),
        source_snapshot_id="snapshot-1",
        profile_id="xetra",
        profile_config_version=4,
        validation_cutoff=BASE + timedelta(minutes=1448),
        selector=_selection,
        evaluator=_evaluate,
        max_workers=2,
    )
    assert len(result.outer_folds) == 12
    assert tuple(item.slot_id for item in result.outer_folds[:4]) == ("k2", "k3", "k4", "k5")
    assert all(item.valid for item in result.outer_folds)
    assert all(item.eligible for item in result.slots)
    assert all(item.valid_fold_count == 3 for item in result.slots)
