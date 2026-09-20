from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

import market_regime_engine.evaluations.k_champion_outer as outer_module
from market_regime_engine.evaluation.errors import RecoverableEvaluationInvalidity
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
    order = ("f0", f"feature_k{state_count}", f"feature_fold_{fold.fold_index}")
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
        validation_cutoff=fold.train_end,
        deployment_cutoff=fold.test_end,
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
    serial = run_k_champion_outer_policy(
        rows,
        plan=plan(),
        source_snapshot_id="snapshot-1",
        profile_id="xetra",
        profile_config_version=4,
        validation_cutoff=BASE + timedelta(minutes=1448),
        selector=_selection,
        evaluator=_evaluate,
        max_workers=1,
    )
    assert len(result.outer_folds) == 12
    assert tuple(item.slot_id for item in result.outer_folds[:4]) == ("k2", "k3", "k4", "k5")
    assert all(item.valid for item in result.outer_folds)
    assert all(item.eligible for item in result.slots)
    assert all(item.valid_fold_count == 3 for item in result.slots)
    assert result == serial
    assert result.result_hash == serial.result_hash
    assert (
        len({(item.fold_id, item.slot_id, item.feature_order_hash) for item in result.outer_folds})
        == 12
    )
    assert all(
        item.selection is not None
        and item.selection.validation_cutoff == item.train_end
        and item.selection.model_family == "gaussian_hmm"
        for item in result.outer_folds
    )


def test_future_test_mutation_cannot_change_pre_test_selection() -> None:
    timestamps = tuple(BASE + timedelta(minutes=index) for index in range(1449))
    rows = pd.DataFrame(
        {
            "timestamp_m1": timestamps,
            "f0": tuple(float(index) for index in range(1449)),
            "f1": tuple(float(1449 - index) for index in range(1449)),
        }
    )
    mutated = rows.copy(deep=True)
    mutated.loc[mutated["timestamp_m1"] >= plan().folds[-1].test_start, "f0"] = -1.0e12
    kwargs = dict(
        plan=plan(),
        source_snapshot_id="snapshot-1",
        profile_id="xetra",
        profile_config_version=4,
        validation_cutoff=BASE + timedelta(minutes=1448),
        selector=_selection,
        evaluator=_evaluate,
        max_workers=1,
    )

    baseline = run_k_champion_outer_policy(rows, **kwargs)
    changed = run_k_champion_outer_policy(mutated, **kwargs)

    assert tuple(item.selection_hash for item in baseline.outer_folds) == tuple(
        item.selection_hash for item in changed.outer_folds
    )
    assert tuple(item.feature_order_hash for item in baseline.outer_folds) == tuple(
        item.feature_order_hash for item in changed.outer_folds
    )


def test_slot_gates_and_independent_aggregation_reconcile_exactly() -> None:
    timestamps = tuple(BASE + timedelta(minutes=index) for index in range(1449))
    rows = pd.DataFrame({"timestamp_m1": timestamps, "f0": range(1449)})

    calls: list[tuple[str, str]] = []

    def selective_failure(train_rows, test_rows, *, selection, slot_id, fold):
        calls.append((slot_id, fold.fold_id))
        if (slot_id, fold.fold_index) in {("k2", 1), ("k3", 3)}:
            raise RecoverableEvaluationInvalidity(f"fixture failure {slot_id}/{fold.fold_id}")
        return _evaluate(
            train_rows,
            test_rows,
            selection=selection,
            slot_id=slot_id,
            fold=fold,
        )

    result = run_k_champion_outer_policy(
        rows,
        plan=plan(),
        source_snapshot_id="snapshot-1",
        profile_id="xetra",
        profile_config_version=4,
        validation_cutoff=BASE + timedelta(minutes=1448),
        selector=_selection,
        evaluator=selective_failure,
        max_workers=1,
    )
    assert len(calls) == 12
    assert len(set(calls)) == 12
    slots = {item.slot_id: item for item in result.slots}
    assert slots["k2"].rejection_reasons == (
        "valid-fold rate below 0.80",
        "fewer than three valid outer folds",
    )
    assert slots["k3"].rejection_reasons == (
        "valid-fold rate below 0.80",
        "fewer than three valid outer folds",
        "latest complete outer fold is invalid",
    )
    assert slots["k4"].eligible and slots["k5"].eligible

    for slot_id, summary in slots.items():
        evidence = tuple(item for item in result.outer_folds if item.slot_id == slot_id)
        valid = tuple(item for item in evidence if item.valid)
        assert summary.valid_fold_count == len(valid)
        assert summary.valid_fold_rate == len(valid) / len(evidence)
        assert summary.latest_complete_fold_valid == evidence[-1].valid
        if valid:
            assert summary.mean_soft_regime_nmi == sum(
                item.soft_regime_nmi for item in valid if item.soft_regime_nmi is not None
            ) / len(valid)
            assert summary.worst_fold_soft_regime_nmi == min(
                item.soft_regime_nmi for item in valid if item.soft_regime_nmi is not None
            )
            assert summary.mean_common_support == sum(
                item.common_support for item in valid if item.common_support is not None
            ) / len(valid)
            assert summary.mean_stability == sum(
                item.stability for item in valid if item.stability is not None
            ) / len(valid)
            assert summary.fold_result_hashes == tuple(item.result_hash for item in evidence)


@pytest.mark.parametrize(
    ("value", "message"),
    [(None, "timezone-aware UTC"), (BASE.replace(tzinfo=None), "timezone-aware UTC")],
)
def test_outer_contract_scalar_helpers_reject_invalid_values(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        outer_module._utc(value, "timestamp")
    with pytest.raises(ValueError, match="non-empty trimmed"):
        outer_module._text(" ", "identity")
    with pytest.raises(ValueError, match="finite and in"):
        outer_module._unit(value if value is not None else True, "score")


def test_outer_contract_timestamp_and_probability_helpers_are_strict() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        outer_module._validate_timestamps((), "candidate OOS")
    with pytest.raises(ValueError, match="strictly increasing"):
        outer_module._validate_timestamps((BASE + timedelta(days=1), BASE), "candidate OOS")
    assert outer_module._validate_timestamps((BASE,), "candidate OOS") == (BASE,)

    with pytest.raises(ValueError, match="wrong state dimension"):
        outer_module._validate_probabilities(((1.0,),), 2, "candidate OOS")
    with pytest.raises(ValueError, match="finite and normalized"):
        outer_module._validate_probabilities(((0.7, 0.7),), 2, "candidate OOS")
    assert outer_module._validate_probabilities(((0.25, 0.75),), 2, "candidate OOS") == (
        (0.25, 0.75),
    )


def test_fold_evaluation_contract_rejects_alignment_and_stability_drift() -> None:
    with pytest.raises(ValueError, match="do not align"):
        outer_module.KChampionFoldEvaluation((BASE,), (), (BASE,), (), 0.5)
    with pytest.raises(ValueError, match="stability"):
        outer_module.KChampionFoldEvaluation((BASE,), ((0.5, 0.5),), (BASE,), ((0.5, 0.5),), 2.0)
