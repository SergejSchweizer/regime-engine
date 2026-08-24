from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from market_regime_engine.evaluation.selection import select_statistical_champion
from market_regime_engine.evaluation.walk_forward import (
    WalkForwardEvaluation,
    WalkForwardFoldResult,
)
from market_regime_engine.training.candidate_grid import (
    CandidateAggregate,
    CandidateGridEvaluation,
)


def fold() -> WalkForwardFoldResult:
    return WalkForwardFoldResult(
        fold_id="fold_001",
        fold_index=1,
        valid=False,
        failure_reason="fixture",
        train_source_observation_count=1260,
        test_source_observation_count=63,
        train_model_observation_count=1260,
        test_model_observation_count=63,
        skipped_train_incomplete_count=0,
        skipped_test_incomplete_count=0,
    )


def evaluation(k: int) -> WalkForwardEvaluation:
    return WalkForwardEvaluation(
        profile_id="xetra",
        profile_config_version=1,
        candidate_id=f"gaussian_hmm_k{k}_full",
        state_count=k,
        source_build_id="build-1",
        feature_order=("f0",),
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        evaluation_plan_hash="c" * 64,
        evaluation_cutoff=datetime(2026, 1, 1, tzinfo=UTC),
        folds=(fold(),),
    )


def aggregate(
    k: int,
    *,
    mean: float = -1.0,
    std: float = 0.2,
    worst: float = -1.5,
    bic: float = 100.0,
    aic: float = 90.0,
    rate: float = 1.0,
) -> CandidateAggregate:
    valid = 10 if rate == 1.0 else 7
    planned = 10
    return CandidateAggregate(
        candidate_id=f"gaussian_hmm_k{k}_full",
        state_count=k,
        planned_fold_count=planned,
        valid_fold_count=valid,
        invalid_fold_count=planned - valid,
        valid_fold_rate=valid / planned,
        passes_valid_fold_rate_gate=valid / planned >= 0.80,
        oos_predictive_loglik_mean=mean,
        oos_predictive_loglik_std=std,
        oos_predictive_loglik_worst_fold=worst,
        oos_predictive_loglik_best_fold=-0.5,
        bic_mean=bic,
        aic_mean=aic,
    )


def grid(items: tuple[CandidateAggregate, ...]) -> CandidateGridEvaluation:
    return CandidateGridEvaluation(
        profile_id="xetra",
        profile_config_version=1,
        source_build_id="build-1",
        feature_order=("f0",),
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        evaluation_plan_hash="c" * 64,
        evaluations=tuple(evaluation(k) for k in (2, 3, 4)),
        aggregates=items,
    )


def base() -> tuple[CandidateAggregate, ...]:
    return tuple(aggregate(k) for k in (2, 3, 4))


def with_candidate(
    items: tuple[CandidateAggregate, ...],
    k: int,
    **changes: float,
) -> tuple[CandidateAggregate, ...]:
    return tuple(replace(item, **changes) if item.state_count == k else item for item in items)


def champion(items: tuple[CandidateAggregate, ...]) -> str:
    return select_statistical_champion(grid(items)).champion_candidate_id


def test_ranking_stage_1_prefers_higher_oos_mean() -> None:
    assert champion(with_candidate(base(), 4, oos_predictive_loglik_mean=-0.9)) == (
        "gaussian_hmm_k4_full"
    )


def test_ranking_stage_2_prefers_lower_population_std_with_numeric_tolerance() -> None:
    items = with_candidate(base(), 3, oos_predictive_loglik_mean=-1.0 + 5e-13)
    items = with_candidate(items, 3, oos_predictive_loglik_std=0.1)
    assert champion(items) == "gaussian_hmm_k3_full"


def test_ranking_stage_3_prefers_higher_worst_fold() -> None:
    items = with_candidate(base(), 4, oos_predictive_loglik_worst_fold=-1.4)
    assert champion(items) == "gaussian_hmm_k4_full"


def test_ranking_stage_4_prefers_lower_bic() -> None:
    items = with_candidate(base(), 3, bic_mean=99.0)
    assert champion(items) == "gaussian_hmm_k3_full"


def test_ranking_stage_5_prefers_lower_aic() -> None:
    items = with_candidate(base(), 4, aic_mean=89.0)
    assert champion(items) == "gaussian_hmm_k4_full"


def test_ranking_stage_6_prefers_fewer_states() -> None:
    assert champion(base()) == "gaussian_hmm_k2_full"


def test_hard_gate_rejects_candidate_below_80_percent_and_records_reason() -> None:
    items = list(base())
    items[0] = CandidateAggregate(
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        planned_fold_count=10,
        valid_fold_count=7,
        invalid_fold_count=3,
        valid_fold_rate=0.7,
        passes_valid_fold_rate_gate=False,
        oos_predictive_loglik_mean=-0.1,
        oos_predictive_loglik_std=0.1,
        oos_predictive_loglik_worst_fold=-0.2,
        oos_predictive_loglik_best_fold=0.0,
        bic_mean=1.0,
        aic_mean=1.0,
    )
    result = select_statistical_champion(grid(tuple(items)))
    assert result.champion_candidate_id == "gaussian_hmm_k3_full"
    rejected = result.evidence[0]
    assert rejected.accepted is False
    assert rejected.rank is None
    assert "valid-fold rate below 0.80" in rejected.rejection_reasons


def test_zero_eligible_candidates_fails_closed() -> None:
    rejected = tuple(
        CandidateAggregate(
            candidate_id=f"gaussian_hmm_k{k}_full",
            state_count=k,
            planned_fold_count=10,
            valid_fold_count=0,
            invalid_fold_count=10,
            valid_fold_rate=0.0,
            passes_valid_fold_rate_gate=False,
            oos_predictive_loglik_mean=None,
            oos_predictive_loglik_std=None,
            oos_predictive_loglik_worst_fold=None,
            oos_predictive_loglik_best_fold=None,
            bic_mean=None,
            aic_mean=None,
        )
        for k in (2, 3, 4)
    )
    with pytest.raises(ValueError, match="no candidate passes"):
        select_statistical_champion(grid(rejected))
