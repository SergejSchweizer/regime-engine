from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import log, pi

import numpy as np
import pandas as pd
import pytest

import market_regime_engine.evaluations.provisional_teacher as module
from market_regime_engine.evaluation.selection import (
    CandidateSelectionEvidence,
    StatisticalChampionSelection,
)
from market_regime_engine.evaluation.walk_forward import (
    WalkForwardEvaluation,
    WalkForwardFoldResult,
)
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan
from market_regime_engine.feature_discovery.contracts import (
    V4_PROVISIONAL_STATE_COUNTS,
)
from market_regime_engine.inference.predictive_likelihood import (
    continued_test_predictive_likelihood,
)
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.profiles.loader import load_profile

HASH = "a" * 64
FEATURES = ("prototype_0", "prototype_1")


def source_rows(row_count: int = 819) -> pd.DataFrame:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    index = np.arange(row_count, dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp_m1": tuple(start + timedelta(days=int(value)) for value in index),
            FEATURES[0]: np.sin(index / 11.0),
            FEATURES[1]: np.cos(index / 17.0),
        }
    )


def invalid_evaluation(
    rows: pd.DataFrame,
    plan: WalkForwardPlan,
    candidate,
) -> WalkForwardEvaluation:
    return WalkForwardEvaluation(
        profile_id="xetra",
        profile_config_version=4,
        candidate_id=candidate.candidate_id,
        state_count=candidate.state_count,
        source_build_id="build-1",
        feature_order=candidate.feature_order,
        feature_selection_definition_hash=HASH,
        feature_selection_execution_hash=HASH,
        evaluation_plan_hash=plan.plan_hash,
        evaluation_cutoff=plan.evaluation_cutoff,
        folds=tuple(
            WalkForwardFoldResult(
                fold_id=fold.fold_id,
                fold_index=fold.fold_index,
                valid=False,
                failure_reason="test runner deliberately invalid",
                train_source_observation_count=fold.train_source_observations,
                test_source_observation_count=fold.test_source_observations,
                train_model_observation_count=0,
                test_model_observation_count=0,
                skipped_train_incomplete_count=fold.train_source_observations,
                skipped_test_incomplete_count=fold.test_source_observations,
            )
            for fold in plan.folds
        ),
    )


def test_inner_plan_is_exact_complete_756_63_63_without_partial_test() -> None:
    plan = module.build_inner_walk_forward_plan(tuple(source_rows(1000)["timestamp_m1"]))

    assert tuple(
        (fold.train_source_observations, fold.test_source_observations) for fold in plan.folds
    ) == ((756, 63), (819, 63), (882, 63))
    assert plan.folds[0].test_start == source_rows(1000).loc[756, "timestamp_m1"]
    assert plan.folds[-1].test_end == source_rows(1000).loc[944, "timestamp_m1"]
    assert plan.evaluation_cutoff == plan.folds[-1].test_end

    no_partial = module.build_inner_walk_forward_plan(tuple(source_rows(1000 - 1)["timestamp_m1"]))
    assert len(no_partial.folds) == 3


def test_inner_plan_rejects_short_or_nonmonotonic_source() -> None:
    with pytest.raises(ValueError, match="at least one complete"):
        module.build_inner_walk_forward_plan(tuple(source_rows(818)["timestamp_m1"]))

    rows = source_rows()
    rows.loc[1, "timestamp_m1"] = rows.loc[0, "timestamp_m1"]
    with pytest.raises(ValueError, match="strictly increasing"):
        module.build_inner_walk_forward_plan(tuple(rows["timestamp_m1"]))


def test_preflight_runs_before_any_candidate_runner() -> None:
    rows = source_rows()
    rows[FEATURES[0]] = 1.0
    calls: list[str] = []
    profile = load_profile("configs/profiles/xetra_v4.yaml")

    def runner(*args):
        calls.append(args[3].candidate_id)
        raise AssertionError("HMM runner must not execute after failed preflight")

    with pytest.raises(ValueError, match="ineligible because the model clock is invalid"):
        module.select_provisional_teacher(
            rows,
            profile=profile,
            prototype_features=FEATURES,
            source_build_id="build-1",
            feature_selection_definition_hash=HASH,
            feature_selection_execution_hash=HASH,
            runner=runner,
        )
    assert calls == []


def test_teacher_runs_only_gaussian_candidates_on_one_shared_prototype_contract(
    monkeypatch,
) -> None:
    rows = source_rows()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    calls: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []

    def runner(frame, plan, shared_profile, candidate, candidate_adapter_factory):
        del candidate_adapter_factory
        calls.append(
            (
                candidate.candidate_id,
                candidate.feature_order,
                tuple(fold.fold_id for fold in plan.folds),
            )
        )
        assert frame is rows
        assert shared_profile is profile
        return invalid_evaluation(frame, plan, candidate)

    winner = "gaussian_hmm_k3_full"
    evidence = tuple(
        CandidateSelectionEvidence(
            candidate_id=f"gaussian_hmm_k{state_count}_full",
            state_count=state_count,
            accepted=True,
            rejection_reasons=(),
            rank={2: 2, 3: 1, 4: 3, 5: 4}[state_count],
        )
        for index, state_count in enumerate(V4_PROVISIONAL_STATE_COUNTS, start=1)
    )
    selection = StatisticalChampionSelection(
        champion_candidate_id=winner,
        champion_state_count=3,
        ranked_candidate_ids=(
            winner,
            "gaussian_hmm_k2_full",
            "gaussian_hmm_k4_full",
            "gaussian_hmm_k5_full",
        ),
        evidence=evidence,
        common_valid_fold_ids=(),
        common_valid_fold_count=0,
        common_valid_fold_rate=1.0,
    )
    monkeypatch.setattr(module, "rank_same_feature_candidates", lambda *_: selection)

    result = module.select_provisional_teacher(
        rows,
        profile=profile,
        prototype_features=FEATURES,
        source_build_id="build-1",
        feature_selection_definition_hash=HASH,
        feature_selection_execution_hash=HASH,
        runner=runner,
        max_workers=2,
    )

    assert {call[0] for call in calls} == {
        f"gaussian_hmm_k{state_count}_full" for state_count in V4_PROVISIONAL_STATE_COUNTS
    }
    assert all(call[1] == FEATURES for call in calls)
    assert all(call[2] == ("fold_001",) for call in calls)
    assert result.provisional_candidate_id == winner
    assert result.provisional_state_count == 3
    assert result.model_clock.status.value == "valid"
    assert all(
        evaluation.evaluation_plan_hash == result.inner_plan.plan_hash
        for evaluation in result.candidate_evaluations
    )


def test_independent_forward_recursion_matches_teacher_continuation_likelihood() -> None:
    artifact = GaussianHMMArtifact(
        state_count=2,
        feature_order=("x",),
        start_probabilities=(0.6, 0.4),
        transition_matrix=((0.8, 0.2), (0.3, 0.7)),
        means=((-1.0,), (1.0,)),
        full_covariances=(((0.5,),), ((0.75,),)),
    )
    train = np.asarray(((-0.8,), (0.2,), (1.1,)), dtype=np.float64)
    test = np.asarray(((-0.4,), (0.9,)), dtype=np.float64)

    def log_density(value: float, mean: float, variance: float) -> float:
        return -0.5 * (log(2.0 * pi * variance) + (value - mean) ** 2 / variance)

    def forward(rows: np.ndarray, initial: np.ndarray | None) -> tuple[float, np.ndarray]:
        alpha = initial
        total = 0.0
        for row in rows:
            prior = (
                np.asarray(artifact.start_probabilities, dtype=np.float64)
                if alpha is None
                else alpha @ np.asarray(artifact.transition_matrix, dtype=np.float64)
            )
            weights = prior * np.exp(
                np.asarray(
                    [
                        log_density(float(row[0]), -1.0, 0.5),
                        log_density(float(row[0]), 1.0, 0.75),
                    ],
                    dtype=np.float64,
                )
            )
            normalizer = float(weights.sum())
            total += log(normalizer)
            alpha = weights / normalizer
        assert alpha is not None
        return total, alpha

    _, terminal_train = forward(train, None)
    expected_test_log_likelihood, _ = forward(test, terminal_train)
    actual = continued_test_predictive_likelihood(train, test, artifact)

    assert actual.test_log_likelihood == pytest.approx(expected_test_log_likelihood)
    assert actual.test_log_likelihood_per_observation == pytest.approx(
        expected_test_log_likelihood / len(test)
    )
