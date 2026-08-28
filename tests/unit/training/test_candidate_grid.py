from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.evaluation.walk_forward import (
    WalkForwardEvaluation,
    WalkForwardFoldResult,
    run_walk_forward_candidate,
)
from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.inference.filtering import causal_filter
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.gaussian_hmm import (
    HmmlearnGaussianHMMAdapter,
    HmmlearnGMMHMMAdapter,
)
from market_regime_engine.models.protocols import FilterResult, FitResult
from market_regime_engine.models.student_t_hmm import StudentTHMMAdapter
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import (
    ResolvedCandidateProfile,
    ResolvedSelectedFeatureProfile,
)
from market_regime_engine.training.candidate_grid import (
    CANDIDATE_VALID_FOLD_RATE_GATE,
    CandidateAggregate,
    CandidateGridEvaluation,
    _default_adapter_builder,
    aggregate_candidate,
    evaluate_candidate_grid,
)

PROFILE_CONFIG = Path("configs/profiles/xetra_v1.yaml")
FEATURES = ("f0", "f1")
UNIVERSE = tuple(f"f{index}" for index in range(48))
MEDOIDS = tuple(f"f{index}" for index in range(8))


def candidate(state_count: int) -> ResolvedCandidateProfile:
    return ResolvedCandidateProfile(
        candidate_id=f"gaussian_hmm_k{state_count}_full",
        state_count=state_count,
        covariance_type="full",
        feature_order=FEATURES,
        feature_dimension=2,
        source_build_id="build-1",
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        original_feature_universe=UNIVERSE,
        preliminary_medoids=MEDOIDS,
    )


def resolved_profile() -> ResolvedSelectedFeatureProfile:
    return ResolvedSelectedFeatureProfile(
        profile_id="xetra",
        profile_config_version=1,
        registered_model="regime-xetra",
        source_build_id="build-1",
        original_feature_universe=UNIVERSE,
        preliminary_medoids=MEDOIDS,
        final_features=FEATURES,
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        candidates=tuple(candidate(k) for k in (2, 3, 4)),
    )


def artifact() -> GaussianHMMArtifact:
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=FEATURES,
        start_probabilities=(0.5, 0.5),
        transition_matrix=((0.8, 0.2), (0.2, 0.8)),
        means=((-1.0, -1.0), (1.0, 1.0)),
        full_covariances=(
            ((0.20, 0.02), (0.02, 0.20)),
            ((0.20, -0.02), (-0.02, 0.20)),
        ),
    )


class DeterministicAdapter:
    def __init__(self) -> None:
        self._artifact = artifact()

    def fit(self, train_rows: object, state_count: int, seed: int) -> FitResult:
        assert state_count == 2
        values = np.asarray(train_rows, dtype=np.float64)
        return FitResult(
            artifact=self._artifact,
            train_log_likelihood=causal_filter(values, self._artifact).log_likelihood,
            converged=True,
            iterations=5,
            seed=seed,
        )

    def extract(self) -> GaussianHMMArtifact:
        return self._artifact

    def reconstruct(self, model_artifact: GaussianHMMArtifact) -> None:
        self._artifact = model_artifact

    def causal_filter(
        self,
        rows: object,
        initial_filtered_probabilities: tuple[float, ...] | None = None,
    ) -> FilterResult:
        raise AssertionError("runner uses backend-independent causal filter")


def source_rows(row_count: int = 1386) -> pd.DataFrame:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    timestamps = tuple(start + timedelta(days=index) for index in range(row_count))
    signs = np.where(np.arange(row_count) % 2 == 0, -1.0, 1.0)
    return pd.DataFrame(
        {
            "timestamp_m1": timestamps,
            "f0": signs,
            "f1": signs + np.where(signs > 0.0, 0.05, -0.05),
        }
    )


def base_evaluation() -> tuple[WalkForwardEvaluation, object, object]:
    rows = source_rows()
    profile = load_profile(PROFILE_CONFIG)
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    evaluation = run_walk_forward_candidate(
        rows,
        plan=plan,
        profile=profile,
        candidate=candidate(2),
        adapter_factory=DeterministicAdapter,
    )
    return evaluation, profile, plan


def test_aggregate_uses_only_valid_fold_per_observation_metrics_and_population_std() -> None:
    evaluation, _, _ = base_evaluation()
    first, second = evaluation.folds
    assert first.oos_predictive_log_likelihood_per_observation is not None
    assert second.oos_predictive_log_likelihood_per_observation is not None
    aggregate = aggregate_candidate(evaluation)
    values = np.asarray(
        [
            first.oos_predictive_log_likelihood_per_observation,
            second.oos_predictive_log_likelihood_per_observation,
        ],
        dtype=np.float64,
    )
    assert aggregate.planned_fold_count == 2
    assert aggregate.valid_fold_count == 2
    assert aggregate.invalid_fold_count == 0
    assert aggregate.valid_fold_rate == 1.0
    assert aggregate.passes_valid_fold_rate_gate is True
    assert aggregate.oos_predictive_loglik_mean == pytest.approx(float(np.mean(values)))
    assert aggregate.oos_predictive_loglik_std == pytest.approx(float(np.std(values, ddof=0)))
    assert aggregate.oos_predictive_loglik_worst_fold == pytest.approx(float(np.min(values)))
    assert aggregate.oos_predictive_loglik_best_fold == pytest.approx(float(np.max(values)))


def test_invalid_folds_are_counted_but_excluded_and_zero_valid_has_no_scores() -> None:
    evaluation, _, _ = base_evaluation()
    invalid = WalkForwardFoldResult(
        fold_id="fold_002",
        fold_index=2,
        valid=False,
        failure_reason="deliberate invalid fold",
        train_source_observation_count=1323,
        test_source_observation_count=63,
        train_model_observation_count=1323,
        test_model_observation_count=63,
        skipped_train_incomplete_count=0,
        skipped_test_incomplete_count=0,
    )
    mixed = replace(evaluation, folds=(evaluation.folds[0], invalid))
    aggregate = aggregate_candidate(mixed)
    assert aggregate.valid_fold_count == 1
    assert aggregate.invalid_fold_count == 1
    assert aggregate.valid_fold_rate == 0.5
    assert aggregate.passes_valid_fold_rate_gate is False
    assert aggregate.oos_predictive_loglik_std == 0.0

    first_invalid = replace(
        invalid,
        fold_id="fold_001",
        fold_index=1,
        train_source_observation_count=1260,
        train_model_observation_count=1260,
    )
    all_invalid = replace(evaluation, folds=(first_invalid, invalid))
    zero = aggregate_candidate(all_invalid)
    assert zero.valid_fold_count == 0
    assert zero.oos_predictive_loglik_mean is None
    assert zero.bic_mean is None


def test_grid_runs_exact_k2_k3_k4_on_one_shared_contract() -> None:
    base, profile, plan = base_evaluation()
    calls: list[tuple[str, tuple[str, ...], str]] = []

    def runner(rows, shared_plan, shared_profile, item, adapter_factory):
        assert len(rows) == 1386
        assert shared_plan is plan
        assert shared_profile is profile
        assert callable(adapter_factory)
        calls.append((item.candidate_id, item.feature_order, item.source_build_id))
        return replace(base, candidate_id=item.candidate_id, state_count=item.state_count)

    result = evaluate_candidate_grid(
        source_rows(),
        plan=plan,
        profile=profile,
        resolved_profile=resolved_profile(),
        adapter_factory_builder=lambda item: DeterministicAdapter,
        runner=runner,
    )
    assert sorted(call[0] for call in calls) == [
        "gaussian_hmm_k2_full",
        "gaussian_hmm_k3_full",
        "gaussian_hmm_k4_full",
    ]
    assert all(call[1] == FEATURES and call[2] == "build-1" for call in calls)
    aggregate_ids = tuple(item.candidate_id for item in result.aggregates)
    assert aggregate_ids == (
        "gaussian_hmm_k2_full",
        "gaussian_hmm_k3_full",
        "gaussian_hmm_k4_full",
    )
    assert all(item.valid_fold_rate == 1.0 for item in result.aggregates)
    assert result.evaluation_plan_hash == plan.plan_hash


def test_grid_default_adapter_factory_and_profile_derived_model_families() -> None:
    base, profile, plan = base_evaluation()
    factories = []

    def runner(rows, shared_plan, shared_profile, item, adapter_factory):
        del rows, shared_plan, shared_profile
        factories.append(adapter_factory)
        return replace(base, candidate_id=item.candidate_id, state_count=item.state_count)

    evaluate_candidate_grid(
        source_rows(),
        plan=plan,
        profile=profile,
        resolved_profile=resolved_profile(),
        runner=runner,
    )

    assert all(isinstance(factory(), HmmlearnGaussianHMMAdapter) for factory in factories)
    student_profile = load_profile(Path("configs/profiles/xetra_v2.yaml"))
    student = _default_adapter_builder(
        student_profile,
        replace(
            candidate(2),
            candidate_id="student_t_hmm_k2_full",
            model_family="student_t_hmm",
        ),
    )()
    assert isinstance(student, StudentTHMMAdapter)
    assert student.settings.initial_nu == student_profile.student_t_hmm.initial_nu
    gmm = _default_adapter_builder(
        student_profile,
        replace(
            candidate(2),
            candidate_id="gmm_hmm_k2_m2_full",
            model_family="gmm_hmm",
            mixture_count=2,
        ),
    )()
    assert isinstance(gmm, HmmlearnGMMHMMAdapter)


def test_grid_rejects_an_unexpected_extra_candidate_for_v1() -> None:
    resolved = resolved_profile()
    gmm_candidate = replace(
        candidate(2),
        candidate_id="gmm_hmm_k2_m2_full",
        model_family="gmm_hmm",
        mixture_count=2,
    )
    with pytest.raises(ValueError, match="IDs/order"):
        replace(resolved, candidates=(*resolved.candidates, gmm_candidate))


def test_grid_fails_closed_on_runner_contract_drift() -> None:
    base, profile, plan = base_evaluation()

    def bad_order(rows, shared_plan, shared_profile, item, adapter_factory):
        del rows, shared_plan, shared_profile, adapter_factory
        wrong_state = 4 if item.state_count == 2 else item.state_count
        return replace(
            base,
            candidate_id=f"gaussian_hmm_k{wrong_state}_full",
            state_count=wrong_state,
        )

    with pytest.raises(ValueError, match="unexpected candidate identities/order"):
        evaluate_candidate_grid(
            source_rows(),
            plan=plan,
            profile=profile,
            resolved_profile=resolved_profile(),
            runner=bad_order,
        )

    def wrong_plan(rows, shared_plan, shared_profile, item, adapter_factory):
        del rows, shared_plan, shared_profile, adapter_factory
        return replace(
            base,
            candidate_id=item.candidate_id,
            state_count=item.state_count,
            evaluation_plan_hash="f" * 64,
        )

    with pytest.raises(ValueError, match="plan hash"):
        evaluate_candidate_grid(
            source_rows(),
            plan=plan,
            profile=profile,
            resolved_profile=resolved_profile(),
            runner=wrong_plan,
        )


def test_aggregate_contract_validates_counts_gate_and_finite_metrics() -> None:
    with pytest.raises(ValueError, match="reconcile"):
        CandidateAggregate(
            candidate_id="gaussian_hmm_k2_full",
            state_count=2,
            planned_fold_count=2,
            valid_fold_count=1,
            invalid_fold_count=0,
            valid_fold_rate=0.5,
            passes_valid_fold_rate_gate=False,
            oos_predictive_loglik_mean=1.0,
            oos_predictive_loglik_std=0.0,
            oos_predictive_loglik_worst_fold=1.0,
            oos_predictive_loglik_best_fold=1.0,
            bic_mean=1.0,
            aic_mean=1.0,
        )
    assert CANDIDATE_VALID_FOLD_RATE_GATE == 0.80


def test_candidate_grid_contract_accepts_profile_version_3() -> None:
    ids = (
        "gaussian_hmm_k2_full",
        "gaussian_hmm_k3_full",
        "gaussian_hmm_k4_full",
        "gaussian_hmm_k5_full",
        "gmm_hmm_k2_m2_full",
        "gmm_hmm_k3_m2_full",
        "gmm_hmm_k4_m2_full",
        "gmm_hmm_k5_m2_full",
        "student_t_hmm_k2_full",
        "student_t_hmm_k3_full",
        "student_t_hmm_k4_full",
        "student_t_hmm_k5_full",
    )
    base, _, _ = base_evaluation()
    evaluations = tuple(
        replace(
            base,
            candidate_id=candidate_id,
            state_count=int(candidate_id.split("_k", 1)[1].split("_", 1)[0]),
        )
        for candidate_id in ids
    )
    aggregates = tuple(aggregate_candidate(item) for item in evaluations)
    grid = CandidateGridEvaluation(
        profile_id="xetra",
        profile_config_version=3,
        source_build_id=base.source_build_id,
        feature_order=base.feature_order,
        feature_selection_definition_hash=base.feature_selection_definition_hash,
        feature_selection_execution_hash=base.feature_selection_execution_hash,
        evaluation_plan_hash=base.evaluation_plan_hash,
        evaluations=evaluations,
        aggregates=aggregates,
    )
    assert tuple(item.candidate_id for item in grid.evaluations) == ids
