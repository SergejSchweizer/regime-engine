"""Deterministic orchestration and aggregate evidence for the exact Xetra candidate grid."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from math import isfinite
from statistics import fmean, pstdev
from typing import TYPE_CHECKING, cast

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.walk_forward import (
    AdapterFactory,
    WalkForwardEvaluation,
    run_walk_forward_candidate,
)
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan
from market_regime_engine.evaluation_runs.stages import StageCheckpoint
from market_regime_engine.evaluations.process_parallel import cpu_process_pool
from market_regime_engine.evaluations.scheduling import randomized_order
from market_regime_engine.feature_discovery.contracts import content_hash
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.profiles.resolution import (
    ResolvedCandidateProfile,
    ResolvedSelectedFeatureProfile,
    expected_candidate_ids,
    validate_candidate_comparison_inputs,
)
from market_regime_engine.runtime.cpu import cpu_worker_count
from market_regime_engine.training.adapter_factory import adapter_factory

if TYPE_CHECKING:
    from market_regime_engine.evaluation_runs.hmm_units import HMMSeedCheckpoint

SUPPORTED_CANDIDATE_IDS = frozenset(expected_candidate_ids())
CANDIDATE_VALID_FOLD_RATE_GATE = 0.80
RANKING_ABS_TOLERANCE = 1e-12

AdapterFactoryBuilder = Callable[[ResolvedCandidateProfile], AdapterFactory]
CandidateRunner = Callable[
    [pd.DataFrame, WalkForwardPlan, ModelProfile, ResolvedCandidateProfile, AdapterFactory],
    WalkForwardEvaluation,
]
SeedCheckpointFactory = Callable[[str, str, int], "HMMSeedCheckpoint"]


@dataclass(frozen=True, slots=True)
class _CandidateProcessTask:
    """Pickle-safe immutable input for one CPU-bound candidate evaluation."""

    source_rows: pd.DataFrame
    plan: WalkForwardPlan
    profile: ModelProfile
    candidate: ResolvedCandidateProfile


def _evaluate_candidate_in_process(task: _CandidateProcessTask) -> WalkForwardEvaluation:
    """Evaluate one candidate outside the caller's interpreter/GIL."""

    return run_walk_forward_candidate(
        task.source_rows,
        plan=task.plan,
        profile=task.profile,
        candidate=task.candidate,
        adapter_factory=cast(AdapterFactory, adapter_factory(task.profile, task.candidate)),
        max_workers=1,
    )


@dataclass(frozen=True, slots=True)
class CandidateAggregate:
    candidate_id: str
    state_count: int
    planned_fold_count: int
    valid_fold_count: int
    invalid_fold_count: int
    valid_fold_rate: float
    passes_valid_fold_rate_gate: bool
    oos_predictive_loglik_mean: float | None
    oos_predictive_loglik_std: float | None
    oos_predictive_loglik_worst_fold: float | None
    oos_predictive_loglik_best_fold: float | None
    bic_mean: float | None
    aic_mean: float | None

    def __post_init__(self) -> None:
        if self.state_count not in (2, 3, 4, 5):
            raise ValueError("candidate aggregate supports exactly K2/K3/K4/K5")
        if self.candidate_id not in SUPPORTED_CANDIDATE_IDS:
            raise ValueError("aggregate candidate identity is unsupported")
        if self.planned_fold_count < 1:
            raise ValueError("candidate aggregate requires at least one planned fold")
        if self.valid_fold_count < 0 or self.invalid_fold_count < 0:
            raise ValueError("candidate fold counts cannot be negative")
        if self.valid_fold_count + self.invalid_fold_count != self.planned_fold_count:
            raise ValueError("candidate valid/invalid counts must reconcile to planned folds")
        expected_rate = self.valid_fold_count / self.planned_fold_count
        if abs(self.valid_fold_rate - expected_rate) > RANKING_ABS_TOLERANCE:
            raise ValueError("candidate valid-fold rate does not reconcile to counts")
        if self.passes_valid_fold_rate_gate != (
            self.valid_fold_rate >= CANDIDATE_VALID_FOLD_RATE_GATE
        ):
            raise ValueError("candidate valid-fold gate result is inconsistent")
        score_values = (
            self.oos_predictive_loglik_mean,
            self.oos_predictive_loglik_std,
            self.oos_predictive_loglik_worst_fold,
            self.oos_predictive_loglik_best_fold,
            self.bic_mean,
            self.aic_mean,
        )
        if self.valid_fold_count == 0:
            if any(value is not None for value in score_values):
                raise ValueError("zero-valid-fold candidate cannot expose aggregate score values")
        elif any(value is None or not isfinite(value) for value in score_values):
            raise ValueError("candidate with valid folds requires finite aggregate score values")
        if self.oos_predictive_loglik_std is not None and self.oos_predictive_loglik_std < 0.0:
            raise ValueError("candidate OOS standard deviation cannot be negative")


@dataclass(frozen=True, slots=True)
class CandidateGridEvaluation:
    profile_id: str
    profile_config_version: int
    source_build_id: str
    feature_order: tuple[str, ...]
    feature_selection_definition_hash: str
    feature_selection_execution_hash: str
    evaluation_plan_hash: str
    evaluations: tuple[WalkForwardEvaluation, ...]
    aggregates: tuple[CandidateAggregate, ...]

    def __post_init__(self) -> None:
        if self.profile_id != "xetra" or self.profile_config_version != 4:
            raise ValueError("candidate grid requires the Xetra v4 profile")
        expected_ids = expected_candidate_ids()
        if tuple(item.candidate_id for item in self.evaluations) != expected_ids:
            raise ValueError(
                "candidate grid evaluations must be ordered by configured candidate ID"
            )
        if tuple(item.candidate_id for item in self.aggregates) != expected_ids:
            raise ValueError("candidate grid aggregates must be ordered by configured candidate ID")
        for evaluation, aggregate in zip(self.evaluations, self.aggregates, strict=True):
            if evaluation.candidate_id != aggregate.candidate_id:
                raise ValueError("candidate evaluation and aggregate identities differ")
            if evaluation.state_count != aggregate.state_count:
                raise ValueError("candidate evaluation and aggregate state counts differ")
            if evaluation.source_build_id != self.source_build_id:
                raise ValueError("candidate source build differs inside candidate grid")
            if evaluation.feature_order != self.feature_order:
                raise ValueError("candidate feature order differs inside candidate grid")
            if (
                evaluation.feature_selection_definition_hash
                != self.feature_selection_definition_hash
            ):
                raise ValueError("candidate definition hash differs inside candidate grid")
            if evaluation.feature_selection_execution_hash != self.feature_selection_execution_hash:
                raise ValueError("candidate execution hash differs inside candidate grid")
            if evaluation.evaluation_plan_hash != self.evaluation_plan_hash:
                raise ValueError("candidate evaluation-plan hash differs inside candidate grid")


def aggregate_candidate(evaluation: WalkForwardEvaluation) -> CandidateAggregate:
    """Aggregate only valid-fold evidence using the exact EVALUATION definitions."""

    valid = evaluation.valid_folds
    valid_count = len(valid)
    planned_count = len(evaluation.folds)
    invalid_count = planned_count - valid_count
    rate = valid_count / planned_count
    if not valid:
        return CandidateAggregate(
            candidate_id=evaluation.candidate_id,
            state_count=evaluation.state_count,
            planned_fold_count=planned_count,
            valid_fold_count=0,
            invalid_fold_count=invalid_count,
            valid_fold_rate=rate,
            passes_valid_fold_rate_gate=rate >= CANDIDATE_VALID_FOLD_RATE_GATE,
            oos_predictive_loglik_mean=None,
            oos_predictive_loglik_std=None,
            oos_predictive_loglik_worst_fold=None,
            oos_predictive_loglik_best_fold=None,
            bic_mean=None,
            aic_mean=None,
        )

    oos_values = tuple(
        fold.oos_predictive_log_likelihood_per_observation
        for fold in valid
        if fold.oos_predictive_log_likelihood_per_observation is not None
    )
    bic_values = tuple(fold.bic for fold in valid if fold.bic is not None)
    aic_values = tuple(fold.aic for fold in valid if fold.aic is not None)
    missing_required_metric = (
        len(oos_values) != valid_count
        or len(bic_values) != valid_count
        or len(aic_values) != valid_count
    )
    if missing_required_metric:
        raise ValueError("valid fold is missing required candidate aggregate metric")
    if any(not isfinite(value) for value in (*oos_values, *bic_values, *aic_values)):
        raise ValueError("candidate aggregate inputs must be finite")

    return CandidateAggregate(
        candidate_id=evaluation.candidate_id,
        state_count=evaluation.state_count,
        planned_fold_count=planned_count,
        valid_fold_count=valid_count,
        invalid_fold_count=invalid_count,
        valid_fold_rate=rate,
        passes_valid_fold_rate_gate=rate >= CANDIDATE_VALID_FOLD_RATE_GATE,
        oos_predictive_loglik_mean=fmean(oos_values),
        oos_predictive_loglik_std=pstdev(oos_values),
        oos_predictive_loglik_worst_fold=min(oos_values),
        oos_predictive_loglik_best_fold=max(oos_values),
        bic_mean=fmean(bic_values),
        aic_mean=fmean(aic_values),
    )


def _default_adapter_builder(
    profile: ModelProfile, candidate: ResolvedCandidateProfile
) -> AdapterFactory:
    return adapter_factory(profile, candidate)  # type: ignore[return-value]


def _default_runner(
    source_rows: pd.DataFrame,
    plan: WalkForwardPlan,
    profile: ModelProfile,
    candidate: ResolvedCandidateProfile,
    adapter_factory: AdapterFactory,
    max_workers: int | None = None,
    seed_checkpoint_factory: Callable[[str], HMMSeedCheckpoint] | None = None,
) -> WalkForwardEvaluation:
    return run_walk_forward_candidate(
        source_rows,
        plan=plan,
        profile=profile,
        candidate=candidate,
        adapter_factory=adapter_factory,
        max_workers=max_workers,
        seed_checkpoint_factory=seed_checkpoint_factory,
    )


def evaluate_candidate_grid(
    source_rows: pd.DataFrame,
    *,
    plan: WalkForwardPlan,
    profile: ModelProfile,
    resolved_profile: ResolvedSelectedFeatureProfile,
    adapter_factory_builder: AdapterFactoryBuilder | None = None,
    runner: CandidateRunner = _default_runner,
    max_workers: int | None = None,
    seed_checkpoint_factory: SeedCheckpointFactory | None = None,
) -> CandidateGridEvaluation:
    """Evaluate K2/K3/K4/K5 concurrently against one frozen source/fold/feature contract."""

    if profile.profile_id != resolved_profile.profile_id:
        raise ValueError("model and resolved profile IDs differ")
    if profile.profile_config_version != resolved_profile.profile_config_version:
        raise ValueError("model and resolved profile config versions differ")
    validate_candidate_comparison_inputs(
        resolved_profile.candidates,
        profile_config_version=profile.profile_config_version,
    )
    if plan.plan_hash == "" or not plan.folds:
        raise ValueError("candidate grid requires a non-empty complete walk-forward plan")

    worker_limit = cpu_worker_count(max_workers, task_count=len(resolved_profile.candidates))
    scheduled_candidates = tuple(
        next(
            candidate
            for candidate in resolved_profile.candidates
            if candidate.candidate_id == candidate_id
        )
        for candidate_id in randomized_order(
            tuple(candidate.candidate_id for candidate in resolved_profile.candidates),
            scope="candidate-grid:" + ",".join(resolved_profile.final_features),
        )
    )

    def evaluate(candidate: ResolvedCandidateProfile) -> WalkForwardEvaluation:
        candidate_adapter = (
            _default_adapter_builder(profile, candidate)
            if adapter_factory_builder is None
            else adapter_factory_builder(candidate)
        )
        if seed_checkpoint_factory is not None and runner is _default_runner:
            seed_checkpoint = seed_checkpoint_factory(
                candidate.candidate_id,
                plan.folds[0].fold_id,
                candidate.state_count,
            )
            stage_checkpoint = StageCheckpoint(
                seed_checkpoint.run_identity,
                seed_checkpoint.store,
                f"{seed_checkpoint.scope}:final_grid_candidate:{candidate.candidate_id}",
            )

            def scoped_seed_checkpoint(fold_id: str) -> HMMSeedCheckpoint:
                return replace(
                    seed_checkpoint_factory(
                        candidate.candidate_id,
                        fold_id,
                        candidate.state_count,
                    ),
                    scope=stage_checkpoint.scope,
                )

            def compute() -> WalkForwardEvaluation:
                return _default_runner(
                    source_rows,
                    plan,
                    profile,
                    candidate,
                    candidate_adapter,
                    max_workers=max_workers,
                    seed_checkpoint_factory=scoped_seed_checkpoint,
                )

            return stage_checkpoint.run(
                "candidate_evaluation",
                compute,
                parameters=(
                    ("candidate_hash", content_hash(candidate)),
                    ("candidate_id", candidate.candidate_id),
                    ("plan_hash", plan.plan_hash),
                ),
            )
        return runner(source_rows, plan, profile, candidate, candidate_adapter)

    use_processes = (
        runner is _default_runner
        and adapter_factory_builder is None
        and seed_checkpoint_factory is None
        and worker_limit > 1
    )
    if use_processes:
        tasks = tuple(
            _CandidateProcessTask(source_rows, plan, profile, candidate)
            for candidate in scheduled_candidates
        )
        with cpu_process_pool(worker_limit) as executor:
            process_futures = {
                task.candidate.candidate_id: executor.submit(_evaluate_candidate_in_process, task)
                for task in tasks
            }
            evaluations_by_id = {
                candidate_id: process_futures[candidate_id].result()
                for candidate_id in process_futures
            }
        evaluations = tuple(
            evaluations_by_id[candidate.candidate_id] for candidate in scheduled_candidates
        )
    elif worker_limit == 1:
        evaluations = tuple(evaluate(candidate) for candidate in scheduled_candidates)
    else:
        with ThreadPoolExecutor(max_workers=worker_limit) as thread_executor:
            futures = [
                thread_executor.submit(evaluate, candidate) for candidate in scheduled_candidates
            ]
            evaluations = tuple(future.result() for future in futures)
    by_candidate_id = {item.candidate_id: item for item in evaluations}
    expected_ids = expected_candidate_ids()
    if set(by_candidate_id) != set(expected_ids) or len(by_candidate_id) != len(evaluations):
        raise ValueError("candidate runner returned unexpected candidate identities/order")
    evaluations = tuple(
        by_candidate_id[candidate.candidate_id] for candidate in resolved_profile.candidates
    )
    if tuple(item.candidate_id for item in evaluations) != expected_ids:
        raise ValueError("candidate runner returned unexpected candidate identities/order")
    expected_fold_ids = tuple(fold.fold_id for fold in plan.folds)
    for evaluation in evaluations:
        if tuple(fold.fold_id for fold in evaluation.folds) != expected_fold_ids:
            raise ValueError("candidates must preserve identical complete planned fold identities")
        if evaluation.evaluation_plan_hash != plan.plan_hash:
            raise ValueError("candidate evaluation plan hash differs from shared plan")
        if evaluation.source_build_id != resolved_profile.source_build_id:
            raise ValueError("candidate source build differs from resolved profile")
        if evaluation.feature_order != resolved_profile.final_features:
            raise ValueError("candidate feature order differs from frozen resolved profile")
        if (
            evaluation.feature_selection_definition_hash
            != resolved_profile.feature_selection_definition_hash
            or evaluation.feature_selection_execution_hash
            != resolved_profile.feature_selection_execution_hash
        ):
            raise ValueError("candidate selection hashes differ from frozen resolved profile")

    aggregates = tuple(aggregate_candidate(evaluation) for evaluation in evaluations)
    return CandidateGridEvaluation(
        profile_id=resolved_profile.profile_id,
        profile_config_version=resolved_profile.profile_config_version,
        source_build_id=resolved_profile.source_build_id,
        feature_order=resolved_profile.final_features,
        feature_selection_definition_hash=resolved_profile.feature_selection_definition_hash,
        feature_selection_execution_hash=resolved_profile.feature_selection_execution_hash,
        evaluation_plan_hash=plan.plan_hash,
        evaluations=evaluations,
        aggregates=aggregates,
    )
