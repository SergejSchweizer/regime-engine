"""Fixed-K Gaussian/GMM/Student-t family evaluation.

The public v4 grid compares twelve candidates on one feature vector.  The
K-slot policy deliberately uses a smaller, independent grid: exactly three
families are evaluated for one requested K and one already-selected feature
tuple.  This module reuses the production walk-forward runner and ranking
kernel; it does not implement a second HMM.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.selection import (
    StatisticalChampionSelection,
    rank_same_feature_candidates,
)
from market_regime_engine.evaluation.walk_forward import (
    AdapterFactory,
    WalkForwardEvaluation,
    run_walk_forward_candidate,
    validate_mandatory_pca_feature_universe,
)
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan
from market_regime_engine.evaluations.process_parallel import cpu_process_pool
from market_regime_engine.feature_discovery.contracts import content_hash
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile
from market_regime_engine.runtime.cpu import cpu_worker_count, nested_worker_limits
from market_regime_engine.training.adapter_factory import adapter_factory
from market_regime_engine.training.candidate_grid import CandidateAggregate, aggregate_candidate

K_FAMILY_GRID_VERSION = "k_family_grid.v1"
K_FAMILY_ORDER = ("gaussian_hmm", "gmm_hmm", "student_t_hmm")
K_FAMILY_IDS = {
    family: lambda state_count, family=family: (
        f"gmm_hmm_k{state_count}_m2_full"
        if family == "gmm_hmm"
        else f"{family}_k{state_count}_full"
    )
    for family in K_FAMILY_ORDER
}

CandidateRunner = Callable[
    [pd.DataFrame, WalkForwardPlan, ModelProfile, ResolvedCandidateProfile, AdapterFactory],
    WalkForwardEvaluation,
]


def _default_runner(
    source_rows: pd.DataFrame,
    plan: WalkForwardPlan,
    profile: ModelProfile,
    candidate: ResolvedCandidateProfile,
    candidate_adapter: AdapterFactory,
    *,
    max_workers: int | None = None,
    pca_raw_feature_order: tuple[str, ...] | None = None,
    pca_variance_threshold: float = 0.90,
) -> WalkForwardEvaluation:
    return run_walk_forward_candidate(
        source_rows,
        plan=plan,
        profile=profile,
        candidate=candidate,
        adapter_factory=candidate_adapter,
        max_workers=max_workers,
        pca_raw_feature_order=pca_raw_feature_order,
        pca_variance_threshold=pca_variance_threshold,
    )


@dataclass(frozen=True, slots=True)
class _FamilyProcessTask:
    source_rows: pd.DataFrame
    plan: WalkForwardPlan
    profile: ModelProfile
    candidate: ResolvedCandidateProfile
    nested_workers: int
    pca_raw_feature_order: tuple[str, ...]
    pca_variance_threshold: float


def _evaluate_family_process(task: _FamilyProcessTask) -> WalkForwardEvaluation:
    candidate_adapter = cast(AdapterFactory, adapter_factory(task.profile, task.candidate))
    return _default_runner(
        task.source_rows,
        task.plan,
        task.profile,
        task.candidate,
        candidate_adapter,
        max_workers=task.nested_workers,
        pca_raw_feature_order=task.pca_raw_feature_order,
        pca_variance_threshold=task.pca_variance_threshold,
    )


@dataclass(frozen=True, slots=True)
class KFamilyGridEvaluation:
    """Three-family evidence and one within-K statistical winner."""

    grid_version: str
    state_count: int
    source_build_id: str
    feature_order: tuple[str, ...]
    feature_order_hash: str
    feature_selection_definition_hash: str
    feature_selection_execution_hash: str
    evaluation_plan_hash: str
    evaluations: tuple[WalkForwardEvaluation, ...]
    aggregates: tuple[CandidateAggregate, ...]
    selection: StatisticalChampionSelection | None
    no_selection_reason: str | None

    def __post_init__(self) -> None:
        if self.grid_version != K_FAMILY_GRID_VERSION:
            raise ValueError("unsupported K-family grid version")
        if self.state_count not in (2, 3, 4, 5):
            raise ValueError("K-family grid state count must be 2, 3, 4 or 5")
        if not self.feature_order or len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("K-family feature order must be non-empty and duplicate-free")
        if self.feature_order_hash != content_hash(self.feature_order):
            raise ValueError("K-family feature-order hash does not reconcile")
        expected_ids = tuple(K_FAMILY_IDS[family](self.state_count) for family in K_FAMILY_ORDER)
        if tuple(item.candidate_id for item in self.evaluations) != expected_ids:
            raise ValueError("K-family evaluations must contain exactly Gaussian/GMM/Student-t")
        if tuple(item.candidate_id for item in self.aggregates) != expected_ids:
            raise ValueError("K-family aggregates must contain exactly Gaussian/GMM/Student-t")
        for evaluation, aggregate in zip(self.evaluations, self.aggregates, strict=True):
            if evaluation.state_count != self.state_count:
                raise ValueError("K-family evaluation has a mismatched state count")
            if evaluation.feature_order != self.feature_order:
                raise ValueError("K-family evaluations must share one feature tuple")
            if evaluation.source_build_id != self.source_build_id:
                raise ValueError("K-family evaluations must share source identity")
            if evaluation.evaluation_plan_hash != self.evaluation_plan_hash:
                raise ValueError("K-family evaluations must share one evaluation plan")
            if aggregate.candidate_id != evaluation.candidate_id:
                raise ValueError("K-family aggregate identity differs from evaluation")
        if (self.selection is None) == (self.no_selection_reason is None):
            raise ValueError("K-family grid requires one selection or an explicit failure")
        if self.selection is not None and self.selection.champion_state_count != self.state_count:
            raise ValueError("K-family selection has a mismatched state count")

    @property
    def canonical_json(self) -> str:
        import json
        from dataclasses import asdict

        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)

    @property
    def evidence_hash(self) -> str:
        return content_hash(self.canonical_json)


def _candidates(
    state_count: int,
    *,
    feature_order: tuple[str, ...],
    original_feature_universe: tuple[str, ...],
    source_build_id: str,
    definition_hash: str,
    execution_hash: str,
) -> tuple[ResolvedCandidateProfile, ...]:
    return tuple(
        ResolvedCandidateProfile(
            candidate_id=K_FAMILY_IDS[family](state_count),
            state_count=state_count,
            covariance_type="full",
            feature_order=feature_order,
            feature_dimension=len(feature_order),
            source_build_id=source_build_id,
            feature_selection_definition_hash=definition_hash,
            feature_selection_execution_hash=execution_hash,
            original_feature_universe=original_feature_universe,
            model_family=family,
            mixture_count=2 if family == "gmm_hmm" else 1,
            feature_contract_version=4,
        )
        for family in K_FAMILY_ORDER
    )


def evaluate_k_family_grid(
    source_rows: pd.DataFrame,
    *,
    state_count: int,
    feature_order: tuple[str, ...],
    original_feature_universe: tuple[str, ...],
    profile: ModelProfile,
    plan: WalkForwardPlan,
    source_build_id: str,
    feature_selection_definition_hash: str,
    feature_selection_execution_hash: str,
    runner: CandidateRunner | None = None,
    max_workers: int | None = None,
    pca_raw_feature_order: tuple[str, ...] | None = None,
    pca_variance_threshold: float = 0.90,
) -> KFamilyGridEvaluation:
    """Evaluate exactly three families for one fixed K in bounded processes."""

    if state_count not in (2, 3, 4, 5):
        raise ValueError("K-family grid accepts only K=2,3,4,5")
    if runner is not None and runner is not _default_runner and not callable(runner):
        raise TypeError("K-family runner must be callable")
    pca_order = validate_mandatory_pca_feature_universe(
        pca_raw_feature_order,
        original_feature_universe,
    )
    candidates = _candidates(
        state_count,
        feature_order=feature_order,
        original_feature_universe=original_feature_universe,
        source_build_id=source_build_id,
        definition_hash=feature_selection_definition_hash,
        execution_hash=feature_selection_execution_hash,
    )
    active_runner = _default_runner if runner is None else runner
    worker_limit = cpu_worker_count(max_workers, task_count=len(candidates))
    total_budget = cpu_worker_count(max_workers)
    if active_runner is _default_runner and worker_limit > 1:
        nested = nested_worker_limits(total_budget, worker_limit)
        tasks = tuple(
            _FamilyProcessTask(
                source_rows,
                plan,
                profile,
                candidate,
                nested[index],
                pca_order,
                pca_variance_threshold,
            )
            for index, candidate in enumerate(candidates)
        )
        with cpu_process_pool(worker_limit) as executor:
            by_id = {
                candidate_id: future.result()
                for candidate_id, future in (
                    (task.candidate.candidate_id, executor.submit(_evaluate_family_process, task))
                    for task in tasks
                )
            }
        evaluations = tuple(by_id[candidate.candidate_id] for candidate in candidates)
    elif worker_limit == 1:
        if active_runner is _default_runner:
            evaluations = tuple(
                _default_runner(
                    source_rows,
                    plan,
                    profile,
                    candidate,
                    cast(AdapterFactory, adapter_factory(profile, candidate)),
                    pca_raw_feature_order=pca_order,
                    pca_variance_threshold=pca_variance_threshold,
                )
                for candidate in candidates
            )
        else:
            evaluations = tuple(
                active_runner(
                    source_rows,
                    plan,
                    profile,
                    candidate,
                    cast(AdapterFactory, adapter_factory(profile, candidate)),
                )
                for candidate in candidates
            )
    else:
        raise RuntimeError("parallel K-family evaluation requires the process-safe default runner")
    aggregates = tuple(aggregate_candidate(evaluation) for evaluation in evaluations)
    reason: str | None
    try:
        selection = rank_same_feature_candidates(evaluations, aggregates)
    except ValueError as exc:
        selection = None
        reason = str(exc)
    else:
        reason = None
    return KFamilyGridEvaluation(
        grid_version=K_FAMILY_GRID_VERSION,
        state_count=state_count,
        source_build_id=source_build_id,
        feature_order=feature_order,
        feature_order_hash=content_hash(feature_order),
        feature_selection_definition_hash=feature_selection_definition_hash,
        feature_selection_execution_hash=feature_selection_execution_hash,
        evaluation_plan_hash=plan.plan_hash,
        evaluations=evaluations,
        aggregates=aggregates,
        selection=selection,
        no_selection_reason=reason,
    )


__all__ = [
    "K_FAMILY_GRID_VERSION",
    "K_FAMILY_ORDER",
    "KFamilyGridEvaluation",
    "evaluate_k_family_grid",
]
