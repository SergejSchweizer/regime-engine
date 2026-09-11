"""Nested prefix search against one frozen causal teacher for Xetra v4."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from typing import TYPE_CHECKING, cast

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.model_clock import (
    build_model_clock_preflight,
    require_model_clock_eligible,
)
from market_regime_engine.evaluation.selection import rank_same_feature_candidates
from market_regime_engine.evaluation.walk_forward import (
    AdapterFactory,
    WalkForwardEvaluation,
    run_walk_forward_candidate,
)
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan
from market_regime_engine.evaluations.agreement_v4 import compute_soft_regime_nmi
from market_regime_engine.evaluations.process_parallel import cpu_process_pool
from market_regime_engine.evaluations.provisional_teacher import build_inner_walk_forward_plan
from market_regime_engine.feature_discovery.contracts import (
    MAX_PREFIX_LENGTH,
    MIN_MODEL_CLOCK_VALID_FOLD_RATE,
    MIN_MODEL_TEST_OBSERVATIONS,
    MIN_MODEL_TRAIN_OBSERVATIONS,
    MIN_PREFIX_LENGTH,
    MIN_TEACHER_SHARED_SUPPORT,
    CandidateEvaluation,
    PrefixEvaluation,
    PrefixSearchResult,
    ProvisionalTeacherReference,
    content_hash,
    final_prefix_upper_bound,
)
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile
from market_regime_engine.runtime.cpu import cpu_worker_count
from market_regime_engine.training.adapter_factory import adapter_factory
from market_regime_engine.training.candidate_grid import aggregate_candidate

if TYPE_CHECKING:
    from market_regime_engine.evaluation_runs.hmm_units import HMMSeedCheckpoint

_TIMESTAMP_COLUMN = "timestamp_m1"
_GAUSSIAN_STATE_COUNTS = (2, 3, 4, 5)
_PREFIX_NMI_TIE_TOLERANCE = 1.0e-12

PrefixCandidateRunner = Callable[
    [pd.DataFrame, WalkForwardPlan, ModelProfile, ResolvedCandidateProfile, AdapterFactory],
    WalkForwardEvaluation,
]


@dataclass(frozen=True, slots=True)
class _PrefixProcessTask:
    """Pickle-safe immutable input for one prefix candidate."""

    source_rows: pd.DataFrame
    plan: WalkForwardPlan
    profile: ModelProfile
    candidate: ResolvedCandidateProfile


def _evaluate_prefix_in_process(task: _PrefixProcessTask) -> WalkForwardEvaluation:
    """Run one prefix candidate in a separate interpreter."""

    return run_walk_forward_candidate(
        task.source_rows,
        plan=task.plan,
        profile=task.profile,
        candidate=task.candidate,
        adapter_factory=cast(AdapterFactory, adapter_factory(task.profile, task.candidate)),
        max_workers=1,
    )


def _evaluate_candidates(
    source_rows: pd.DataFrame,
    plan: WalkForwardPlan,
    profile: ModelProfile,
    candidates: tuple[ResolvedCandidateProfile, ...],
    runner: PrefixCandidateRunner,
    max_workers: int | None,
    seed_checkpoint_factory: Callable[[str, str, int], HMMSeedCheckpoint] | None = None,
) -> dict[str, WalkForwardEvaluation]:
    """Evaluate one prefix's candidates concurrently with deterministic output assembly."""

    worker_limit = cpu_worker_count(max_workers, task_count=len(candidates))

    def evaluate(candidate: ResolvedCandidateProfile) -> WalkForwardEvaluation:
        if seed_checkpoint_factory is not None and runner is run_prefix_gaussian_candidate:
            return run_prefix_gaussian_candidate(
                source_rows,
                plan,
                profile,
                candidate,
                cast(AdapterFactory, adapter_factory(profile, candidate)),
                seed_checkpoint_factory=lambda fold_id: seed_checkpoint_factory(
                    candidate.candidate_id,
                    fold_id,
                    candidate.state_count,
                ),
            )
        return runner(
            source_rows,
            plan,
            profile,
            candidate,
            cast(AdapterFactory, adapter_factory(profile, candidate)),
        )

    use_processes = (
        runner is run_prefix_gaussian_candidate
        and seed_checkpoint_factory is None
        and worker_limit > 1
    )
    if use_processes:
        tasks = tuple(
            _PrefixProcessTask(source_rows, plan, profile, candidate) for candidate in candidates
        )
        with cpu_process_pool(worker_limit) as executor:
            futures = {
                task.candidate.candidate_id: executor.submit(_evaluate_prefix_in_process, task)
                for task in tasks
            }
            return {candidate_id: futures[candidate_id].result() for candidate_id in futures}
    if worker_limit == 1:
        return {candidate.candidate_id: evaluate(candidate) for candidate in candidates}
    with ThreadPoolExecutor(max_workers=worker_limit) as thread_executor:
        futures = {
            candidate.candidate_id: thread_executor.submit(evaluate, candidate)
            for candidate in candidates
        }
        return {candidate_id: future.result() for candidate_id, future in futures.items()}


def _utc(value: object, field_name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != UTC.utcoffset(value)
    ):
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    return value


def _validate_features(
    ranked_features: tuple[str, ...], original_feature_universe: tuple[str, ...]
) -> None:
    if (
        not isinstance(ranked_features, tuple)
        or len(ranked_features) < MIN_PREFIX_LENGTH
        or any(not isinstance(name, str) or not name.strip() for name in ranked_features)
        or len(set(ranked_features)) != len(ranked_features)
    ):
        raise ValueError("ranked_features must contain at least two unique feature names")
    if (
        not isinstance(original_feature_universe, tuple)
        or not original_feature_universe
        or len(set(original_feature_universe)) != len(original_feature_universe)
        or any(feature not in original_feature_universe for feature in ranked_features)
    ):
        raise ValueError("original feature universe must contain every ranked feature exactly once")


def _validate_profile(profile: ModelProfile) -> None:
    if profile.profile_id != "xetra" or profile.profile_config_version != 4:
        raise ValueError("prefix search requires the canonical Xetra v4 profile")
    if profile.gaussian_hmm.candidate_states != _GAUSSIAN_STATE_COUNTS:
        raise ValueError("prefix search requires Gaussian K2/K3/K4/K5 candidates")


def _prefix_candidates(
    feature_order: tuple[str, ...],
    *,
    original_feature_universe: tuple[str, ...],
    source_build_id: str,
    feature_selection_definition_hash: str,
    feature_selection_execution_hash: str,
) -> tuple[ResolvedCandidateProfile, ...]:
    return tuple(
        ResolvedCandidateProfile(
            candidate_id=f"gaussian_hmm_k{state_count}_full",
            state_count=state_count,
            covariance_type="full",
            feature_order=feature_order,
            feature_dimension=len(feature_order),
            source_build_id=source_build_id,
            feature_selection_definition_hash=feature_selection_definition_hash,
            feature_selection_execution_hash=feature_selection_execution_hash,
            original_feature_universe=original_feature_universe,
            feature_contract_version=4,
        )
        for state_count in _GAUSSIAN_STATE_COUNTS
    )


def run_prefix_gaussian_candidate(
    source_rows: pd.DataFrame,
    plan: WalkForwardPlan,
    profile: ModelProfile,
    candidate: ResolvedCandidateProfile,
    candidate_adapter_factory: AdapterFactory,
    *,
    seed_checkpoint_factory: Callable[[str], HMMSeedCheckpoint] | None = None,
) -> WalkForwardEvaluation:
    """Run one Gaussian prefix candidate through the shared walk-forward runner."""

    return run_walk_forward_candidate(
        source_rows,
        plan=plan,
        profile=profile,
        candidate=candidate,
        adapter_factory=candidate_adapter_factory,
        seed_checkpoint_factory=seed_checkpoint_factory,
    )


def _evaluation_support(
    evaluation: WalkForwardEvaluation,
) -> tuple[tuple[datetime, ...], tuple[tuple[float, ...], ...]]:
    timestamps: list[datetime] = []
    probabilities: list[tuple[float, ...]] = []
    for fold in evaluation.valid_folds:
        if len(fold.oos_timestamps) != len(fold.oos_filtered_probabilities):
            raise ValueError("prefix OOS timestamps and probabilities must align")
        for timestamp, row in zip(
            fold.oos_timestamps, fold.oos_filtered_probabilities, strict=True
        ):
            current = _utc(timestamp, "prefix OOS timestamp")
            if timestamps and current <= timestamps[-1]:
                raise ValueError("prefix OOS timestamps must be unique and strictly increasing")
            timestamps.append(current)
            probabilities.append(tuple(float(value) for value in row))
    if not timestamps:
        raise ValueError("prefix candidate has no valid OOS probability support")
    return tuple(timestamps), tuple(probabilities)


def _invalid_prefix(
    prefix_length: int,
    feature_order: tuple[str, ...],
    reason: str,
    *,
    candidate_evaluations: tuple[CandidateEvaluation, ...] = (),
    shared_timestamp_count: int = 0,
) -> PrefixEvaluation:
    return PrefixEvaluation(
        prefix_length=prefix_length,
        feature_order=feature_order,
        candidate_id="gaussian_hmm_k2_full",
        shared_timestamp_count=shared_timestamp_count,
        shared_teacher_coverage=0.0,
        soft_regime_nmi=0.0,
        candidate_evaluations=candidate_evaluations,
        valid=False,
        invalid_reason=reason,
    )


def _rank_prefixes(prefixes: tuple[PrefixEvaluation, ...]) -> PrefixEvaluation:
    eligible = tuple(item for item in prefixes if item.valid)
    if not eligible:
        raise ValueError("no eligible prefix: every nested prefix failed its selection gates")
    maximum_nmi = max(item.soft_regime_nmi for item in eligible)
    nmi_ties = tuple(
        item for item in eligible if item.soft_regime_nmi >= maximum_nmi - _PREFIX_NMI_TIE_TOLERANCE
    )
    maximum_shared = max(item.shared_timestamp_count for item in nmi_ties)
    shared_ties = tuple(item for item in nmi_ties if item.shared_timestamp_count == maximum_shared)
    return min(shared_ties, key=lambda item: item.prefix_length)


def search_ranked_prefixes(
    source_rows: pd.DataFrame,
    *,
    ranked_features: tuple[str, ...],
    teacher: ProvisionalTeacherReference,
    profile: ModelProfile,
    inner_plan: WalkForwardPlan | None = None,
    source_build_id: str | None = None,
    original_feature_universe: tuple[str, ...] | None = None,
    feature_selection_definition_hash: str | None = None,
    feature_selection_execution_hash: str | None = None,
    runner: PrefixCandidateRunner = run_prefix_gaussian_candidate,
    max_workers: int | None = None,
    seed_checkpoint_factory: Callable[[str, str, int], HMMSeedCheckpoint] | None = None,
) -> PrefixSearchResult:
    """Evaluate every exact ranked prefix and choose only by teacher soft NMI.

    Candidate ranking is performed independently inside each prefix through
    the shared PR-220 predictive ranking.  Likelihood and information criteria
    never cross prefix dimensions; cross-prefix selection uses only soft NMI,
    shared teacher timestamp count, and smaller prefix length.
    """

    if not isinstance(source_rows, pd.DataFrame):
        raise TypeError("prefix search requires a pandas DataFrame")
    if not isinstance(teacher, ProvisionalTeacherReference):
        raise TypeError("prefix search requires a ProvisionalTeacherReference")
    _validate_profile(profile)
    build_id = teacher.source_build_id if source_build_id is None else source_build_id
    if build_id != teacher.source_build_id:
        raise ValueError("prefix search source build differs from the frozen teacher")
    universe = ranked_features if original_feature_universe is None else original_feature_universe
    _validate_features(ranked_features, universe)
    if not isinstance(build_id, str) or not build_id or build_id.strip() != build_id:
        raise ValueError("prefix search source_build_id must be non-empty and trimmed")
    definition_hash = feature_selection_definition_hash or content_hash(
        ("xetra_global_regime_v4", "prefix_definition")
    )
    execution_hash = feature_selection_execution_hash or content_hash(
        ("xetra_global_regime_v4", "ranked_prefixes", ranked_features)
    )
    if inner_plan is None:
        if _TIMESTAMP_COLUMN not in source_rows.columns:
            raise ValueError(f"source rows must contain {_TIMESTAMP_COLUMN}")
        inner_plan = build_inner_walk_forward_plan(tuple(source_rows[_TIMESTAMP_COLUMN]))
    if not isinstance(inner_plan, WalkForwardPlan) or not inner_plan.folds:
        raise ValueError("prefix search requires a non-empty inner walk-forward plan")
    if _TIMESTAMP_COLUMN not in source_rows.columns:
        raise ValueError(f"source rows must contain {_TIMESTAMP_COLUMN}")
    source_timestamps = tuple(
        _utc(value, "source timestamp") for value in source_rows[_TIMESTAMP_COLUMN]
    )
    if any(current <= previous for previous, current in pairwise(source_timestamps)):
        raise ValueError("source timestamps must be strictly increasing and unique")
    upper_bound = min(final_prefix_upper_bound(len(ranked_features)), MAX_PREFIX_LENGTH)
    prefix_results: list[PrefixEvaluation] = []
    for prefix_length in range(MIN_PREFIX_LENGTH, upper_bound + 1):
        feature_order = ranked_features[:prefix_length]
        try:
            candidates = _prefix_candidates(
                feature_order,
                original_feature_universe=universe,
                source_build_id=build_id,
                feature_selection_definition_hash=definition_hash,
                feature_selection_execution_hash=execution_hash,
            )
            preflight = build_model_clock_preflight(
                source_rows,
                feature_order,
                inner_plan,
                minimum_model_train_observations=MIN_MODEL_TRAIN_OBSERVATIONS,
                minimum_model_test_observations=MIN_MODEL_TEST_OBSERVATIONS,
                minimum_valid_fold_rate=MIN_MODEL_CLOCK_VALID_FOLD_RATE,
            )
            require_model_clock_eligible(preflight, "prefix")
            if seed_checkpoint_factory is None:
                evaluations_by_id = _evaluate_candidates(
                    source_rows,
                    inner_plan,
                    profile,
                    candidates,
                    runner,
                    max_workers,
                )
            else:
                evaluations_by_id = _evaluate_candidates(
                    source_rows,
                    inner_plan,
                    profile,
                    candidates,
                    runner,
                    max_workers,
                    seed_checkpoint_factory,
                )
            raw_evaluations = tuple(
                evaluations_by_id[candidate.candidate_id] for candidate in candidates
            )
            aggregates = tuple(aggregate_candidate(evaluation) for evaluation in raw_evaluations)
            candidate_summaries = tuple(
                CandidateEvaluation(
                    candidate_id=aggregate.candidate_id,
                    feature_order=feature_order,
                    source_build_id=build_id,
                    plan_hash=inner_plan.plan_hash,
                    valid_fold_count=aggregate.valid_fold_count,
                    total_fold_count=aggregate.planned_fold_count,
                    oos_predictive_loglik_mean=aggregate.oos_predictive_loglik_mean,
                    oos_predictive_loglik_std=aggregate.oos_predictive_loglik_std,
                    oos_predictive_loglik_worst=aggregate.oos_predictive_loglik_worst_fold,
                    mean_bic=aggregate.bic_mean,
                    mean_aic=aggregate.aic_mean,
                    valid=aggregate.valid_fold_count > 0,
                    invalid_reason=(
                        None if aggregate.valid_fold_count > 0 else "candidate has no valid folds"
                    ),
                )
                for aggregate in aggregates
            )
            selection = rank_same_feature_candidates(raw_evaluations, aggregates)
            selected_candidate = next(
                evaluation
                for evaluation in raw_evaluations
                if evaluation.candidate_id == selection.champion_candidate_id
            )
            candidate_times, candidate_probabilities = _evaluation_support(selected_candidate)
            agreement = compute_soft_regime_nmi(
                candidate_times,
                candidate_probabilities,
                teacher.timestamps,
                teacher.filtered_probabilities,
            )
            teacher_coverage = agreement.shared_timestamp_count / len(teacher.timestamps)
            if teacher_coverage < MIN_TEACHER_SHARED_SUPPORT:
                prefix_results.append(
                    _invalid_prefix(
                        prefix_length,
                        feature_order,
                        "shared teacher support "
                        f"{teacher_coverage:.6f} below {MIN_TEACHER_SHARED_SUPPORT:.2f}",
                        candidate_evaluations=candidate_summaries,
                        shared_timestamp_count=agreement.shared_timestamp_count,
                    )
                )
                continue
            prefix_results.append(
                PrefixEvaluation(
                    prefix_length=prefix_length,
                    feature_order=feature_order,
                    candidate_id=selection.champion_candidate_id,
                    shared_timestamp_count=agreement.shared_timestamp_count,
                    shared_teacher_coverage=teacher_coverage,
                    soft_regime_nmi=agreement.soft_regime_nmi,
                    candidate_evaluations=candidate_summaries,
                )
            )
        except (ValueError, TypeError) as exc:
            prefix_results.append(_invalid_prefix(prefix_length, feature_order, str(exc)))
    prefix_evaluations = tuple(prefix_results)
    selected = _rank_prefixes(prefix_evaluations)
    return PrefixSearchResult(
        ranked_features=ranked_features,
        evaluations=prefix_evaluations,
        selected_prefix_length=selected.prefix_length,
        selected_candidate_id=selected.candidate_id,
    )


evaluate_prefix_search = search_ranked_prefixes
select_prefix = search_ranked_prefixes


__all__ = [
    "PrefixCandidateRunner",
    "evaluate_prefix_search",
    "run_prefix_gaussian_candidate",
    "search_ranked_prefixes",
    "select_prefix",
]
