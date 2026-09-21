"""Nested prefix search against one frozen causal teacher for Xetra v4."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from itertools import pairwise
from typing import TYPE_CHECKING, Any, cast

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.calendar_clock import (
    MIN_CALENDAR_MODEL_TEST_OBSERVATIONS,
    plan_calendar_month,
)
from market_regime_engine.evaluation.model_clock import (
    build_calendar_model_clock_preflight,
    build_model_clock_preflight,
    require_model_clock_eligible,
)
from market_regime_engine.evaluation.selection import rank_same_feature_candidates
from market_regime_engine.evaluation.walk_forward import (
    AdapterFactory,
    WalkForwardEvaluation,
    run_walk_forward_candidate,
    validate_mandatory_pca_feature_universe,
)
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan
from market_regime_engine.evaluation_runs.stages import StageCheckpoint
from market_regime_engine.evaluations.agreement_v4 import compute_soft_regime_nmi
from market_regime_engine.evaluations.process_parallel import cpu_process_pool, is_pickleable
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
from market_regime_engine.runtime.cpu import cpu_worker_count, nested_worker_limits
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
PrefixEvaluationSink = Callable[[int, str, WalkForwardEvaluation], None]


@dataclass(frozen=True, slots=True)
class _PrefixProcessTask:
    """Pickle-safe immutable input for one prefix candidate."""

    source_rows: pd.DataFrame
    plan: WalkForwardPlan
    profile: ModelProfile
    candidate: ResolvedCandidateProfile
    max_workers: int
    pca_raw_feature_order: tuple[str, ...]
    pca_variance_threshold: float
    runner: PrefixCandidateRunner


@dataclass(frozen=True, slots=True)
class _PrefixSearchTask:
    """Immutable input for evaluating one ranked prefix."""

    prefix_length: int
    source_rows: pd.DataFrame
    plan: WalkForwardPlan
    profile: ModelProfile
    teacher: ProvisionalTeacherReference
    feature_order: tuple[str, ...]
    state_counts: tuple[int, ...]
    original_feature_universe: tuple[str, ...]
    source_build_id: str
    feature_selection_definition_hash: str
    feature_selection_execution_hash: str
    runner: PrefixCandidateRunner
    max_workers: int
    seed_checkpoint_factory: Callable[[str, str, int], HMMSeedCheckpoint] | None
    pca_raw_feature_order: tuple[str, ...] | None
    pca_variance_threshold: float


@dataclass(frozen=True, slots=True)
class _EvaluatedPrefix:
    """Prefix result plus the raw selected candidate needed by the sink."""

    evaluation: PrefixEvaluation
    selected_candidate: WalkForwardEvaluation | None


def _evaluate_prefix_in_process(task: _PrefixProcessTask) -> WalkForwardEvaluation:
    """Run one prefix candidate in a separate interpreter."""

    candidate_adapter = cast(AdapterFactory, adapter_factory(task.profile, task.candidate))
    if task.runner is run_prefix_gaussian_candidate:
        return run_walk_forward_candidate(
            task.source_rows,
            plan=task.plan,
            profile=task.profile,
            candidate=task.candidate,
            adapter_factory=candidate_adapter,
            max_workers=task.max_workers,
            pca_raw_feature_order=task.pca_raw_feature_order,
            pca_variance_threshold=task.pca_variance_threshold,
        )
    return task.runner(
        task.source_rows,
        task.plan,
        task.profile,
        task.candidate,
        candidate_adapter,
    )


def _threaded_child_worker_limits(
    total_worker_budget: int,
    task_count: int,
) -> tuple[int, ...]:
    """Partition child-process lanes for checkpoint orchestration."""

    if total_worker_budget < 1 or task_count < 1:
        raise ValueError("worker budget and task count must be positive")
    concurrent_tasks = min(task_count, max(1, total_worker_budget // 2))
    baseline, remainder = divmod(total_worker_budget, concurrent_tasks)
    return tuple(baseline + int(index < remainder) for index in range(concurrent_tasks))


def _evaluate_candidates(
    source_rows: pd.DataFrame,
    plan: WalkForwardPlan,
    profile: ModelProfile,
    candidates: tuple[ResolvedCandidateProfile, ...],
    runner: PrefixCandidateRunner,
    max_workers: int | None,
    seed_checkpoint_factory: Callable[[str, str, int], HMMSeedCheckpoint] | None = None,
    pca_raw_feature_order: tuple[str, ...] | None = None,
    pca_variance_threshold: float = 0.90,
) -> dict[str, WalkForwardEvaluation]:
    """Evaluate one prefix's candidates concurrently with deterministic output assembly."""

    if pca_raw_feature_order is None:
        raise ValueError("PCA prefix evaluation requires a raw feature order")
    worker_limit = cpu_worker_count(max_workers, task_count=len(candidates))
    total_worker_budget = cpu_worker_count(max_workers)

    def evaluate(
        candidate: ResolvedCandidateProfile,
        candidate_max_workers: int,
    ) -> WalkForwardEvaluation:
        if seed_checkpoint_factory is not None and runner is run_prefix_gaussian_candidate:
            seed_checkpoint = seed_checkpoint_factory(
                candidate.candidate_id,
                plan.folds[0].fold_id,
                candidate.state_count,
            )
            stage_checkpoint = StageCheckpoint(
                seed_checkpoint.run_identity,
                seed_checkpoint.store,
                (
                    f"{seed_checkpoint.scope}:prefix_candidate:"
                    f"{len(candidate.feature_order)}:{candidate.candidate_id}"
                ),
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
                kwargs: dict[str, Any] = {
                    "max_workers": candidate_max_workers,
                    "seed_checkpoint_factory": scoped_seed_checkpoint,
                }
                kwargs.update(
                    {
                        "pca_raw_feature_order": pca_raw_feature_order,
                        "pca_variance_threshold": pca_variance_threshold,
                    }
                )
                return run_prefix_gaussian_candidate(
                    source_rows,
                    plan,
                    profile,
                    candidate,
                    cast(AdapterFactory, adapter_factory(profile, candidate)),
                    **kwargs,
                )

            return stage_checkpoint.run(
                "candidate_evaluation",
                compute,
                parameters=(
                    ("candidate_hash", content_hash(candidate)),
                    ("candidate_id", candidate.candidate_id),
                    ("feature_count", str(len(candidate.feature_order))),
                    ("plan_hash", plan.plan_hash),
                ),
            )
        if runner is not run_prefix_gaussian_candidate:
            return runner(
                source_rows,
                plan,
                profile,
                candidate,
                cast(AdapterFactory, adapter_factory(profile, candidate)),
            )
        return run_prefix_gaussian_candidate(
            source_rows,
            plan,
            profile,
            candidate,
            cast(AdapterFactory, adapter_factory(profile, candidate)),
            max_workers=candidate_max_workers,
            pca_raw_feature_order=pca_raw_feature_order,
            pca_variance_threshold=pca_variance_threshold,
        )

    use_processes = seed_checkpoint_factory is None and worker_limit > 1 and is_pickleable(runner)
    if use_processes:
        nested_limits = nested_worker_limits(total_worker_budget, worker_limit)
        tasks = tuple(
            _PrefixProcessTask(
                source_rows,
                plan,
                profile,
                candidate,
                nested_limits[index % worker_limit],
                pca_raw_feature_order,
                pca_variance_threshold,
                runner,
            )
            for index, candidate in enumerate(candidates)
        )
        with cpu_process_pool(worker_limit) as executor:
            futures = {
                task.candidate.candidate_id: executor.submit(_evaluate_prefix_in_process, task)
                for task in tasks
            }
            return {candidate_id: futures[candidate_id].result() for candidate_id in futures}
    if worker_limit == 1:
        return {
            candidate.candidate_id: evaluate(candidate, total_worker_budget)
            for candidate in candidates
        }
    if seed_checkpoint_factory is None or runner is not run_prefix_gaussian_candidate:
        raise RuntimeError(
            "parallel prefix-candidate evaluation requires a pickleable CPU runner; "
            "set max_workers=1 for an explicitly serial custom runner"
        )
    child_limits = _threaded_child_worker_limits(total_worker_budget, len(candidates))
    evaluated: list[WalkForwardEvaluation] = []
    with ThreadPoolExecutor(max_workers=len(child_limits)) as thread_executor:
        for offset in range(0, len(candidates), len(child_limits)):
            batch = candidates[offset : offset + len(child_limits)]
            batch_futures = [
                thread_executor.submit(evaluate, candidate, child_limits[index])
                for index, candidate in enumerate(batch)
            ]
            evaluated.extend(future.result() for future in batch_futures)
    return {evaluation.candidate_id: evaluation for evaluation in evaluated}


def _evaluate_prefix(task: _PrefixSearchTask) -> _EvaluatedPrefix:
    """Evaluate one prefix without invoking parent-owned sink side effects."""

    try:
        candidates = _prefix_candidates(
            task.feature_order,
            state_counts=task.state_counts,
            original_feature_universe=task.original_feature_universe,
            source_build_id=task.source_build_id,
            feature_selection_definition_hash=task.feature_selection_definition_hash,
            feature_selection_execution_hash=task.feature_selection_execution_hash,
        )
        if task.plan.folds and hasattr(task.plan.folds[0], "test_calendar_month"):
            calendar_plan = plan_calendar_month(
                tuple(task.source_rows[_TIMESTAMP_COLUMN]),
                minimum_train_source_observations=task.plan.folds[0].train_source_observations,
            )
            preflight = build_calendar_model_clock_preflight(
                task.source_rows,
                task.feature_order,
                calendar_plan,
                minimum_model_train_observations=MIN_MODEL_TRAIN_OBSERVATIONS,
                minimum_model_test_observations=MIN_CALENDAR_MODEL_TEST_OBSERVATIONS,
                minimum_valid_fold_rate=MIN_MODEL_CLOCK_VALID_FOLD_RATE,
            )
        else:
            preflight = build_model_clock_preflight(
                task.source_rows,
                task.feature_order,
                task.plan,
                minimum_model_train_observations=MIN_MODEL_TRAIN_OBSERVATIONS,
                minimum_model_test_observations=MIN_MODEL_TEST_OBSERVATIONS,
                minimum_valid_fold_rate=MIN_MODEL_CLOCK_VALID_FOLD_RATE,
            )
        require_model_clock_eligible(preflight, "prefix")
        evaluations_by_id = _evaluate_candidates(
            task.source_rows,
            task.plan,
            task.profile,
            candidates,
            task.runner,
            task.max_workers,
            seed_checkpoint_factory=task.seed_checkpoint_factory,
            pca_raw_feature_order=task.pca_raw_feature_order,
            pca_variance_threshold=task.pca_variance_threshold,
        )
        raw_evaluations = tuple(
            evaluations_by_id[candidate.candidate_id] for candidate in candidates
        )
        aggregates = tuple(aggregate_candidate(evaluation) for evaluation in raw_evaluations)
        candidate_summaries = tuple(
            CandidateEvaluation(
                candidate_id=aggregate.candidate_id,
                feature_order=task.feature_order,
                source_build_id=task.source_build_id,
                plan_hash=task.plan.plan_hash,
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
            task.teacher.timestamps,
            task.teacher.filtered_probabilities,
        )
        teacher_coverage = agreement.shared_timestamp_count / len(task.teacher.timestamps)
        if teacher_coverage < MIN_TEACHER_SHARED_SUPPORT:
            return _EvaluatedPrefix(
                _invalid_prefix(
                    task.prefix_length,
                    task.feature_order,
                    "shared teacher support "
                    f"{teacher_coverage:.6f} below {MIN_TEACHER_SHARED_SUPPORT:.2f}",
                    candidate_id=raw_evaluations[0].candidate_id,
                    candidate_evaluations=candidate_summaries,
                    shared_timestamp_count=agreement.shared_timestamp_count,
                ),
                None,
            )
        return _EvaluatedPrefix(
            PrefixEvaluation(
                prefix_length=task.prefix_length,
                feature_order=task.feature_order,
                candidate_id=selection.champion_candidate_id,
                shared_timestamp_count=agreement.shared_timestamp_count,
                shared_teacher_coverage=teacher_coverage,
                soft_regime_nmi=agreement.soft_regime_nmi,
                candidate_evaluations=candidate_summaries,
            ),
            selected_candidate,
        )
    except (ValueError, TypeError) as exc:
        return _EvaluatedPrefix(
            _invalid_prefix(
                task.prefix_length,
                task.feature_order,
                str(exc),
                candidate_id=f"gaussian_hmm_k{task.state_counts[0]}_full",
            ),
            None,
        )


def _evaluate_prefix_in_search_process(task: _PrefixSearchTask) -> _EvaluatedPrefix:
    """Evaluate one complete prefix in a separate interpreter."""

    return _evaluate_prefix(task)


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
    state_counts: tuple[int, ...] = _GAUSSIAN_STATE_COUNTS,
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
        for state_count in state_counts
    )


def run_prefix_gaussian_candidate(
    source_rows: pd.DataFrame,
    plan: WalkForwardPlan,
    profile: ModelProfile,
    candidate: ResolvedCandidateProfile,
    candidate_adapter_factory: AdapterFactory,
    *,
    max_workers: int | None = None,
    seed_checkpoint_factory: Callable[[str], HMMSeedCheckpoint] | None = None,
    pca_raw_feature_order: tuple[str, ...] | None = None,
    pca_variance_threshold: float = 0.90,
) -> WalkForwardEvaluation:
    """Run one Gaussian prefix candidate through the shared walk-forward runner."""

    return run_walk_forward_candidate(
        source_rows,
        plan=plan,
        profile=profile,
        candidate=candidate,
        adapter_factory=candidate_adapter_factory,
        max_workers=max_workers,
        seed_checkpoint_factory=seed_checkpoint_factory,
        pca_raw_feature_order=pca_raw_feature_order,
        pca_variance_threshold=pca_variance_threshold,
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
    candidate_id: str = "gaussian_hmm_k2_full",
    candidate_evaluations: tuple[CandidateEvaluation, ...] = (),
    shared_timestamp_count: int = 0,
) -> PrefixEvaluation:
    return PrefixEvaluation(
        prefix_length=prefix_length,
        feature_order=feature_order,
        candidate_id=candidate_id,
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
    evaluation_sink: PrefixEvaluationSink | None = None,
    pca_raw_feature_order: tuple[str, ...] | None = None,
    pca_variance_threshold: float = 0.90,
    state_counts: tuple[int, ...] = _GAUSSIAN_STATE_COUNTS,
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
    state_counts = tuple(state_counts)
    if (
        not state_counts
        or state_counts != tuple(sorted(set(state_counts)))
        or any(state_count not in _GAUSSIAN_STATE_COUNTS for state_count in state_counts)
    ):
        raise ValueError("state_counts must be an ordered non-empty subset of K=2,3,4,5")
    build_id = teacher.source_build_id if source_build_id is None else source_build_id
    if build_id != teacher.source_build_id:
        raise ValueError("prefix search source build differs from the frozen teacher")
    universe = (
        tuple(column for column in source_rows.columns if column != _TIMESTAMP_COLUMN)
        if original_feature_universe is None
        else original_feature_universe
    )
    pca_order = validate_mandatory_pca_feature_universe(
        pca_raw_feature_order,
        universe,
    )
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
    prefix_lengths = tuple(range(MIN_PREFIX_LENGTH, upper_bound + 1))
    total_worker_budget = cpu_worker_count(max_workers)
    prefix_worker_limit = cpu_worker_count(max_workers, task_count=len(prefix_lengths))
    prefix_tasks = tuple(
        _PrefixSearchTask(
            prefix_length,
            source_rows,
            inner_plan,
            profile,
            teacher,
            ranked_features[:prefix_length],
            state_counts,
            universe,
            build_id,
            definition_hash,
            execution_hash,
            runner,
            total_worker_budget,
            seed_checkpoint_factory,
            pca_order,
            pca_variance_threshold,
        )
        for prefix_length in prefix_lengths
    )
    evaluated_prefixes: tuple[_EvaluatedPrefix, ...]
    if prefix_worker_limit == 1:
        evaluated_prefixes = tuple(_evaluate_prefix(task) for task in prefix_tasks)
    else:
        nested_limits = nested_worker_limits(total_worker_budget, prefix_worker_limit)
        scheduled_tasks = tuple(
            replace(task, max_workers=nested_limits[index % prefix_worker_limit])
            for index, task in enumerate(prefix_tasks)
        )
        use_processes = seed_checkpoint_factory is None and is_pickleable(runner)
        if use_processes:
            with cpu_process_pool(prefix_worker_limit) as executor:
                futures = tuple(
                    executor.submit(_evaluate_prefix_in_search_process, task)
                    for task in scheduled_tasks
                )
                evaluated_prefixes = tuple(future.result() for future in futures)
        elif seed_checkpoint_factory is not None and runner is run_prefix_gaussian_candidate:
            with ThreadPoolExecutor(max_workers=prefix_worker_limit) as executor:
                futures = tuple(executor.submit(_evaluate_prefix, task) for task in scheduled_tasks)
                evaluated_prefixes = tuple(future.result() for future in futures)
        else:
            raise RuntimeError(
                "parallel prefix search requires a pickleable CPU runner; "
                "set max_workers=1 for an explicitly serial custom runner"
            )
    prefix_results: list[PrefixEvaluation] = []
    for result in evaluated_prefixes:
        prefix_evaluation = result.evaluation
        if evaluation_sink is not None and result.selected_candidate is not None:
            try:
                evaluation_sink(
                    prefix_evaluation.prefix_length,
                    prefix_evaluation.candidate_id,
                    result.selected_candidate,
                )
            except (ValueError, TypeError) as exc:
                prefix_evaluation = _invalid_prefix(
                    prefix_evaluation.prefix_length,
                    prefix_evaluation.feature_order,
                    str(exc),
                    candidate_id=prefix_evaluation.candidate_id,
                )
        prefix_results.append(prefix_evaluation)
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
    "PrefixEvaluationSink",
    "evaluate_prefix_search",
    "run_prefix_gaussian_candidate",
    "search_ranked_prefixes",
    "select_prefix",
]
