"""Deterministic, process-safe contract for feature-subset SFFS."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any

from market_regime_engine.feature_discovery.feature_roles import SFFS_MAX_FEATURES
from market_regime_engine.feature_discovery.feature_subset_score import (
    SCORE_ABS_TOLERANCE,
    FeatureSubsetCandidate,
    FeatureSubsetFoldEvidence,
    score_feature_subset,
    to_sffs_score,
)
from market_regime_engine.runtime.cpu import cpu_worker_count
from market_regime_engine.runtime.processes import is_pickleable
from market_regime_engine.runtime.task_frontier import FrontierTask, SharedTaskFrontier
from market_regime_engine.training.multistart import (
    MultistartBatchJob,
    MultistartResult,
    run_multistart_batch,
)

DIMENSION_INDEPENDENT_SCORE = "dimension_independent_feature_subset_score"


@dataclass(frozen=True, slots=True)
class FeatureSubsetScore:
    feature_names: tuple[str, ...]
    value: float
    metric: str = DIMENSION_INDEPENDENT_SCORE
    forecast_score: float | None = None
    worst_fold_forecast_score: float | None = None
    calibration_score: float | None = None
    stability_score: float | None = None
    robustness_score: float | None = None
    model_family: str | None = None
    state_count: int | None = None

    def __post_init__(self) -> None:
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature-subset score requires a non-empty unique feature tuple")
        if not isfinite(self.value):
            raise ValueError("feature-subset score must be finite")
        if self.metric != DIMENSION_INDEPENDENT_SCORE:
            raise ValueError(
                "SFFS cannot compare raw likelihood, AIC, BIC, or other dimension-dependent scores"
            )
        for field in (
            "forecast_score",
            "worst_fold_forecast_score",
            "calibration_score",
            "stability_score",
            "robustness_score",
        ):
            component = getattr(self, field)
            if component is not None and (not isfinite(component) or not 0.0 <= component <= 1.0):
                raise ValueError(f"{field} must be finite and in [0, 1]")
        if self.model_family is not None and self.model_family != "gaussian_hmm":
            raise ValueError("SFFS feature scores must use the Gaussian HMM selector")
        if self.state_count is not None and self.state_count not in (2, 3, 4, 5):
            raise ValueError("SFFS feature score state_count must be 2, 3, 4, or 5")


@dataclass(frozen=True, slots=True)
class SFFSStep:
    action: str
    selected_features: tuple[str, ...]
    score: FeatureSubsetScore


@dataclass(frozen=True, slots=True)
class SFFSEvaluation:
    """One candidate evaluation, including candidates rejected by the score gate."""

    action: str
    candidate: tuple[str, ...]
    selected_features: tuple[str, ...]
    score: FeatureSubsetScore | None


@dataclass(frozen=True, slots=True)
class SFFSResult:
    selected_features: tuple[str, ...]
    best_singleton: str
    steps: tuple[SFFSStep, ...]
    evaluations: tuple[SFFSEvaluation, ...] = ()


ScoreFunction = Callable[[tuple[str, ...]], FeatureSubsetScore | None]


@dataclass(frozen=True, slots=True)
class FrontierFoldJob:
    """One candidate-subset/inner-fold job whose seeds share a frontier."""

    candidate_subset: tuple[str, ...]
    fold_id: str
    job: MultistartBatchJob

    def __post_init__(self) -> None:
        if self.job.job_id != f"{self.fold_id}:{','.join(self.candidate_subset)}":
            raise ValueError("frontier fold job identity is not canonical")


@dataclass(frozen=True, slots=True)
class FrontierFeatureSubsetEvaluator:
    """Build and aggregate real HMM seed jobs for SFFS candidate tuples."""

    job_factory: Callable[[int, tuple[str, ...]], tuple[FrontierFoldJob, ...]]
    evidence_factory: Callable[[FrontierFoldJob, MultistartResult], FeatureSubsetFoldEvidence]
    feature_order_hash: str | Callable[[tuple[str, ...]], str]
    source_build_id: str
    evaluation_plan_hash: str
    latest_fold_id: str
    state_count: int
    max_workers: int | None = None
    frontier: SharedTaskFrontier[Any, Any] | None = None

    def evaluate_many(
        self,
        feature_sets: Iterable[tuple[str, ...]],
        *,
        state_count: int | None = None,
    ) -> tuple[FeatureSubsetScore | None, ...]:
        requested_state_count = self.state_count if state_count is None else state_count
        if requested_state_count not in (2, 3, 4, 5):
            raise ValueError("frontier SFFS state_count must be 2, 3, 4, or 5")
        candidates = tuple(feature_sets)
        entries = tuple(
            entry
            for candidate in candidates
            for entry in self.job_factory(requested_state_count, candidate)
        )
        if not entries and candidates:
            return tuple(None for _candidate in candidates)
        if len({entry.job.job_id for entry in entries}) != len(entries):
            raise ValueError("frontier SFFS fold job IDs must be unique")
        results = run_multistart_batch(
            tuple(entry.job for entry in entries),
            max_workers=self.max_workers,
            # A failed multistart gate invalidates only this candidate/fold.
            # The serial callback path has the same candidate-local contract;
            # allowing the batch to return None keeps one bad candidate from
            # aborting the complete outer fold.
            allow_invalid=True,
            frontier=self.frontier,
        )
        evidence_by_candidate: dict[tuple[str, ...], list[FeatureSubsetFoldEvidence]] = {
            candidate: [] for candidate in candidates
        }
        invalid_candidates: set[tuple[str, ...]] = set()
        for entry, result in zip(entries, results, strict=True):
            if result is None:
                invalid_candidates.add(entry.candidate_subset)
                continue
            evidence_by_candidate[entry.candidate_subset].append(
                self.evidence_factory(entry, result)
            )
        scores: list[FeatureSubsetScore | None] = []
        for candidate in candidates:
            if candidate in invalid_candidates:
                scores.append(None)
                continue
            evidence = tuple(
                sorted(evidence_by_candidate[candidate], key=lambda item: item.fold_id)
            )
            if not evidence:
                scores.append(None)
                continue
            breakdown = score_feature_subset(
                FeatureSubsetCandidate(
                    candidate,
                    (
                        self.feature_order_hash(candidate)
                        if callable(self.feature_order_hash)
                        else self.feature_order_hash
                    ),
                    self.source_build_id,
                    self.evaluation_plan_hash,
                    evidence,
                ),
                latest_fold_id=self.latest_fold_id,
            )
            score = to_sffs_score(breakdown)
            scores.append(
                None
                if score is None
                else replace(
                    score,
                    model_family="gaussian_hmm",
                    state_count=requested_state_count,
                )
            )
        return tuple(scores)

    def __call__(self, features: tuple[str, ...]) -> FeatureSubsetScore | None:
        return self.evaluate_many((features,))[0]


def _score_in_process(
    task: tuple[ScoreFunction, tuple[str, ...]],
) -> FeatureSubsetScore | None:
    score, features = task
    return _score(score, features)


def _score_in_frontier(
    task: FrontierTask[ScoreFunction],
) -> FeatureSubsetScore | None:
    return _score(task.payload, task.candidate_subset)


def _canonical_subset(names: Iterable[str]) -> tuple[str, ...]:
    result = tuple(names)
    if not result or len(result) != len(set(result)):
        raise ValueError("SFFS feature candidates must be non-empty and unique")
    if any(not name or name.strip() != name for name in result):
        raise ValueError("SFFS feature candidates must be non-empty trimmed names")
    return result


def _score(score: ScoreFunction, features: tuple[str, ...]) -> FeatureSubsetScore | None:
    result = score(features)
    if result is None:
        return None
    if result.feature_names != features:
        raise ValueError("SFFS score must identify exactly the evaluated feature tuple")
    return result


def _better(
    candidate: tuple[tuple[str, ...], FeatureSubsetScore],
    incumbent: tuple[tuple[str, ...], FeatureSubsetScore] | None,
    candidate_order: dict[str, int],
) -> bool:
    if incumbent is None:
        return True
    candidate_names, candidate_score = candidate
    incumbent_names, incumbent_score = incumbent
    candidate_rank_key = (
        candidate_score.value,
        candidate_score.forecast_score if candidate_score.forecast_score is not None else 0.0,
        candidate_score.worst_fold_forecast_score
        if candidate_score.worst_fold_forecast_score is not None
        else 0.0,
        candidate_score.calibration_score if candidate_score.calibration_score is not None else 0.0,
        candidate_score.stability_score if candidate_score.stability_score is not None else 0.0,
        candidate_score.robustness_score if candidate_score.robustness_score is not None else 0.0,
        -len(candidate_names),
    )
    incumbent_rank_key = (
        incumbent_score.value,
        incumbent_score.forecast_score if incumbent_score.forecast_score is not None else 0.0,
        incumbent_score.worst_fold_forecast_score
        if incumbent_score.worst_fold_forecast_score is not None
        else 0.0,
        incumbent_score.calibration_score if incumbent_score.calibration_score is not None else 0.0,
        incumbent_score.stability_score if incumbent_score.stability_score is not None else 0.0,
        incumbent_score.robustness_score if incumbent_score.robustness_score is not None else 0.0,
        -len(incumbent_names),
    )
    if candidate_rank_key != incumbent_rank_key:
        return candidate_rank_key > incumbent_rank_key
    candidate_order_key = tuple(candidate_order[name] for name in candidate_names)
    incumbent_order_key = tuple(candidate_order[name] for name in incumbent_names)
    return candidate_order_key < incumbent_order_key


def select_sffs(
    candidates: Iterable[str],
    score: ScoreFunction,
    *,
    max_features: int = SFFS_MAX_FEATURES,
    max_workers: int | None = None,
    frontier: SharedTaskFrontier[ScoreFunction, FeatureSubsetScore | None] | None = None,
    frontier_state_count: int = 2,
) -> SFFSResult:
    """Run deterministic sequential floating forward selection.

    The score callback is intentionally the only model-specific seam.  It
    must return the same dimension-independent subset score for every feature
    dimension; a callback returning raw PLL/AIC/BIC is rejected by
    :class:`FeatureSubsetScore`.
    """

    candidate_tuple = _canonical_subset(candidates)
    if max_features < 1 or max_features > SFFS_MAX_FEATURES:
        raise ValueError("SFFS max_features must be between 1 and 10")
    if max_features > len(candidate_tuple):
        max_features = len(candidate_tuple)
    candidate_order = {name: index for index, name in enumerate(candidate_tuple)}

    parallel = max_workers != 1 and is_pickleable(score)
    worker_limit = cpu_worker_count(max_workers, task_count=len(candidate_tuple)) if parallel else 1
    if frontier is not None and frontier_state_count not in (2, 3, 4, 5):
        raise ValueError("frontier_state_count must be 2, 3, 4, or 5")
    frontier_context: Any = (
        SharedTaskFrontier(worker_limit)
        if frontier is None and parallel and worker_limit > 1
        else nullcontext(frontier)
    )
    with frontier_context as execution_frontier:
        evaluations: list[SFFSEvaluation] = []

        def evaluate_many(
            feature_sets: tuple[tuple[str, ...], ...],
            *,
            action: str,
            selected_features: tuple[str, ...],
        ) -> tuple[FeatureSubsetScore | None, ...]:
            batch_evaluator = getattr(score, "evaluate_many", None)
            if callable(batch_evaluator):
                results = tuple(batch_evaluator(feature_sets))
                if len(results) != len(feature_sets):
                    raise ValueError("frontier SFFS evaluator returned the wrong result count")
                for features, evaluated in zip(feature_sets, results, strict=True):
                    if evaluated is not None and evaluated.feature_names != features:
                        raise ValueError(
                            "frontier SFFS score must identify exactly the evaluated feature tuple"
                        )
            else:
                tasks = tuple((score, features) for features in feature_sets)
                active_frontier = frontier or execution_frontier
                if active_frontier is not None:
                    frontier_tasks = tuple(
                        FrontierTask(
                            task_id=f"{action}:{index}:{','.join(features)}",
                            state_count=frontier_state_count,
                            fold_id="sffs",
                            candidate_subset=features,
                            seed=index,
                            profile_hash="a" * 64,
                            matrix_identity="sffs-candidate-frontier",
                            row_indices=(0,),
                            column_indices=(0,),
                            payload=score,
                        )
                        for index, features in enumerate(feature_sets)
                    )
                    frontier_result = active_frontier.map(frontier_tasks, _score_in_frontier)
                    by_task_id = {task.task_id: item for task, item in frontier_result.values}
                    results = tuple(by_task_id[task.task_id] for task in frontier_tasks)
                else:
                    results = tuple(_score_in_process(task) for task in tasks)
            evaluations.extend(
                SFFSEvaluation(action, features, selected_features, evaluated)
                for features, evaluated in zip(feature_sets, results, strict=True)
            )
            return results

        singleton_sets = tuple((name,) for name in candidate_tuple)
        singleton_scores = evaluate_many(singleton_sets, action="singleton", selected_features=())
        best: tuple[tuple[str, ...], FeatureSubsetScore] | None = None
        for features, evaluated in zip(singleton_sets, singleton_scores, strict=True):
            if evaluated is not None and _better((features, evaluated), best, candidate_order):
                best = (features, evaluated)
        if best is None:
            raise ValueError("SFFS requires at least one eligible singleton")

        selected, selected_score = best
        steps = [SFFSStep("start", selected, selected_score)]
        visited: set[tuple[str, ...]] = {selected}
        while len(selected) < max_features:
            forward_sets = tuple(
                (*selected, name)
                for name in candidate_tuple
                if name not in selected and (*selected, name) not in visited
            )
            forward_scores = evaluate_many(
                forward_sets, action="forward", selected_features=selected
            )
            forward: tuple[tuple[str, ...], FeatureSubsetScore] | None = None
            for proposal, evaluated in zip(forward_sets, forward_scores, strict=True):
                if (
                    evaluated is not None
                    and evaluated.value > selected_score.value + SCORE_ABS_TOLERANCE
                    and _better((proposal, evaluated), forward, candidate_order)
                ):
                    forward = (proposal, evaluated)
            if forward is None:
                break
            selected, selected_score = forward
            visited.add(selected)
            steps.append(SFFSStep("add", selected, selected_score))

            while len(selected) > 1:
                backward_sets = tuple(
                    tuple(item for item in selected if item != name) for name in selected
                )
                backward_scores = evaluate_many(
                    backward_sets, action="backward", selected_features=selected
                )
                backward: tuple[tuple[str, ...], FeatureSubsetScore] | None = None
                for proposal, evaluated in zip(backward_sets, backward_scores, strict=True):
                    if (
                        evaluated is not None
                        and evaluated.value > selected_score.value + SCORE_ABS_TOLERANCE
                        and _better((proposal, evaluated), backward, candidate_order)
                    ):
                        backward = (proposal, evaluated)
                if backward is None:
                    break
                selected, selected_score = backward
                visited.add(selected)
                steps.append(SFFSStep("remove", selected, selected_score))

    return SFFSResult(
        selected_features=selected,
        best_singleton=best[0][0],
        steps=tuple(steps),
        evaluations=tuple(evaluations),
    )


__all__ = [
    "DIMENSION_INDEPENDENT_SCORE",
    "FeatureSubsetScore",
    "FrontierFeatureSubsetEvaluator",
    "FrontierFoldJob",
    "SFFSEvaluation",
    "SFFSResult",
    "SFFSStep",
    "select_sffs",
]
