"""Deterministic one-feature-at-a-time ablation over a frozen SFFS tuple."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from market_regime_engine.feature_discovery.sffs import (
    FeatureSubsetScore,
    _canonical_subset,
)
from market_regime_engine.runtime.cpu import cpu_worker_count
from market_regime_engine.runtime.processes import is_pickleable
from market_regime_engine.runtime.task_frontier import FrontierTask, SharedTaskFrontier


def _require_text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty trimmed string")


def _require_sha256(value: str, field: str) -> None:
    _require_text(value, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class AblationObservation:
    removed_feature: str | None
    remaining_features: tuple[str, ...]
    score: FeatureSubsetScore | None
    hmm_evaluation: HMMSubsetEvaluation
    ablation_loss: float | None = None


@dataclass(frozen=True, slots=True)
class HMMSubsetEvaluation:
    """Evidence returned by one fresh HMM fit for one feature tuple."""

    score: FeatureSubsetScore | None
    model_family: str
    state_count: int
    selector_contract_hash: str
    fit_execution_hash: str
    evaluation_plan_hash: str | None = None
    seed_identity: str | None = None
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.model_family, "model_family")
        if isinstance(self.state_count, bool) or self.state_count < 2:
            raise ValueError("HMM ablation state_count must be at least two")
        _require_sha256(self.selector_contract_hash, "selector_contract_hash")
        _require_sha256(self.fit_execution_hash, "fit_execution_hash")
        for value, field in (
            (self.evaluation_plan_hash, "evaluation_plan_hash"),
            (self.seed_identity, "seed_identity"),
        ):
            if value is not None:
                _require_sha256(value, field)
        if self.score is None:
            _require_text(self.invalid_reason or "", "invalid_reason")
        elif self.invalid_reason is not None:
            raise ValueError("valid HMM ablations cannot carry an invalid reason")


@dataclass(frozen=True, slots=True)
class AblationResult:
    selected_features: tuple[str, ...]
    baseline: AblationObservation
    one_feature_results: tuple[AblationObservation, ...]

    @property
    def selector_contract_hash(self) -> str:
        return self.baseline.hmm_evaluation.selector_contract_hash

    @property
    def model_family(self) -> str:
        return self.baseline.hmm_evaluation.model_family

    @property
    def state_count(self) -> int:
        return self.baseline.hmm_evaluation.state_count

    @property
    def fit_execution_hashes(self) -> tuple[str, ...]:
        return (
            self.baseline.hmm_evaluation.fit_execution_hash,
            *(item.hmm_evaluation.fit_execution_hash for item in self.one_feature_results),
        )

    @property
    def ablation_losses(self) -> tuple[float | None, ...]:
        return tuple(item.ablation_loss for item in self.one_feature_results)

    def __post_init__(self) -> None:
        expected = tuple(
            item for item in self.selected_features if item not in {self.baseline.removed_feature}
        )
        if self.baseline.removed_feature is not None:
            raise ValueError("ablation baseline must not remove a feature")
        if self.baseline.remaining_features != self.selected_features:
            raise ValueError("ablation baseline must evaluate the complete selected tuple")
        if len(self.one_feature_results) != len(self.selected_features):
            raise ValueError("ablation requires exactly one result per selected feature")
        removed = tuple(item.removed_feature for item in self.one_feature_results)
        if removed != self.selected_features or any(
            len(item.remaining_features) != len(self.selected_features) - 1
            or item.removed_feature in item.remaining_features
            for item in self.one_feature_results
        ):
            raise ValueError("ablation results must remove exactly one selected feature")
        if expected != self.selected_features:
            raise ValueError("ablation selected tuple is invalid")


SubsetEvaluator = Callable[[tuple[str, ...]], FeatureSubsetScore | None]
HMMSubsetEvaluator = Callable[[tuple[str, ...]], HMMSubsetEvaluation | None]


def _evaluate_ablation_in_frontier(
    task: FrontierTask[HMMSubsetEvaluator],
) -> HMMSubsetEvaluation | None:
    return task.payload(task.candidate_subset)


def run_one_feature_hmm_ablation(
    selected_features: tuple[str, ...],
    evaluate: HMMSubsetEvaluator,
    *,
    selector_contract_hash: str,
    max_workers: int | None = None,
    frontier: SharedTaskFrontier[HMMSubsetEvaluator, HMMSubsetEvaluation | None] | None = None,
) -> AblationResult:
    """Refit and evaluate the baseline and every one-feature removal.

    The evaluator is required to return fresh HMM-fit evidence for every
    tuple.  A dimension-independent score alone is intentionally insufficient
    for final ablation acceptance.
    """

    selected = _canonical_subset(selected_features)
    _require_sha256(selector_contract_hash, "selector_contract_hash")

    def validate_evaluation(
        features: tuple[str, ...],
        result: HMMSubsetEvaluation | None,
        *,
        baseline: HMMSubsetEvaluation | None = None,
    ) -> HMMSubsetEvaluation:
        if result is None:
            raise ValueError("ablation subset must be eligible under the HMM selector contract")
        if result.score is not None and result.score.feature_names != features:
            raise ValueError("HMM ablation score must identify exactly the evaluated tuple")
        if result.selector_contract_hash != selector_contract_hash:
            raise ValueError("HMM ablation changed the selector contract")
        if baseline is not None and (
            result.model_family != baseline.model_family
            or result.state_count != baseline.state_count
        ):
            raise ValueError("HMM ablation changed the model selector")
        if baseline is not None:
            for field in ("evaluation_plan_hash", "seed_identity"):
                expected = getattr(baseline, field)
                actual = getattr(result, field)
                if expected is not None and actual != expected:
                    raise ValueError(f"ablation changed the {field}")
        return result

    baseline_evaluation = evaluate(selected)
    if baseline_evaluation is None:
        raise ValueError("ablation baseline must be eligible under the same selector contract")
    baseline_evaluation = validate_evaluation(selected, baseline_evaluation)
    if baseline_evaluation.score is None:
        raise ValueError("ablation baseline must be eligible under the same selector contract")
    baseline = AblationObservation(
        None,
        selected,
        baseline_evaluation.score,
        baseline_evaluation,
    )
    results: list[AblationObservation] = []
    fit_hashes = {baseline_evaluation.fit_execution_hash}
    removal_tasks = tuple(
        (feature, tuple(item for item in selected if item != feature)) for feature in selected
    )
    if any(not remaining for _, remaining in removal_tasks):
        raise ValueError("ablation requires a selected tuple with at least two features")
    parallel = (
        frontier is None
        and max_workers != 1
        and os.environ.get("REGIME_CPU_PROCESS_WORKER") != "1"
        and is_pickleable(evaluate)
    )
    worker_limit = cpu_worker_count(max_workers, task_count=len(removal_tasks)) if parallel else 1
    frontier_context: Any = (
        nullcontext(frontier)
        if frontier is not None
        else SharedTaskFrontier(worker_limit)
        if parallel
        else nullcontext(None)
    )
    with frontier_context as execution_frontier:
        if execution_frontier is not None:
            frontier_tasks = tuple(
                FrontierTask(
                    task_id=f"ablation:{index}:{feature}",
                    state_count=baseline_evaluation.state_count,
                    fold_id="ablation",
                    candidate_subset=remaining,
                    seed=index,
                    profile_hash=selector_contract_hash,
                    matrix_identity="ablation-candidate-frontier",
                    row_indices=(0,),
                    column_indices=(0,),
                    payload=evaluate,
                )
                for index, (feature, remaining) in enumerate(removal_tasks)
            )
            frontier_result = execution_frontier.map(frontier_tasks, _evaluate_ablation_in_frontier)
            by_task_id = {task.task_id: item for task, item in frontier_result.values}
            evaluated_removals = tuple(by_task_id[task.task_id] for task in frontier_tasks)
        else:
            evaluated_removals = tuple(evaluate(remaining) for _, remaining in removal_tasks)
    for (feature, remaining), raw_evaluation in zip(removal_tasks, evaluated_removals, strict=True):
        evaluation = validate_evaluation(remaining, raw_evaluation, baseline=baseline_evaluation)
        if evaluation.fit_execution_hash in fit_hashes:
            raise ValueError("each HMM ablation tuple must be refit independently")
        fit_hashes.add(evaluation.fit_execution_hash)
        loss = (
            None
            if evaluation.score is None
            else baseline_evaluation.score.value - evaluation.score.value
        )
        results.append(AblationObservation(feature, remaining, evaluation.score, evaluation, loss))
    return AblationResult(selected, baseline, tuple(results))


__all__ = [
    "AblationObservation",
    "AblationResult",
    "HMMSubsetEvaluation",
    "HMMSubsetEvaluator",
    "run_one_feature_hmm_ablation",
]
