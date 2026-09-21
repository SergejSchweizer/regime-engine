"""Deterministic, process-safe contract for feature-subset SFFS."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from math import isfinite

from market_regime_engine.evaluations.process_parallel import cpu_process_pool, is_pickleable
from market_regime_engine.feature_discovery.feature_roles import SFFS_MAX_FEATURES
from market_regime_engine.feature_discovery.feature_subset_score import SCORE_ABS_TOLERANCE
from market_regime_engine.runtime.cpu import cpu_worker_count

DIMENSION_INDEPENDENT_SCORE = "dimension_independent_feature_subset_score"


@dataclass(frozen=True, slots=True)
class FeatureSubsetScore:
    feature_names: tuple[str, ...]
    value: float
    metric: str = DIMENSION_INDEPENDENT_SCORE

    def __post_init__(self) -> None:
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature-subset score requires a non-empty unique feature tuple")
        if not isfinite(self.value):
            raise ValueError("feature-subset score must be finite")
        if self.metric != DIMENSION_INDEPENDENT_SCORE:
            raise ValueError(
                "SFFS cannot compare raw likelihood, AIC, BIC, or other dimension-dependent scores"
            )


@dataclass(frozen=True, slots=True)
class SFFSStep:
    action: str
    selected_features: tuple[str, ...]
    score: FeatureSubsetScore


@dataclass(frozen=True, slots=True)
class SFFSResult:
    selected_features: tuple[str, ...]
    best_singleton: str
    steps: tuple[SFFSStep, ...]


ScoreFunction = Callable[[tuple[str, ...]], FeatureSubsetScore | None]


def _score_in_process(
    task: tuple[ScoreFunction, tuple[str, ...]],
) -> FeatureSubsetScore | None:
    score, features = task
    return _score(score, features)


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
    if candidate_score.value != incumbent_score.value:
        return candidate_score.value > incumbent_score.value
    candidate_key = tuple(candidate_order[name] for name in candidate_names)
    incumbent_key = tuple(candidate_order[name] for name in incumbent_names)
    return candidate_key < incumbent_key


def select_sffs(
    candidates: Iterable[str],
    score: ScoreFunction,
    *,
    max_features: int = SFFS_MAX_FEATURES,
    max_workers: int | None = None,
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
    pool_context = (
        cpu_process_pool(worker_limit) if parallel and worker_limit > 1 else nullcontext()
    )
    with pool_context as executor:

        def evaluate_many(
            feature_sets: tuple[tuple[str, ...], ...],
        ) -> tuple[FeatureSubsetScore | None, ...]:
            tasks = tuple((score, features) for features in feature_sets)
            if executor is None:
                return tuple(_score_in_process(task) for task in tasks)
            return tuple(executor.map(_score_in_process, tasks))

        singleton_sets = tuple((name,) for name in candidate_tuple)
        singleton_scores = evaluate_many(singleton_sets)
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
            forward_scores = evaluate_many(forward_sets)
            forward: tuple[tuple[str, ...], FeatureSubsetScore] | None = None
            for proposal, evaluated in zip(forward_sets, forward_scores, strict=True):
                if evaluated is not None and _better(
                    (proposal, evaluated), forward, candidate_order
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
                backward_scores = evaluate_many(backward_sets)
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
    )


__all__ = [
    "DIMENSION_INDEPENDENT_SCORE",
    "FeatureSubsetScore",
    "SFFSResult",
    "SFFSStep",
    "select_sffs",
]
