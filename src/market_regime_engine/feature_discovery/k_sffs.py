"""Independent, deterministic SFFS runs for the four production HMM K slots."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from market_regime_engine.evaluations.process_parallel import is_pickleable
from market_regime_engine.evaluations.task_frontier import SharedTaskFrontier
from market_regime_engine.feature_discovery.feature_roles import SFFS_MAX_FEATURES
from market_regime_engine.feature_discovery.sffs import (
    FeatureSubsetScore,
    SFFSResult,
    select_sffs,
)

LEGAL_K = (2, 3, 4, 5)
GAUSSIAN_HMM = "gaussian_hmm"
KSubsetScore = Callable[[int, tuple[str, ...]], FeatureSubsetScore | None]


@dataclass(frozen=True, slots=True)
class _FixedKScore:
    evaluator: KSubsetScore
    state_count: int

    def __call__(self, features: tuple[str, ...]) -> FeatureSubsetScore | None:
        result = self.evaluator(self.state_count, features)
        return self._validate(result)

    def evaluate_many(
        self, feature_sets: Iterable[tuple[str, ...]]
    ) -> tuple[FeatureSubsetScore | None, ...]:
        batch_evaluator = getattr(self.evaluator, "evaluate_many", None)
        if callable(batch_evaluator):
            results = tuple(batch_evaluator(feature_sets, state_count=self.state_count))
        else:
            results = tuple(self(features) for features in feature_sets)
        return tuple(self._validate(result) for result in results)

    def _validate(self, result: FeatureSubsetScore | None) -> FeatureSubsetScore | None:
        if result is None:
            return None
        if result.state_count != self.state_count or result.model_family != GAUSSIAN_HMM:
            raise ValueError(
                "K SFFS evaluator must return a Gaussian-HMM score for the requested K"
            )
        return result


@dataclass(frozen=True, slots=True)
class KSlotSFFSResult:
    """One fixed-K selector result and its immutable selector identity."""

    state_count: int
    model_family: str
    sffs: SFFSResult

    def __post_init__(self) -> None:
        if self.state_count not in LEGAL_K:
            raise ValueError("SFFS state_count must be 2, 3, 4, or 5")
        if self.model_family != GAUSSIAN_HMM:
            raise ValueError("feature SFFS must use the Gaussian full-covariance HMM")

    @property
    def selected_features(self) -> tuple[str, ...]:
        return self.sffs.selected_features


def select_k_slot_sffs(
    candidates: Iterable[str],
    evaluate_gaussian_subset: KSubsetScore,
    *,
    state_counts: Iterable[int] = LEGAL_K,
    max_features: int = SFFS_MAX_FEATURES,
    max_workers: int | None = None,
    frontier: SharedTaskFrontier[Any, Any] | None = None,
) -> tuple[KSlotSFFSResult, ...]:
    """Run an independent Gaussian full-covariance SFFS search for every K.

    ``evaluate_gaussian_subset`` is the sole model-fitting seam and receives
    the fixed K explicitly.  The returned tuples are later reusable by every
    emission family at that same K; no other family can influence selection.
    """

    requested = tuple(state_counts)
    if requested != tuple(sorted(set(requested))) or any(k not in LEGAL_K for k in requested):
        raise ValueError("state_counts must be a unique ordered subset of (2, 3, 4, 5)")
    if not requested:
        raise ValueError("at least one state_count is required")

    candidate_tuple = tuple(candidates)
    results: list[KSlotSFFSResult] = []
    use_frontier = frontier is not None or (
        max_workers != 1 and is_pickleable(evaluate_gaussian_subset)
    )
    frontier_context: Any = (
        nullcontext(frontier)
        if frontier is not None
        else SharedTaskFrontier(max_workers)
        if use_frontier
        else nullcontext(None)
    )
    with frontier_context as frontier:

        def select_one(state_count: int) -> KSlotSFFSResult:
            fixed_k_score = _FixedKScore(evaluate_gaussian_subset, state_count)
            selected = select_sffs(
                candidate_tuple,
                fixed_k_score,
                max_features=max_features,
                max_workers=max_workers,
                frontier=frontier,
                frontier_state_count=state_count,
            )
            return KSlotSFFSResult(state_count, GAUSSIAN_HMM, selected)

        if frontier is not None and len(requested) > 1:
            # K slots are independent coordinators.  Threads only submit and
            # collect work; all CPU-bound fitting remains in the shared
            # process frontier, while each slot's SFFS decisions stay ordered.
            with ThreadPoolExecutor(max_workers=len(requested)) as coordinator:
                results.extend(coordinator.map(select_one, requested))
        else:
            results.extend(select_one(state_count) for state_count in requested)
    return tuple(results)


__all__ = ["GAUSSIAN_HMM", "LEGAL_K", "KSlotSFFSResult", "select_k_slot_sffs"]
