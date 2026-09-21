"""Independent, deterministic SFFS runs for the four production HMM K slots."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial

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
    for state_count in requested:
        fixed_k_score = partial(evaluate_gaussian_subset, state_count)
        selected = select_sffs(
            candidate_tuple,
            fixed_k_score,
            max_features=max_features,
            max_workers=max_workers,
        )
        results.append(KSlotSFFSResult(state_count, GAUSSIAN_HMM, selected))
    return tuple(results)


__all__ = ["GAUSSIAN_HMM", "LEGAL_K", "KSlotSFFSResult", "select_k_slot_sffs"]
