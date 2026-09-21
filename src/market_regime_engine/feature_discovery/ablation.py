"""Deterministic one-feature-at-a-time ablation over a frozen SFFS tuple."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from market_regime_engine.feature_discovery.sffs import (
    FeatureSubsetScore,
    _canonical_subset,
    _score,
)


@dataclass(frozen=True, slots=True)
class AblationObservation:
    removed_feature: str | None
    remaining_features: tuple[str, ...]
    score: FeatureSubsetScore


@dataclass(frozen=True, slots=True)
class AblationResult:
    selected_features: tuple[str, ...]
    baseline: AblationObservation
    one_feature_results: tuple[AblationObservation, ...]

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


def run_one_feature_ablation(
    selected_features: tuple[str, ...],
    evaluate: SubsetEvaluator,
) -> AblationResult:
    """Evaluate the baseline and every exactly-one-feature removal."""

    selected = _canonical_subset(selected_features)
    baseline_score = _score(evaluate, selected)
    if baseline_score is None:
        raise ValueError("ablation baseline must be eligible under the same selector contract")
    baseline = AblationObservation(None, selected, baseline_score)
    results: list[AblationObservation] = []
    for feature in selected:
        remaining = tuple(item for item in selected if item != feature)
        if not remaining:
            raise ValueError("ablation requires a selected tuple with at least two features")
        score = _score(evaluate, remaining)
        if score is None:
            raise ValueError("ablation subset must use the same eligible selector contract")
        results.append(AblationObservation(feature, remaining, score))
    return AblationResult(selected, baseline, tuple(results))


__all__ = [
    "AblationObservation",
    "AblationResult",
    "run_one_feature_ablation",
]
