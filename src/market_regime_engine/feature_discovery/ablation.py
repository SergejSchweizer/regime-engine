"""Deterministic one-feature-at-a-time ablation over a frozen SFFS tuple."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from market_regime_engine.feature_discovery.sffs import (
    FeatureSubsetScore,
    _canonical_subset,
)


def _require_text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty trimmed string")


@dataclass(frozen=True, slots=True)
class AblationObservation:
    removed_feature: str | None
    remaining_features: tuple[str, ...]
    score: FeatureSubsetScore
    hmm_evaluation: HMMSubsetEvaluation


@dataclass(frozen=True, slots=True)
class HMMSubsetEvaluation:
    """Evidence returned by one fresh HMM fit for one feature tuple."""

    score: FeatureSubsetScore
    model_family: str
    state_count: int
    selector_contract_hash: str
    fit_execution_hash: str

    def __post_init__(self) -> None:
        _require_text(self.model_family, "model_family")
        if isinstance(self.state_count, bool) or self.state_count < 2:
            raise ValueError("HMM ablation state_count must be at least two")
        _require_text(self.selector_contract_hash, "selector_contract_hash")
        _require_text(self.fit_execution_hash, "fit_execution_hash")


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


def run_one_feature_hmm_ablation(
    selected_features: tuple[str, ...],
    evaluate: HMMSubsetEvaluator,
    *,
    selector_contract_hash: str,
) -> AblationResult:
    """Refit and evaluate the baseline and every one-feature removal.

    The evaluator is required to return fresh HMM-fit evidence for every
    tuple.  A dimension-independent score alone is intentionally insufficient
    for final ablation acceptance.
    """

    selected = _canonical_subset(selected_features)
    _require_text(selector_contract_hash, "selector_contract_hash")

    def validate_evaluation(
        features: tuple[str, ...],
        result: HMMSubsetEvaluation | None,
        *,
        baseline: HMMSubsetEvaluation | None = None,
    ) -> HMMSubsetEvaluation:
        if result is None:
            raise ValueError("ablation subset must be eligible under the HMM selector contract")
        if result.score.feature_names != features:
            raise ValueError("HMM ablation score must identify exactly the evaluated tuple")
        if result.selector_contract_hash != selector_contract_hash:
            raise ValueError("HMM ablation changed the selector contract")
        if baseline is not None and (
            result.model_family != baseline.model_family
            or result.state_count != baseline.state_count
        ):
            raise ValueError("HMM ablation changed the model selector")
        return result

    baseline_evaluation = evaluate(selected)
    if baseline_evaluation is None:
        raise ValueError("ablation baseline must be eligible under the same selector contract")
    baseline_evaluation = validate_evaluation(selected, baseline_evaluation)
    baseline = AblationObservation(
        None,
        selected,
        baseline_evaluation.score,
        baseline_evaluation,
    )
    results: list[AblationObservation] = []
    fit_hashes = {baseline_evaluation.fit_execution_hash}
    for feature in selected:
        remaining = tuple(item for item in selected if item != feature)
        if not remaining:
            raise ValueError("ablation requires a selected tuple with at least two features")
        evaluation = validate_evaluation(
            remaining,
            evaluate(remaining),
            baseline=baseline_evaluation,
        )
        if evaluation.fit_execution_hash in fit_hashes:
            raise ValueError("each HMM ablation tuple must be refit independently")
        fit_hashes.add(evaluation.fit_execution_hash)
        results.append(AblationObservation(feature, remaining, evaluation.score, evaluation))
    return AblationResult(selected, baseline, tuple(results))


__all__ = [
    "AblationObservation",
    "AblationResult",
    "HMMSubsetEvaluation",
    "HMMSubsetEvaluator",
    "run_one_feature_hmm_ablation",
]
