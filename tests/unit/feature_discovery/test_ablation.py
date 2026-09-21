import pytest

from market_regime_engine.feature_discovery.ablation import (
    HMMSubsetEvaluation,
    run_one_feature_hmm_ablation,
)
from market_regime_engine.feature_discovery.sffs import FeatureSubsetScore


def test_ablation_evaluates_baseline_and_exactly_one_removed_feature_each() -> None:
    selected = ("a", "b", "c")
    fit_count = 0

    def evaluate(features: tuple[str, ...]) -> HMMSubsetEvaluation:
        nonlocal fit_count
        fit_count += 1
        return HMMSubsetEvaluation(
            FeatureSubsetScore(features, float(len(features))),
            "gaussian_hmm",
            2,
            "selector-v1",
            f"fit-{fit_count}",
        )

    result = run_one_feature_hmm_ablation(
        selected,
        evaluate,
        selector_contract_hash="selector-v1",
    )
    assert result.baseline.removed_feature is None
    assert result.baseline.remaining_features == selected
    assert tuple(item.removed_feature for item in result.one_feature_results) == selected
    assert tuple(item.remaining_features for item in result.one_feature_results) == (
        ("b", "c"),
        ("a", "c"),
        ("a", "b"),
    )


def test_ablation_uses_the_same_score_contract_and_rejects_ineligible_results() -> None:
    with pytest.raises(ValueError, match="baseline"):
        run_one_feature_hmm_ablation(
            ("a", "b"), lambda _features: None, selector_contract_hash="selector-v1"
        )

    def evaluate(features: tuple[str, ...]) -> HMMSubsetEvaluation | None:
        return (
            None
            if len(features) < 3
            else HMMSubsetEvaluation(
                FeatureSubsetScore(features, 1.0),
                "gaussian_hmm",
                2,
                "selector-v1",
                f"fit-{len(features)}",
            )
        )

    with pytest.raises(ValueError, match="HMM selector"):
        run_one_feature_hmm_ablation(
            ("a", "b", "c"), evaluate, selector_contract_hash="selector-v1"
        )


def test_ablation_requires_at_least_two_selected_features() -> None:
    with pytest.raises(ValueError, match="at least two"):
        run_one_feature_hmm_ablation(
            ("a",),
            lambda features: HMMSubsetEvaluation(
                FeatureSubsetScore(features, 1.0),
                "gaussian_hmm",
                2,
                "selector-v1",
                "fit-1",
            ),
            selector_contract_hash="selector-v1",
        )


def test_ablation_rejects_reusing_one_hmm_fit() -> None:
    def evaluate(features: tuple[str, ...]) -> HMMSubsetEvaluation:
        return HMMSubsetEvaluation(
            FeatureSubsetScore(features, 1.0),
            "gaussian_hmm",
            2,
            "selector-v1",
            "same-fit",
        )

    with pytest.raises(ValueError, match="refit independently"):
        run_one_feature_hmm_ablation(
            ("a", "b"),
            evaluate,
            selector_contract_hash="selector-v1",
        )
