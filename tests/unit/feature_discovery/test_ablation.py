import pytest

from market_regime_engine.feature_discovery.ablation import run_one_feature_ablation
from market_regime_engine.feature_discovery.sffs import FeatureSubsetScore


def test_ablation_evaluates_baseline_and_exactly_one_removed_feature_each() -> None:
    selected = ("a", "b", "c")

    def evaluate(features: tuple[str, ...]) -> FeatureSubsetScore:
        return FeatureSubsetScore(features, float(len(features)))

    result = run_one_feature_ablation(selected, evaluate)
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
        run_one_feature_ablation(("a", "b"), lambda _features: None)

    def evaluate(features: tuple[str, ...]) -> FeatureSubsetScore | None:
        return None if len(features) < 3 else FeatureSubsetScore(features, 1.0)

    with pytest.raises(ValueError, match="same eligible"):
        run_one_feature_ablation(("a", "b", "c"), evaluate)


def test_ablation_requires_at_least_two_selected_features() -> None:
    with pytest.raises(ValueError, match="at least two"):
        run_one_feature_ablation(("a",), lambda features: FeatureSubsetScore(features, 1.0))
