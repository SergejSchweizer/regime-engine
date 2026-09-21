import pytest

from market_regime_engine.feature_discovery.k_sffs import select_k_slot_sffs
from market_regime_engine.feature_discovery.sffs import FeatureSubsetScore


def _picklable_k_score(state_count: int, features: tuple[str, ...]) -> FeatureSubsetScore:
    value = float(state_count + len(features))
    return FeatureSubsetScore(
        features,
        value,
        model_family="gaussian_hmm",
        state_count=state_count,
    )


def test_k_sffs_keeps_each_gaussian_slot_independent_and_deterministic() -> None:
    def score(state_count: int, features: tuple[str, ...]) -> FeatureSubsetScore:
        values = {
            2: {("a",): 1.0, ("b",): 2.0, ("b", "a"): 3.0},
            3: {("a",): 4.0, ("b",): 2.0, ("a", "b"): 5.0},
        }
        return FeatureSubsetScore(
            features,
            values[state_count].get(features, 0.0),
            model_family="gaussian_hmm",
            state_count=state_count,
        )

    result = select_k_slot_sffs(
        ("a", "b"), score, state_counts=(2, 3), max_features=2, max_workers=1
    )

    assert tuple(item.state_count for item in result) == (2, 3)
    assert all(item.model_family == "gaussian_hmm" for item in result)
    assert result[0].selected_features == ("b", "a")
    assert result[1].selected_features == ("a", "b")


def test_k_sffs_rejects_a_score_from_the_wrong_model_or_k() -> None:
    def wrong_score(state_count: int, features: tuple[str, ...]) -> FeatureSubsetScore:
        return FeatureSubsetScore(features, 1.0, model_family="gaussian_hmm", state_count=5)

    with pytest.raises(ValueError, match="Gaussian-HMM score"):
        select_k_slot_sffs(("a",), wrong_score, state_counts=(2,), max_workers=1)


def test_k_sffs_reuses_the_shared_frontier_for_pickleable_scores() -> None:
    result = select_k_slot_sffs(
        ("a", "b"),
        _picklable_k_score,
        state_counts=(2, 3, 4, 5),
        max_features=2,
        max_workers=2,
    )
    assert tuple(item.state_count for item in result) == (2, 3, 4, 5)
    assert all(item.selected_features == ("a", "b") for item in result)
