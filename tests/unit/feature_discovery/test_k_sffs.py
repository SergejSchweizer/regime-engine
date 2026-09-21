from market_regime_engine.feature_discovery.k_sffs import select_k_slot_sffs
from market_regime_engine.feature_discovery.sffs import FeatureSubsetScore


def test_k_sffs_keeps_each_gaussian_slot_independent_and_deterministic() -> None:
    def score(state_count: int, features: tuple[str, ...]) -> FeatureSubsetScore:
        values = {
            2: {("a",): 1.0, ("b",): 2.0, ("b", "a"): 3.0},
            3: {("a",): 4.0, ("b",): 2.0, ("a", "b"): 5.0},
        }
        return FeatureSubsetScore(features, values[state_count].get(features, 0.0))

    result = select_k_slot_sffs(
        ("a", "b"), score, state_counts=(2, 3), max_features=2, max_workers=1
    )

    assert tuple(item.state_count for item in result) == (2, 3)
    assert all(item.model_family == "gaussian_hmm" for item in result)
    assert result[0].selected_features == ("b", "a")
    assert result[1].selected_features == ("a", "b")
