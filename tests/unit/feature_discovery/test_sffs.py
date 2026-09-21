import pytest

from market_regime_engine.feature_discovery.sffs import (
    DIMENSION_INDEPENDENT_SCORE,
    FeatureSubsetScore,
    select_sffs,
)


def _picklable_score(features: tuple[str, ...]) -> FeatureSubsetScore:
    values = {
        ("a",): 1.0,
        ("b",): 2.0,
        ("b", "a"): 3.0,
    }
    return FeatureSubsetScore(features, values.get(features, 0.0))


def test_sffs_starts_at_best_singleton_and_obeys_hard_cap() -> None:
    candidates = ("a", "b", "c", "d")
    values = {
        ("a",): 1.0,
        ("b",): 3.0,
        ("c",): 2.0,
        ("d",): 0.0,
        ("b", "a"): 4.0,
        ("b", "c"): 5.0,
        ("b", "d"): 3.5,
        ("b", "c", "a"): 5.5,
        ("b", "c", "d"): 5.1,
    }

    def score(features: tuple[str, ...]) -> FeatureSubsetScore:
        return FeatureSubsetScore(features, values.get(features, 1.0))

    result = select_sffs(candidates, score, max_features=3)
    assert result.best_singleton == "b"
    assert result.selected_features == ("b", "c", "a")
    assert len(result.selected_features) == 3
    assert result.steps[0].action == "start"


def test_sffs_floating_step_can_remove_a_feature_when_score_improves() -> None:
    values = {
        ("a",): 1.0,
        ("b",): 3.0,
        ("c",): 2.0,
        ("b", "a"): 4.0,
        ("b", "c"): 2.5,
        ("b", "a", "c"): 3.0,
    }

    def score(features: tuple[str, ...]) -> FeatureSubsetScore:
        return FeatureSubsetScore(features, values.get(features, 0.0))

    result = select_sffs(("a", "b", "c"), score, max_features=3)
    assert result.selected_features == ("b", "a")
    assert any(step.action == "remove" for step in result.steps)


def test_sffs_rejects_non_dimension_independent_scores_and_no_singletons() -> None:
    with pytest.raises(ValueError, match="dimension-dependent"):
        FeatureSubsetScore(("a",), 1.0, metric="raw_log_likelihood")
    with pytest.raises(ValueError, match="eligible singleton"):
        select_sffs(("a", "b"), lambda _features: None)
    with pytest.raises(ValueError, match="between 1 and 10"):
        select_sffs(("a",), lambda features: FeatureSubsetScore(features, 1.0), max_features=11)


def test_sffs_score_contract_uses_the_canonical_metric() -> None:
    score = FeatureSubsetScore(("a",), 1.0)
    assert score.metric == DIMENSION_INDEPENDENT_SCORE


def test_sffs_uses_process_workers_for_picklable_score_and_preserves_order() -> None:
    result = select_sffs(("a", "b"), _picklable_score, max_features=2, max_workers=2)
    assert result.selected_features == ("b", "a")
    assert tuple(step.action for step in result.steps) == ("start", "add")
