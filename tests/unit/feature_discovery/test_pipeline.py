from market_regime_engine.feature_discovery.feature_roles import (
    TEMPORAL_KEY,
    build_feature_role_contract,
)
from market_regime_engine.feature_discovery.pipeline import run_canonical_feature_selection
from market_regime_engine.feature_discovery.sffs import FeatureSubsetScore


def test_canonical_pipeline_composes_train_only_stages_in_order() -> None:
    names = (
        "vix_log_level",
        "us_10y_log_level",
        "vix_delta_1obs",
        "vix_delta_5obs",
        "usd_broad_delta_1obs",
    )
    contract = build_feature_role_contract((TEMPORAL_KEY, *names))
    rows = tuple(float(index) for index in range(30))
    feature_values = {
        "vix_log_level": rows,
        "us_10y_log_level": tuple(float((index % 7) ** 2) for index in range(30)),
        "vix_delta_1obs": rows,
        "vix_delta_5obs": tuple(-value for value in rows),
        "usd_broad_delta_1obs": tuple(float((index % 5) * 3) for index in range(30)),
    }

    weights = {
        "vix_log_level": 1.0,
        "us_10y_log_level": 2.0,
        "family_pc_vix_1": 3.0,
        "family_pc_usd_broad_1": 4.0,
    }

    def evaluate(features: tuple[str, ...]) -> FeatureSubsetScore:
        return FeatureSubsetScore(features, sum(weights[name] for name in features))

    result = run_canonical_feature_selection(
        feature_values,
        contract,
        quality_eligible_features=names,
        evaluate_subset=evaluate,
        max_sffs_features=2,
    )

    assert result.quality_eligible_features == names
    assert result.family_reduction.removed_features == ("vix_delta_5obs",)
    assert tuple(item.family for item in result.family_pca) == ("usd_broad", "vix")
    assert result.global_reduction.representatives == (
        "vix_log_level",
        "us_10y_log_level",
        "family_pc_usd_broad_1",
    )
    assert result.selected_features == ("family_pc_usd_broad_1", "us_10y_log_level")
    assert len(result.ablation.one_feature_results) == 2
    assert (
        result.evidence_metadata["feature_selection_profile_hash"] == contract.profile.profile_hash
    )


def test_canonical_pipeline_can_run_without_transformations() -> None:
    names = ("vix_log_level", "us_10y_log_level")
    contract = build_feature_role_contract((TEMPORAL_KEY, *names))
    values = {
        names[0]: tuple(float(index) for index in range(30)),
        names[1]: tuple(float((index % 4) ** 2) for index in range(30)),
    }

    def evaluate(features: tuple[str, ...]) -> FeatureSubsetScore:
        return FeatureSubsetScore(features, float(len(features)))

    result = run_canonical_feature_selection(
        values,
        contract,
        quality_eligible_features=names,
        evaluate_subset=evaluate,
        max_sffs_features=2,
    )

    assert result.family_reduction.retained_features == ()
    assert result.family_pca == ()
    assert result.selected_features == names
