from hashlib import sha256
from pathlib import Path

import duckdb
import pandas as pd

from market_regime_engine.feature_discovery.ablation import HMMSubsetEvaluation
from market_regime_engine.feature_discovery.feature_roles import (
    TEMPORAL_KEY,
    build_feature_role_contract,
)
from market_regime_engine.feature_discovery.metadata_store import FeatureSelectionMetadataStore
from market_regime_engine.feature_discovery.monthly_refit import _materialize_family_pca
from market_regime_engine.feature_discovery.pipeline import run_canonical_feature_selection
from market_regime_engine.feature_discovery.sffs import FeatureSubsetScore

SELECTOR_HASH = "a" * 64


def _frontier_score(features: tuple[str, ...]) -> FeatureSubsetScore:
    return FeatureSubsetScore(features, float(len(features)))


def _frontier_hmm(features: tuple[str, ...]) -> HMMSubsetEvaluation:
    return HMMSubsetEvaluation(
        _frontier_score(features),
        "gaussian_hmm",
        2,
        SELECTOR_HASH,
        sha256("|".join(features).encode()).hexdigest(),
    )


def _frontier_k_score(state_count: int, features: tuple[str, ...]) -> FeatureSubsetScore:
    return FeatureSubsetScore(
        features,
        float(len(features)),
        model_family="gaussian_hmm",
        state_count=state_count,
    )


def test_canonical_pipeline_composes_train_only_stages_in_order(tmp_path: Path) -> None:
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

    def evaluate_by_k(state_count: int, features: tuple[str, ...]) -> FeatureSubsetScore:
        return FeatureSubsetScore(
            features,
            sum(weights[name] for name in features) + state_count * 0.0,
            model_family="gaussian_hmm",
            state_count=state_count,
        )

    fit_count = 0

    def evaluate_hmm(features: tuple[str, ...]) -> HMMSubsetEvaluation:
        nonlocal fit_count
        fit_count += 1
        return HMMSubsetEvaluation(
            FeatureSubsetScore(features, sum(weights[name] for name in features)),
            "gaussian_hmm",
            2,
            SELECTOR_HASH,
            f"{fit_count:064x}",
        )

    result = run_canonical_feature_selection(
        feature_values,
        contract,
        quality_eligible_features=names,
        evaluate_subset=evaluate,
        evaluate_hmm_subset=evaluate_hmm,
        hmm_selector_contract_hash=SELECTOR_HASH,
        max_sffs_features=2,
        metadata_store=FeatureSelectionMetadataStore(tmp_path),
        metadata_fold_id="fold-001",
        metadata_source_build_id="build-001",
        metadata_state_count=2,
        evaluate_gaussian_subset_by_k=evaluate_by_k,
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
    materialized = _materialize_family_pca(pd.DataFrame(feature_values), contract, result)
    assert tuple(materialized["family_pc_vix_1"].shape) == (30,)
    assert materialized["family_pc_vix_1"].notna().all()
    assert tuple(item.state_count for item in result.k_sffs) == (2, 3, 4, 5)
    assert all(
        features == result.selected_features for _, features in result.emission_feature_orders
    )
    assert len(result.ablation.one_feature_results) == 2
    assert (
        result.evidence_metadata["feature_selection_profile_hash"] == contract.profile.profile_hash
    )
    with duckdb.connect(str(tmp_path / "feature_selection.duckdb"), read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM sffs_steps").fetchone() == (20,)


def test_canonical_pipeline_can_run_without_transformations() -> None:
    names = ("vix_log_level", "us_10y_log_level")
    contract = build_feature_role_contract((TEMPORAL_KEY, *names))
    values = {
        names[0]: tuple(float(index) for index in range(30)),
        names[1]: tuple(float((index % 4) ** 2) for index in range(30)),
    }

    def evaluate(features: tuple[str, ...]) -> FeatureSubsetScore:
        return FeatureSubsetScore(features, float(len(features)))

    fit_count = 0

    def evaluate_hmm(features: tuple[str, ...]) -> HMMSubsetEvaluation:
        nonlocal fit_count
        fit_count += 1
        return HMMSubsetEvaluation(
            FeatureSubsetScore(features, float(len(features))),
            "gaussian_hmm",
            2,
            SELECTOR_HASH,
            f"{fit_count:064x}",
        )

    result = run_canonical_feature_selection(
        values,
        contract,
        quality_eligible_features=names,
        evaluate_subset=evaluate,
        evaluate_hmm_subset=evaluate_hmm,
        hmm_selector_contract_hash=SELECTOR_HASH,
        max_sffs_features=2,
    )

    assert result.family_reduction.retained_features == ()
    assert result.family_pca == ()
    assert result.selected_features == names


def test_canonical_pipeline_reuses_frontier_for_selection_and_ablation() -> None:
    names = ("vix_log_level", "us_10y_log_level")
    contract = build_feature_role_contract((TEMPORAL_KEY, *names))
    values = {
        names[0]: tuple(float(index) for index in range(30)),
        names[1]: tuple(float((index % 7) ** 2) for index in range(30)),
    }

    result = run_canonical_feature_selection(
        values,
        contract,
        quality_eligible_features=names,
        evaluate_subset=_frontier_score,
        evaluate_hmm_subset=_frontier_hmm,
        hmm_selector_contract_hash=SELECTOR_HASH,
        max_sffs_features=2,
        max_workers=2,
        evaluate_gaussian_subset_by_k=_frontier_k_score,
    )

    assert result.selected_features == names
    assert tuple(item.state_count for item in result.k_sffs) == (2, 3, 4, 5)
    assert result.ablation.ablation_losses == (1.0, 1.0)


def test_zero_rank_family_is_invalid_without_blocking_core_selection() -> None:
    names = ("vix_log_level", "us_10y_log_level", "vix_delta_1obs")
    contract = build_feature_role_contract((TEMPORAL_KEY, *names))
    values = {
        "vix_log_level": tuple(float(index) for index in range(30)),
        "us_10y_log_level": tuple(float((index % 4) ** 2) for index in range(30)),
        "vix_delta_1obs": (1.0,) * 30,
    }

    def evaluate(features: tuple[str, ...]) -> FeatureSubsetScore:
        return FeatureSubsetScore(features, float(len(features)))

    def evaluate_hmm(features: tuple[str, ...]) -> HMMSubsetEvaluation:
        return HMMSubsetEvaluation(
            FeatureSubsetScore(features, float(len(features))),
            "gaussian_hmm",
            2,
            SELECTOR_HASH,
            sha256("|".join(features).encode()).hexdigest(),
        )

    result = run_canonical_feature_selection(
        values,
        contract,
        quality_eligible_features=names,
        evaluate_subset=evaluate,
        evaluate_hmm_subset=evaluate_hmm,
        hmm_selector_contract_hash=SELECTOR_HASH,
        max_sffs_features=2,
    )

    assert result.invalid_families == ("vix",)
    assert result.family_pca == ()
    assert result.selected_features == ("vix_log_level", "us_10y_log_level")
