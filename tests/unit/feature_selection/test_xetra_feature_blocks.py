from __future__ import annotations

from pathlib import Path

import yaml

from market_regime_engine.feature_selection import FeatureBlock, FeatureSelectionPolicy

CONFIG = Path("configs/feature_selection/xetra_semantic_medoid_v1.yaml")
EXPECTED_BLOCK_IDS = (
    "us_equity_volatility_spot",
    "us_equity_volatility_term_structure",
    "europe_equity_volatility",
    "rates_volatility",
    "systemic_stress",
    "credit_stress",
    "rates_yield_curve",
    "usd_fx",
)
EXPECTED_BLOCK_SIZES = (4, 21, 4, 4, 3, 3, 7, 2)
EXPECTED_FEATURES = (
    "vix_level",
    "vix_delta_5obs",
    "vix_delta_20obs",
    "vix_zscore_60obs",
    "vix9d_level",
    "vix9d_delta_5obs",
    "vix9d_delta_20obs",
    "vix9d_zscore_60obs",
    "vix3m_level",
    "vix3m_delta_5obs",
    "vix3m_delta_20obs",
    "vix3m_zscore_60obs",
    "vix6m_level",
    "vix6m_delta_5obs",
    "vix6m_delta_20obs",
    "vix6m_zscore_60obs",
    "vix1y_level",
    "vix1y_delta_5obs",
    "vix1y_delta_20obs",
    "vix1y_zscore_60obs",
    "vix9d_vix_ratio",
    "vix_vix3m_ratio",
    "vix3m_minus_vix",
    "vix6m_minus_vix",
    "vix1y_minus_vix",
    "vstoxx_level",
    "vstoxx_delta_5obs",
    "vstoxx_delta_20obs",
    "vstoxx_zscore_60obs",
    "move_level",
    "move_delta_5obs",
    "move_delta_20obs",
    "move_zscore_60obs",
    "ciss_level",
    "ciss_delta_5obs",
    "ciss_delta_20obs",
    "euro_hy_oas_level",
    "euro_hy_oas_delta_5obs",
    "euro_hy_oas_delta_20obs",
    "us_2y_level",
    "us_2y_delta_20obs",
    "us_10y_level",
    "us_10y_delta_20obs",
    "estr_level",
    "estr_delta_20obs",
    "us_10y_minus_us_2y",
    "usd_broad_level",
    "usd_broad_delta_20obs",
)


def load_policy() -> tuple[dict[str, object], FeatureSelectionPolicy]:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    raw_blocks = raw["blocks"]
    assert isinstance(raw_blocks, list)
    blocks = tuple(
        FeatureBlock(
            block_id=str(item["block_id"]),
            features=tuple(str(value) for value in item["features"]),
        )
        for item in raw_blocks
    )
    policy = FeatureSelectionPolicy(
        policy_id=str(raw["policy_id"]),
        blocks=blocks,
        within_block_method=str(raw["within_block_method"]),
        cross_block_method=str(raw["cross_block_method"]),
        minimum_feature_coverage=float(raw["minimum_feature_coverage"]),
        minimum_nonzero_variance=float(raw["minimum_nonzero_variance"]),
        minimum_block_complete_observations=int(raw["minimum_block_complete_observations"]),
        maximum_cross_block_abs_spearman=float(raw["maximum_cross_block_abs_spearman"]),
        numeric_tie_abs_tolerance=float(raw["numeric_tie_abs_tolerance"]),
    )
    return raw, policy


def test_xetra_policy_matches_exact_pinned_constants() -> None:
    raw, policy = load_policy()
    assert policy.policy_id == "xetra_semantic_medoid_v1"
    assert policy.within_block_method == "absolute_spearman_medoid"
    assert policy.cross_block_method == "absolute_spearman_prune"
    assert policy.minimum_feature_coverage == 0.90
    assert policy.minimum_nonzero_variance == 1e-12
    assert policy.minimum_block_complete_observations == 504
    assert policy.maximum_cross_block_abs_spearman == 0.85
    assert policy.numeric_tie_abs_tolerance == 1e-12
    assert set(raw) == {
        "policy_id",
        "within_block_method",
        "cross_block_method",
        "minimum_feature_coverage",
        "minimum_nonzero_variance",
        "minimum_block_complete_observations",
        "maximum_cross_block_abs_spearman",
        "numeric_tie_abs_tolerance",
        "blocks",
    }


def test_xetra_blocks_are_exact_exhaustive_and_ordered() -> None:
    _, policy = load_policy()
    assert tuple(block.block_id for block in policy.blocks) == EXPECTED_BLOCK_IDS
    assert tuple(len(block.features) for block in policy.blocks) == EXPECTED_BLOCK_SIZES
    assert policy.feature_universe == EXPECTED_FEATURES
    assert len(policy.feature_universe) == 48
    assert len(set(policy.feature_universe)) == 48


def test_no_temporal_target_or_economic_field_is_present() -> None:
    _, policy = load_policy()
    forbidden_exact = {"timestamp_m1", "state", "label", "target"}
    forbidden_terms = (
        "etf",
        "portfolio",
        "sharpe",
        "sortino",
        "calmar",
        "drawdown",
        "return",
        "trading",
    )
    for feature in policy.feature_universe:
        lowered = feature.lower()
        assert lowered not in forbidden_exact
        assert all(term not in lowered for term in forbidden_terms)
