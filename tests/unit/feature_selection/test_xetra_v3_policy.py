from __future__ import annotations

from pathlib import Path

import yaml

from market_regime_engine.feature_selection import FeatureBlock, FeatureSelectionPolicy

CONFIG = Path("configs/feature_selection/xetra_semantic_medoid_v3.yaml")
V1 = Path("configs/feature_selection/xetra_semantic_medoid_v1.yaml")
V2 = Path("configs/feature_selection/xetra_semantic_medoid_v2.yaml")

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
EXPECTED_BLOCK_SIZES = (5, 25, 5, 5, 4, 4, 10, 3)
EXPECTED_DELTAS = (
    "vix_delta_1obs",
    "vix9d_delta_1obs",
    "vix3m_delta_1obs",
    "vix6m_delta_1obs",
    "vix1y_delta_1obs",
    "vstoxx_delta_1obs",
    "move_delta_1obs",
    "ciss_delta_1obs",
    "euro_hy_oas_delta_1obs",
    "us_2y_delta_1obs",
    "us_10y_delta_1obs",
    "estr_delta_1obs",
    "usd_broad_delta_1obs",
)
EXPECTED_FEATURES = (
    "vix_level",
    "vix_delta_5obs",
    "vix_delta_20obs",
    "vix_zscore_60obs",
    "vix_delta_1obs",
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
    "vix9d_delta_1obs",
    "vix3m_delta_1obs",
    "vix6m_delta_1obs",
    "vix1y_delta_1obs",
    "vstoxx_level",
    "vstoxx_delta_5obs",
    "vstoxx_delta_20obs",
    "vstoxx_zscore_60obs",
    "vstoxx_delta_1obs",
    "move_level",
    "move_delta_5obs",
    "move_delta_20obs",
    "move_zscore_60obs",
    "move_delta_1obs",
    "ciss_level",
    "ciss_delta_5obs",
    "ciss_delta_20obs",
    "ciss_delta_1obs",
    "euro_hy_oas_level",
    "euro_hy_oas_delta_5obs",
    "euro_hy_oas_delta_20obs",
    "euro_hy_oas_delta_1obs",
    "us_2y_level",
    "us_2y_delta_20obs",
    "us_10y_level",
    "us_10y_delta_20obs",
    "estr_level",
    "estr_delta_20obs",
    "us_10y_minus_us_2y",
    "us_2y_delta_1obs",
    "us_10y_delta_1obs",
    "estr_delta_1obs",
    "usd_broad_level",
    "usd_broad_delta_20obs",
    "usd_broad_delta_1obs",
)
EXPECTED_DELTA_BLOCK = {
    "vix_delta_1obs": "us_equity_volatility_spot",
    "vix9d_delta_1obs": "us_equity_volatility_term_structure",
    "vix3m_delta_1obs": "us_equity_volatility_term_structure",
    "vix6m_delta_1obs": "us_equity_volatility_term_structure",
    "vix1y_delta_1obs": "us_equity_volatility_term_structure",
    "vstoxx_delta_1obs": "europe_equity_volatility",
    "move_delta_1obs": "rates_volatility",
    "ciss_delta_1obs": "systemic_stress",
    "euro_hy_oas_delta_1obs": "credit_stress",
    "us_2y_delta_1obs": "rates_yield_curve",
    "us_10y_delta_1obs": "rates_yield_curve",
    "estr_delta_1obs": "rates_yield_curve",
    "usd_broad_delta_1obs": "usd_fx",
}


def _load(path: Path) -> tuple[dict[str, object], FeatureSelectionPolicy]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
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
    return raw, FeatureSelectionPolicy(
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


def test_v3_policy_is_exact_and_exhaustive() -> None:
    raw, policy = _load(CONFIG)
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
    assert policy.policy_id == "xetra_semantic_medoid_v3"
    assert tuple(block.block_id for block in policy.blocks) == EXPECTED_BLOCK_IDS
    assert tuple(len(block.features) for block in policy.blocks) == EXPECTED_BLOCK_SIZES
    assert policy.feature_universe == EXPECTED_FEATURES
    assert len(policy.feature_universe) == 61
    assert len(set(policy.feature_universe)) == 61
    assert policy.within_block_method == "absolute_spearman_medoid"
    assert policy.cross_block_method == "absolute_spearman_prune"
    assert policy.minimum_feature_coverage == 0.90
    assert policy.minimum_nonzero_variance == 1e-12
    assert policy.minimum_block_complete_observations == 504
    assert policy.maximum_cross_block_abs_spearman == 0.85
    assert policy.numeric_tie_abs_tolerance == 1e-12


def test_delta_tuple_has_exact_block_membership() -> None:
    _, policy = _load(CONFIG)
    seen: list[str] = []
    for block in policy.blocks:
        for feature in block.features:
            if feature in EXPECTED_DELTA_BLOCK:
                assert EXPECTED_DELTA_BLOCK[feature] == block.block_id
                seen.append(feature)
    assert tuple(seen) == EXPECTED_DELTAS


def test_v2_feature_order_is_preserved_as_subsequence() -> None:
    _, v2 = _load(V2)
    _, v3 = _load(CONFIG)
    delta_set = set(EXPECTED_DELTAS)
    assert tuple(feature for feature in v3.feature_universe if feature not in delta_set) == (
        v2.feature_universe
    )


def test_historical_policy_files_remain_distinct_and_unchanged_in_identity() -> None:
    _, v1 = _load(V1)
    _, v2 = _load(V2)
    assert v1.policy_id == "xetra_semantic_medoid_v1"
    assert v2.policy_id == "xetra_semantic_medoid_v2"
    assert len(v1.feature_universe) == 48
    assert len(v2.feature_universe) == 48


def test_no_temporal_target_or_economic_field_is_present() -> None:
    _, policy = _load(CONFIG)
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
