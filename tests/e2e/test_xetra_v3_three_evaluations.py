from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.evaluations.clocks import build_evaluation_clock
from market_regime_engine.evaluations.contracts import (
    DELTA1_FEATURES,
    EvaluationId,
    FeatureSpec,
    candidate_specs,
)
from market_regime_engine.feature_selection.contracts import FeatureBlock, FeatureSelectionPolicy
from market_regime_engine.feature_selection.freeze import freeze_first_train_features
from market_regime_engine.profiles.loader import load_profile

pytestmark = pytest.mark.integration


def _policy() -> FeatureSelectionPolicy:
    raw = yaml.safe_load(
        Path("configs/feature_selection/xetra_semantic_medoid_v3.yaml").read_text()
    )
    assert isinstance(raw, dict)
    return FeatureSelectionPolicy(
        str(raw["policy_id"]),
        tuple(
            FeatureBlock(str(item["block_id"]), tuple(item["features"])) for item in raw["blocks"]
        ),
        str(raw["within_block_method"]),
        str(raw["cross_block_method"]),
        float(raw["minimum_feature_coverage"]),
        float(raw["minimum_nonzero_variance"]),
        int(raw["minimum_block_complete_observations"]),
        float(raw["maximum_cross_block_abs_spearman"]),
        float(raw["numeric_tie_abs_tolerance"]),
    )


def test_three_evaluation_contract_uses_independent_canonical_clocks() -> None:
    policy = _policy()
    profile = load_profile("configs/profiles/xetra_v3.yaml")
    timestamps = tuple(
        datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=index) for index in range(1386)
    )
    rng = np.random.default_rng(20260901)
    common_signal = rng.normal(size=len(timestamps))
    rows = pd.DataFrame(
        {"timestamp_m1": timestamps}
        | {
            feature: (
                common_signal + rng.normal(scale=0.01, size=len(timestamps))
                if feature.endswith("_delta_1obs")
                else rng.normal(size=len(timestamps))
            )
            for feature in policy.feature_universe
        }
    )
    expected_medoids = tuple(
        next(feature for feature in block.features if feature.endswith("_delta_1obs"))
        for block in policy.blocks
    )
    rows.loc[1260, expected_medoids[0]] = np.nan
    rows.loc[1261, DELTA1_FEATURES[0]] = np.nan
    plan = plan_walk_forward(timestamps, profile.walk_forward)
    selection = freeze_first_train_features(
        rows.iloc[: plan.folds[0].train_source_observations],
        policy,
        source_build_id="hermetic-v3-build",
        data_sha256="a" * 64,
        evaluation_plan_hash=plan.plan_hash,
    )
    medoids = selection.evidence.preliminary_medoids
    medoid_clock = build_evaluation_clock(
        rows, plan, FeatureSpec(EvaluationId.MEDOID_UNIVARIATE, medoids)
    )
    delta_clock = build_evaluation_clock(
        rows, plan, FeatureSpec(EvaluationId.DELTA1_UNIVARIATE, DELTA1_FEATURES)
    )
    assert len(policy.feature_universe) == 61
    assert medoids[0] == expected_medoids[0]
    assert selection.final_features == selection.evidence.final_features
    assert 1 <= len(selection.final_features) <= 8
    assert any(feature in DELTA1_FEATURES for feature in medoids)
    assert medoid_clock.clock_hash != delta_clock.clock_hash
    assert medoid_clock.evaluation_id is EvaluationId.MEDOID_UNIVARIATE
    assert delta_clock.evaluation_id is EvaluationId.DELTA1_UNIVARIATE
    assert len(candidate_specs(medoids)) == 12
    assert 8 * len(candidate_specs((medoids[0],))) == 96
    assert len(DELTA1_FEATURES) * len(candidate_specs((DELTA1_FEATURES[0],))) == 156
    assert 12 + 96 + 156 == 264
