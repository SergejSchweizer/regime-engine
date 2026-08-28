from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import market_regime_engine.evaluations.univariate_grid as grid_module
from market_regime_engine.evaluation.walk_forward import (
    WalkForwardEvaluation,
    WalkForwardFoldResult,
)
from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.evaluations.clocks import build_evaluation_clock
from market_regime_engine.evaluations.contracts import EvaluationId, EvaluationLineage, FeatureSpec
from market_regime_engine.evaluations.univariate_grid import evaluate_univariate_feature_grid
from market_regime_engine.profiles.loader import load_profile

HASH = "a" * 64
MEDOIDS = tuple(f"medoid_{index}" for index in range(8))


def source_rows() -> pd.DataFrame:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    result: dict[str, object] = {
        "timestamp_m1": tuple(start + timedelta(days=index) for index in range(1323))
    }
    for feature in MEDOIDS:
        result[feature] = np.ones(1323)
    return pd.DataFrame(result)


def test_univariate_grid_constructs_twelve_candidates_on_shared_clock() -> None:
    rows = source_rows()
    rows.loc[0, MEDOIDS[0]] = np.nan
    profile = load_profile("configs/profiles/xetra_v3.yaml")
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    spec = FeatureSpec(EvaluationId.MEDOID_UNIVARIATE, MEDOIDS)
    clock = build_evaluation_clock(rows, plan, spec)
    lineage = EvaluationLineage(
        EvaluationId.MEDOID_UNIVARIATE, "build", plan.plan_hash, HASH, HASH, clock.clock_hash
    )
    captured: list[object] = []

    def runner(frame, plan, profile, candidate, adapter):
        del adapter
        captured.append(candidate)
        assert pd.isna(frame.loc[0, MEDOIDS[1]])
        return WalkForwardEvaluation(
            profile_id=profile.profile_id,
            profile_config_version=profile.profile_config_version,
            candidate_id=candidate.candidate_id,
            state_count=candidate.state_count,
            source_build_id=candidate.source_build_id,
            feature_order=candidate.feature_order,
            feature_selection_definition_hash=candidate.feature_selection_definition_hash,
            feature_selection_execution_hash=candidate.feature_selection_execution_hash,
            evaluation_plan_hash=plan.plan_hash,
            evaluation_cutoff=plan.evaluation_cutoff,
            folds=(
                WalkForwardFoldResult("fold_001", 1, False, "deliberate", 1260, 63, 0, 0, 1260, 63),
            ),
        )

    result = evaluate_univariate_feature_grid(
        rows,
        plan=plan,
        profile=profile,
        feature_spec=spec,
        feature_name=MEDOIDS[1],
        clock=clock,
        lineage=lineage,
        runner=runner,
    )
    assert len(captured) == 12
    assert all(
        candidate.feature_order == (MEDOIDS[1],) and candidate.feature_dimension == 1
        for candidate in captured
    )
    assert result.diagnostic_feature_model_winner is None
    assert result.no_winner_reason == "no candidate passes statistical hard gates"


def test_univariate_grid_rejects_multivariate_spec() -> None:
    profile = load_profile("configs/profiles/xetra_v3.yaml")
    rows = source_rows()
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    spec = FeatureSpec(EvaluationId.MEDOID_MULTIVARIATE, (MEDOIDS[0],))
    clock = build_evaluation_clock(rows, plan, spec)
    lineage = EvaluationLineage(
        EvaluationId.MEDOID_MULTIVARIATE, "build", plan.plan_hash, HASH, HASH, clock.clock_hash
    )
    with pytest.raises(ValueError, match="rejects multivariate"):
        evaluate_univariate_feature_grid(
            rows,
            plan=plan,
            profile=profile,
            feature_spec=spec,
            feature_name=MEDOIDS[0],
            clock=clock,
            lineage=lineage,
            runner=lambda *_: None,
        )


def test_univariate_grid_rejects_incompatible_inputs(monkeypatch) -> None:
    rows = source_rows()
    profile = load_profile("configs/profiles/xetra_v3.yaml")
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    spec = FeatureSpec(EvaluationId.MEDOID_UNIVARIATE, MEDOIDS)
    clock = build_evaluation_clock(rows, plan, spec)
    lineage = EvaluationLineage(
        EvaluationId.MEDOID_UNIVARIATE, "build", plan.plan_hash, HASH, HASH, clock.clock_hash
    )
    arguments = dict(
        plan=plan,
        profile=profile,
        feature_spec=spec,
        feature_name=MEDOIDS[0],
        clock=clock,
        lineage=lineage,
        runner=lambda *_: None,
    )
    with pytest.raises(ValueError, match="canonical Xetra v3"):
        evaluate_univariate_feature_grid(
            rows, **(arguments | {"profile": load_profile("configs/profiles/xetra_v2.yaml")})
        )
    with pytest.raises(ValueError, match="feature_name"):
        evaluate_univariate_feature_grid(rows, **(arguments | {"feature_name": "absent"}))
    with pytest.raises(ValueError, match="clock"):
        evaluate_univariate_feature_grid(
            rows,
            **(
                arguments
                | {
                    "clock": build_evaluation_clock(
                        rows, plan, FeatureSpec(EvaluationId.MEDOID_MULTIVARIATE, (MEDOIDS[0],))
                    )
                }
            ),
        )
    with pytest.raises(ValueError, match="lineage"):
        evaluate_univariate_feature_grid(
            rows,
            **(
                arguments
                | {
                    "lineage": EvaluationLineage(
                        EvaluationId.DELTA1_UNIVARIATE,
                        "build",
                        plan.plan_hash,
                        HASH,
                        HASH,
                        clock.clock_hash,
                    )
                }
            ),
        )
    with pytest.raises(ValueError, match="timestamp and selected"):
        grid_module._clocked_source_rows(rows.drop(columns=[MEDOIDS[0]]), clock, MEDOIDS[0])

    monkeypatch.setattr(
        grid_module,
        "select_statistical_champion",
        lambda _: SimpleNamespace(champion_candidate_id="gaussian_hmm_k2_full"),
    )

    def invalid_runner(frame, plan, profile, candidate, adapter):
        del frame, adapter
        return WalkForwardEvaluation(
            profile.profile_id,
            profile.profile_config_version,
            candidate.candidate_id,
            candidate.state_count,
            candidate.source_build_id,
            candidate.feature_order,
            candidate.feature_selection_definition_hash,
            candidate.feature_selection_execution_hash,
            plan.plan_hash,
            plan.evaluation_cutoff,
            (WalkForwardFoldResult("fold_001", 1, False, "deliberate", 1260, 63, 0, 0, 1260, 63),),
        )

    result = evaluate_univariate_feature_grid(rows, **(arguments | {"runner": invalid_runner}))
    assert result.diagnostic_feature_model_winner == "gaussian_hmm_k2_full"
