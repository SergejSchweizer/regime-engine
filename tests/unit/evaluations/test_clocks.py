from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan, plan_walk_forward
from market_regime_engine.evaluations.clocks import (
    ClockFoldEvidence,
    EvaluationClock,
    build_evaluation_clock,
)
from market_regime_engine.evaluations.contracts import (
    DELTA1_FEATURES,
    EvaluationId,
    FeatureSpec,
)
from market_regime_engine.profiles.loader import load_profile

MEDOIDS = tuple(f"medoid_{index}" for index in range(8))


def source_rows() -> pd.DataFrame:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    rows = 1323
    result: dict[str, object] = {
        "timestamp_m1": tuple(start + timedelta(days=index) for index in range(rows))
    }
    for column in (*MEDOIDS, *DELTA1_FEATURES):
        result[column] = np.ones(rows)
    return pd.DataFrame(result)


def test_univariate_clocks_are_scoped_to_their_own_feature_tuples() -> None:
    rows = source_rows()
    rows.loc[4, MEDOIDS[0]] = np.nan
    rows.loc[1264, DELTA1_FEATURES[0]] = np.nan
    profile = load_profile("configs/profiles/xetra_v1.yaml")
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)

    medoid = build_evaluation_clock(
        rows, plan, FeatureSpec(EvaluationId.MEDOID_UNIVARIATE, MEDOIDS)
    )
    delta = build_evaluation_clock(
        rows, plan, FeatureSpec(EvaluationId.DELTA1_UNIVARIATE, DELTA1_FEATURES)
    )

    assert medoid.clock_hash != delta.clock_hash
    medoid_fold, delta_fold = medoid.fold_evidence[0], delta.fold_evidence[0]
    assert rows.loc[4, "timestamp_m1"] in medoid_fold.skipped_train_timestamps
    assert rows.loc[4, "timestamp_m1"] in delta_fold.retained_train_timestamps
    assert rows.loc[1264, "timestamp_m1"] in medoid_fold.retained_test_timestamps
    assert rows.loc[1264, "timestamp_m1"] in delta_fold.skipped_test_timestamps
    assert len(medoid_fold.retained_train_timestamps) == 1259
    assert len(delta_fold.retained_test_timestamps) == 62


def test_clock_fails_closed_for_invalid_source_plan_and_clock_evidence() -> None:
    rows = source_rows()
    profile = load_profile("configs/profiles/xetra_v1.yaml")
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    spec = FeatureSpec(EvaluationId.MEDOID_UNIVARIATE, MEDOIDS)
    timestamp = rows.loc[0, "timestamp_m1"]

    with pytest.raises(ValueError, match="missing required columns"):
        build_evaluation_clock(rows.drop(columns=[MEDOIDS[0]]), plan, spec)
    repeated = rows.copy()
    repeated.loc[1, "timestamp_m1"] = repeated.loc[0, "timestamp_m1"]
    with pytest.raises(ValueError, match="strictly increasing"):
        build_evaluation_clock(repeated, plan, spec)
    with pytest.raises(ValueError, match="at least one planned fold"):
        build_evaluation_clock(rows, WalkForwardPlan((), None, "a" * 64), spec)
    with pytest.raises(ValueError, match="source-row counts"):
        build_evaluation_clock(
            rows,
            replace(plan, folds=(replace(plan.folds[0], train_source_observations=1261),)),
            spec,
        )
    with pytest.raises(ValueError, match="source timestamps do not match"):
        build_evaluation_clock(
            rows,
            replace(
                plan,
                folds=(
                    replace(plan.folds[0], test_end=plan.folds[0].test_end + timedelta(days=1)),
                ),
            ),
            spec,
        )
    with pytest.raises(ValueError, match="non-empty"):
        ClockFoldEvidence("", (), (), (), ())
    with pytest.raises(ValueError, match="strictly increasing"):
        ClockFoldEvidence("fold_001", (timestamp + timedelta(days=1), timestamp), (), (), ())
    evidence = ClockFoldEvidence("fold_001", (), (), (), ())
    with pytest.raises(ValueError, match="duplicate-free"):
        EvaluationClock(EvaluationId.MEDOID_UNIVARIATE, ("f", "f"), "a" * 64, (evidence,))
    with pytest.raises(ValueError, match="SHA-256"):
        EvaluationClock(EvaluationId.MEDOID_UNIVARIATE, ("f",), "a", (evidence,))
    with pytest.raises(ValueError, match="ordered"):
        EvaluationClock(
            EvaluationId.MEDOID_UNIVARIATE,
            ("f",),
            "a" * 64,
            (ClockFoldEvidence("fold_002", (), (), (), ()), evidence),
        )
