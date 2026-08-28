from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.evaluations.clocks import build_evaluation_clock
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
