from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.evaluation.model_clock import (
    build_model_clock_preflight,
    require_model_clock_eligible,
)
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan, plan_walk_forward
from market_regime_engine.feature_discovery.contracts import DiscoveryStatus
from market_regime_engine.profiles.loader import load_profile

FEATURES = ("f0", "f1", "f2")


def _rows(count: int = 1386) -> pd.DataFrame:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    result: dict[str, object] = {
        "timestamp_m1": tuple(start + timedelta(days=index) for index in range(count))
    }
    for feature_index, feature in enumerate(FEATURES):
        result[feature] = np.arange(count, dtype=np.float64) + feature_index
    return pd.DataFrame(result)


def _plan(rows: pd.DataFrame):
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    return plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)


def test_common_complete_case_counts_variances_and_hash_are_deterministic() -> None:
    rows = _rows()
    plan = _plan(rows)
    rows.loc[10, "f0"] = np.nan
    rows.loc[20, "f1"] = np.nan
    rows.loc[1330, "f2"] = np.nan

    result = build_model_clock_preflight(rows, FEATURES, plan)

    assert result.status is DiscoveryStatus.VALID
    assert result.first_train_complete_observations == 1258
    assert tuple(fold.fold_id for fold in result.folds) == ("fold_001", "fold_002")
    assert result.folds[0].train_complete_observations == 1258
    assert result.folds[0].test_complete_observations == 63
    assert result.folds[1].train_complete_observations == 1321
    assert result.folds[1].test_complete_observations == 62
    assert all(variance > 1.0e-12 for _, variance in result.first_train_feature_variances)
    assert result.preflight_hash == build_model_clock_preflight(rows, FEATURES, plan).preflight_hash
    require_model_clock_eligible(result, "prototype")


def test_individual_feature_coverage_does_not_replace_common_clock() -> None:
    rows = _rows()
    plan = _plan(rows)
    rows.loc[0:800, "f0"] = np.nan
    rows.loc[801:900, "f1"] = np.nan
    rows.loc[901:1000, "f2"] = np.nan

    result = build_model_clock_preflight(rows, FEATURES, plan)

    assert result.status is DiscoveryStatus.INVALID
    assert result.first_train_complete_observations < 504
    assert result.folds[0].invalid_reason is not None
    assert "first TRAIN complete observations" in (result.invalid_reason or "")


def test_first_train_variance_and_fold_rate_are_hard_gates() -> None:
    rows = _rows()
    rows["f1"] = 1.0
    plan = _plan(rows)
    rows.loc[1260:, FEATURES[0]] = np.nan

    result = build_model_clock_preflight(rows, FEATURES, plan, minimum_valid_fold_rate=1.0)

    assert result.status is DiscoveryStatus.INVALID
    assert "variance" in (result.invalid_reason or "")
    assert "valid-fold rate" in (result.invalid_reason or "")
    with pytest.raises(ValueError, match="outer selection"):
        require_model_clock_eligible(result, "prototype")
    with pytest.raises(ValueError, match="this prefix"):
        require_model_clock_eligible(result, "prefix")


def test_thresholds_can_be_supplied_without_changing_the_common_feature_set() -> None:
    rows = _rows()
    plan = _plan(rows)

    result = build_model_clock_preflight(
        rows,
        FEATURES,
        plan,
        minimum_model_train_observations=1200,
        minimum_model_test_observations=60,
        minimum_valid_fold_rate=1.0,
    )

    assert result.status is DiscoveryStatus.VALID
    assert result.feature_order == FEATURES


def test_source_and_plan_fail_closed_without_any_model_work() -> None:
    rows = _rows()
    plan = _plan(rows)
    with pytest.raises(TypeError, match="pandas DataFrame"):
        build_model_clock_preflight(object(), FEATURES, plan)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="WalkForwardPlan"):
        build_model_clock_preflight(rows, FEATURES, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing required"):
        build_model_clock_preflight(rows.drop(columns=["f1"]), FEATURES, plan)
    repeated = rows.copy()
    repeated.loc[1, "timestamp_m1"] = repeated.loc[0, "timestamp_m1"]
    with pytest.raises(ValueError, match="strictly increasing"):
        build_model_clock_preflight(repeated, FEATURES, plan)
    with pytest.raises(ValueError, match="at least one planned"):
        build_model_clock_preflight(rows, FEATURES, WalkForwardPlan((), None, "a" * 64))
    with pytest.raises(ValueError, match="counts do not match"):
        bad_fold = plan.folds[0]
        bad_plan = WalkForwardPlan(
            folds=(
                type(bad_fold)(
                    fold_index=bad_fold.fold_index,
                    fold_id=bad_fold.fold_id,
                    train_start=bad_fold.train_start,
                    train_end=bad_fold.train_end,
                    test_start=bad_fold.test_start,
                    test_end=bad_fold.test_end,
                    train_source_observations=bad_fold.train_source_observations + 1,
                    test_source_observations=bad_fold.test_source_observations,
                ),
            ),
            evaluation_cutoff=plan.evaluation_cutoff,
            plan_hash=plan.plan_hash,
        )
        build_model_clock_preflight(rows, FEATURES, bad_plan)
    with pytest.raises(ValueError, match="timestamps do not match"):
        bad_plan = replace(
            plan,
            folds=(
                replace(plan.folds[0], test_start=plan.folds[0].test_start - timedelta(hours=1)),
            ),
        )
        build_model_clock_preflight(rows, FEATURES, bad_plan)
    with pytest.raises(ValueError, match="numeric"):
        bad_values = rows.copy()
        bad_values.loc[0, "f0"] = "not-a-number"
        build_model_clock_preflight(bad_values, FEATURES, plan)
    with pytest.raises(ValueError, match="thresholds"):
        build_model_clock_preflight(rows, FEATURES, plan, minimum_model_test_observations=0)


def test_input_contract_and_empty_clock_cases_fail_closed() -> None:
    rows = _rows()
    plan = _plan(rows)
    with pytest.raises(ValueError, match="feature_order"):
        build_model_clock_preflight(rows, (*FEATURES, "f0"), plan)
    with pytest.raises(ValueError, match="minimum_valid_fold_rate"):
        build_model_clock_preflight(rows, FEATURES, plan, minimum_valid_fold_rate=2.0)
    naive = rows.copy()
    naive.loc[0, "timestamp_m1"] = datetime(2020, 1, 1)
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        build_model_clock_preflight(naive, FEATURES, plan)

    empty = rows.copy()
    empty.loc[:, list(FEATURES)] = np.nan
    result = build_model_clock_preflight(empty, FEATURES, plan)
    assert result.first_train_complete_observations == 0
    assert all(variance == 0.0 for _, variance in result.first_train_feature_variances)
    with pytest.raises(ValueError, match="selection_scope"):
        require_model_clock_eligible(result, "invalid")  # type: ignore[arg-type]
