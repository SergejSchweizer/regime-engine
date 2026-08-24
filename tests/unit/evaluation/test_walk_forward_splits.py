from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.profiles.config import WalkForwardConfig

START = datetime(2020, 1, 1, tzinfo=UTC)


def config() -> WalkForwardConfig:
    return WalkForwardConfig(
        minimum_train_source_observations=1260,
        test_source_observations=63,
        step_source_observations=63,
        allow_partial_final_test=False,
        minimum_model_train_observations=504,
        minimum_model_test_observations=42,
        ranking_abs_tolerance=1e-12,
    )


def timestamps(count: int) -> tuple[datetime, ...]:
    return tuple(START + timedelta(days=index) for index in range(count))


def test_exact_expanding_plan_and_cutoff() -> None:
    plan = plan_walk_forward(timestamps(1386), config())
    assert [fold.fold_id for fold in plan.folds] == ["fold_001", "fold_002"]
    assert plan.folds[0].train_source_observations == 1260
    assert plan.folds[0].test_source_observations == 63
    assert plan.folds[1].train_source_observations == 1323
    assert plan.folds[0].train_end < plan.folds[0].test_start
    assert plan.evaluation_cutoff == timestamps(1386)[-1]
    assert len(plan.plan_hash) == 64
    assert plan.plan_hash == plan_walk_forward(timestamps(1386), config()).plan_hash


def test_partial_final_test_is_ignored_without_synthetic_rows() -> None:
    source = timestamps(1400)
    plan = plan_walk_forward(source, config())
    assert len(plan.folds) == 2
    assert plan.evaluation_cutoff == source[1385]
    assert source[-1] > plan.evaluation_cutoff


def test_no_complete_fold_returns_no_cutoff() -> None:
    plan = plan_walk_forward(timestamps(1322), config())
    assert plan.folds == ()
    assert plan.evaluation_cutoff is None


def test_wrong_plan_or_bad_timestamp_sequence_fails_closed() -> None:
    with pytest.raises(ValueError, match="pinned"):
        plan_walk_forward(timestamps(1323), replace(config(), test_source_observations=64))
    bad = list(timestamps(1323))
    bad[100] = bad[99]
    with pytest.raises(ValueError, match="strictly increasing"):
        plan_walk_forward(bad, config())
    naive = list(timestamps(1323))
    naive[0] = naive[0].replace(tzinfo=None)
    with pytest.raises(ValueError, match="UTC"):
        plan_walk_forward(naive, config())
