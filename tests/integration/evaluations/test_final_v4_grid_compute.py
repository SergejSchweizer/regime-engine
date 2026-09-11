from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import market_regime_engine.evaluation.walk_forward as walk_forward
from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.evaluations.final_v4_grid import evaluate_final_v4_grid
from market_regime_engine.feature_discovery.contracts import FINAL_CANDIDATE_IDS
from market_regime_engine.models.protocols import FitResult
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.training.multistart import (
    MULTISTART_SEEDS,
    MultistartResult,
    StartDiagnostic,
)

pytestmark = pytest.mark.integration


def source_rows() -> pd.DataFrame:
    row_count = 1323  # exactly one complete outer 1260/63/63 fold
    start = datetime(2020, 1, 1, tzinfo=UTC)
    index = np.arange(row_count, dtype=np.float64)
    regime_id = np.random.default_rng(2028).integers(0, 3, size=row_count)
    first_axis = np.asarray((-1.5, 0.0, 1.5))[regime_id]
    second_axis = np.asarray((1.0, -1.0, 0.5))[regime_id]
    noise = np.random.default_rng(2029)
    return pd.DataFrame(
        {
            "timestamp_m1": tuple(start + timedelta(days=int(value)) for value in index),
            "f0": first_axis + 0.10 * noise.normal(size=row_count),
            "f1": second_axis + 0.10 * noise.normal(size=row_count),
        }
    )


def _one_real_fit_multistart(train_rows, *, state_count, adapter_factory, **_kwargs):
    """Retain real backend fitting while avoiding 12 x 8 redundant CI fits."""

    result: FitResult = adapter_factory().fit(train_rows, state_count, MULTISTART_SEEDS[0])
    diagnostics = tuple(
        StartDiagnostic(
            seed=seed,
            success=True,
            converged=True,
            iterations=result.iterations,
            train_log_likelihood=result.train_log_likelihood,
            artifact=result.artifact,
            failure_reason=None,
        )
        for seed in MULTISTART_SEEDS
    )
    return MultistartResult(
        state_count=state_count,
        winner=result,
        diagnostics=diagnostics,
    )


def test_exact_12_candidate_grid_runs_all_model_families_with_real_fits(monkeypatch) -> None:
    monkeypatch.setattr(walk_forward, "run_multistart", _one_real_fit_multistart)
    rows = source_rows()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)

    result = evaluate_final_v4_grid(
        rows,
        feature_order=("f0", "f1"),
        profile=profile,
        plan=plan,
        source_build_id="synthetic-build",
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
    )

    assert tuple(item.candidate_id for item in result.grid.evaluations) == FINAL_CANDIDATE_IDS
    assert tuple(item.candidate_id for item in result.grid.aggregates) == FINAL_CANDIDATE_IDS
    assert result.selection is not None or result.no_champion_reason
