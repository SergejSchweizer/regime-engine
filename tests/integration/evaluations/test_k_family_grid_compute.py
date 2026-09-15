from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import market_regime_engine.evaluation.walk_forward as walk_forward
from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.evaluations.k_family_grid import (
    K_FAMILY_ORDER,
    evaluate_k_family_grid,
)
from market_regime_engine.models.protocols import FitResult
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.training.multistart import (
    MULTISTART_SEEDS,
    MultistartResult,
    StartDiagnostic,
)

pytestmark = pytest.mark.integration


def _source_rows() -> pd.DataFrame:
    row_count = 1323  # one complete 1260/63 outer fold
    start = datetime(2020, 1, 1, tzinfo=UTC)
    index = np.arange(row_count, dtype=np.float64)
    regime = np.random.default_rng(2031).integers(0, 3, size=row_count)
    first_axis = np.asarray((-1.5, 0.0, 1.5))[regime]
    second_axis = np.asarray((1.0, -1.0, 0.5))[regime]
    noise = np.random.default_rng(2032)
    return pd.DataFrame(
        {
            "timestamp_m1": tuple(start + timedelta(days=int(value)) for value in index),
            "f0": first_axis + 0.10 * noise.normal(size=row_count),
            "f1": second_axis + 0.10 * noise.normal(size=row_count),
            **{
                f"pca_pc_{component:03d}": np.sin(index / (17.0 + component))
                for component in range(1, 9)
            },
        }
    )


def _one_real_fit_multistart(train_rows, *, state_count, adapter_factory, **_kwargs):
    """Use one real fit while retaining the production multistart contract."""

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


@pytest.mark.parametrize("state_count", (2, 3, 4, 5))
def test_k_family_grid_runs_real_gaussian_gmm_and_student_t_for_each_k(
    monkeypatch: pytest.MonkeyPatch, state_count: int
) -> None:
    monkeypatch.setattr(walk_forward, "run_multistart", _one_real_fit_multistart)
    rows = _source_rows()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    universe = tuple(column for column in rows.columns if column != "timestamp_m1")

    result = evaluate_k_family_grid(
        rows,
        state_count=state_count,
        feature_order=("f0", "f1"),
        original_feature_universe=universe,
        profile=profile,
        plan=plan,
        source_build_id="synthetic-k-family-build",
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        max_workers=1,
        pca_raw_feature_order=("f0", "f1"),
    )

    expected_ids = tuple(
        (
            f"gmm_hmm_k{state_count}_m2_full"
            if family == "gmm_hmm"
            else f"{family}_k{state_count}_full"
        )
        for family in K_FAMILY_ORDER
    )
    assert tuple(item.candidate_id for item in result.evaluations) == expected_ids
    assert tuple(item.candidate_id for item in result.aggregates) == expected_ids
    assert all(item.state_count == state_count for item in result.evaluations)
    assert result.selection is not None or result.no_selection_reason
