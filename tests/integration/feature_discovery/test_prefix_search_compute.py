from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import market_regime_engine.evaluation.walk_forward as walk_forward
from market_regime_engine.evaluations.provisional_teacher import build_inner_walk_forward_plan
from market_regime_engine.feature_discovery.contracts import ProvisionalTeacherReference
from market_regime_engine.feature_discovery.prefix_search import search_ranked_prefixes
from market_regime_engine.models.protocols import FitResult
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.training.multistart import (
    MULTISTART_SEEDS,
    MultistartResult,
    StartDiagnostic,
)

pytestmark = pytest.mark.integration

FEATURES = ("f0", "f1", "f2", "f3")


def source_rows() -> pd.DataFrame:
    row_count = 819  # exactly one complete inner 756/63/63 fold
    start = datetime(2020, 1, 1, tzinfo=UTC)
    index = np.arange(row_count, dtype=np.float64)
    regime_id = np.random.default_rng(2026).integers(0, 5, size=row_count)
    first_axis = np.asarray((-2.0, -1.0, 0.0, 1.0, 2.0))[regime_id]
    second_axis = np.asarray((2.0, -1.0, 1.0, -2.0, 0.0))[regime_id]
    noise = np.random.default_rng(2027)
    return pd.DataFrame(
        {
            "timestamp_m1": tuple(start + timedelta(days=int(value)) for value in index),
            "f0": first_axis + 0.10 * noise.normal(size=row_count),
            "f1": second_axis + 0.10 * noise.normal(size=row_count),
            "f2": np.sin(index / 13.0) + 0.05 * noise.normal(size=row_count),
            "f3": np.cos(index / 19.0) + 0.05 * noise.normal(size=row_count),
        }
    )


def _one_real_fit_multistart(train_rows, *, state_count, adapter_factory, **_kwargs):
    """Keep this integration test bounded while retaining a real HMM fit per candidate."""

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


def test_all_prefixes_and_gaussian_states_execute_with_real_hmm_math(monkeypatch) -> None:
    monkeypatch.setattr(walk_forward, "run_multistart", _one_real_fit_multistart)
    rows = source_rows()
    plan = build_inner_walk_forward_plan(tuple(rows["timestamp_m1"]))
    timestamps = tuple(rows["timestamp_m1"].iloc[756:819])
    probabilities = tuple(
        ((1.0, 0.0) if index % 2 == 0 else (0.0, 1.0)) for index in range(len(timestamps))
    )
    result = search_ranked_prefixes(
        rows,
        ranked_features=FEATURES,
        teacher=ProvisionalTeacherReference(
            candidate_id="gaussian_hmm_k2_full",
            state_count=2,
            timestamps=timestamps,
            filtered_probabilities=probabilities,
            dominant_states=tuple(index % 2 for index in range(len(timestamps))),
            valid_inner_fold_ids=("fold_001",),
            source_build_id="synthetic-build",
            inner_plan_hash=plan.plan_hash,
        ),
        profile=load_profile("configs/profiles/xetra_v4.yaml"),
        source_build_id="synthetic-build",
    )

    assert tuple(item.prefix_length for item in result.evaluations) == (2, 3, 4)
    assert all(len(item.candidate_evaluations) == 4 for item in result.evaluations if item.valid)
    assert any(item.valid for item in result.evaluations)
