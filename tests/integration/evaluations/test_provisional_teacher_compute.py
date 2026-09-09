from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from market_regime_engine.evaluations.provisional_teacher import select_provisional_teacher
from market_regime_engine.feature_discovery.contracts import V4_PROVISIONAL_STATE_COUNTS
from market_regime_engine.profiles.loader import load_profile

HASH = "a" * 64
FEATURES = ("prototype_0", "prototype_1")


def source_rows() -> pd.DataFrame:
    row_count = 819  # exactly one complete inner 756/63/63 fold
    start = datetime(2020, 1, 1, tzinfo=UTC)
    index = np.arange(row_count, dtype=np.float64)
    regime_id = np.random.default_rng(2026).integers(0, 5, size=row_count)
    first_regime_axis = np.asarray((-2.0, -1.0, 0.0, 1.0, 2.0))[regime_id]
    second_regime_axis = np.asarray((2.0, -1.0, 1.0, -2.0, 0.0))[regime_id]
    noise = np.random.default_rng(2027)
    return pd.DataFrame(
        {
            "timestamp_m1": tuple(start + timedelta(days=int(value)) for value in index),
            FEATURES[0]: first_regime_axis + 0.10 * noise.normal(size=row_count),
            FEATURES[1]: second_regime_axis + 0.10 * noise.normal(size=row_count),
        }
    )


def test_real_four_k_gaussian_teacher_compute_and_selection() -> None:
    result = select_provisional_teacher(
        source_rows(),
        profile=load_profile("configs/profiles/xetra_v4.yaml"),
        prototype_features=FEATURES,
        source_build_id="synthetic-build",
        feature_selection_definition_hash=HASH,
        feature_selection_execution_hash=HASH,
    )

    assert len(result.inner_plan.folds) == 1
    assert (
        result.inner_plan.folds[0].train_source_observations,
        result.inner_plan.folds[0].test_source_observations,
    ) == (756, 63)
    assert result.model_clock.structural_valid_fold_rate == 1.0
    assert tuple(evaluation.candidate_id for evaluation in result.candidate_evaluations) == tuple(
        f"gaussian_hmm_k{k}_full" for k in V4_PROVISIONAL_STATE_COUNTS
    )
    assert all(len(evaluation.folds) == 1 for evaluation in result.candidate_evaluations)
    assert any(len(evaluation.valid_folds) == 1 for evaluation in result.candidate_evaluations)
    assert result.selection is not None
    assert result.provisional_state_count in V4_PROVISIONAL_STATE_COUNTS
