from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.evaluation.walk_forward import run_walk_forward_candidate
from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import FilterResult, FitResult
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile

PROFILE_CONFIG = Path("configs/profiles/xetra_v1.yaml")


def candidate() -> ResolvedCandidateProfile:
    return ResolvedCandidateProfile(
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        covariance_type="full",
        feature_order=("f0", "f1"),
        feature_dimension=2,
        source_build_id="build-1",
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        original_feature_universe=tuple(f"f{index}" for index in range(48)),
        preliminary_medoids=tuple(f"f{index}" for index in range(8)),
    )


def model_artifact() -> GaussianHMMArtifact:
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=("f0", "f1"),
        start_probabilities=(0.5, 0.5),
        transition_matrix=((0.8, 0.2), (0.2, 0.8)),
        means=((-1.0, -1.0), (1.0, 1.0)),
        full_covariances=(
            ((0.20, 0.02), (0.02, 0.20)),
            ((0.20, -0.02), (-0.02, 0.20)),
        ),
    )


class DeterministicAdapter:
    def __init__(self) -> None:
        self._artifact = model_artifact()

    def fit(self, train_rows: object, state_count: int, seed: int) -> FitResult:
        assert state_count == 2
        values = np.asarray(train_rows, dtype=np.float64)
        return FitResult(
            artifact=self._artifact,
            train_log_likelihood=-float(np.sum(values * values)) + seed * 1e-6,
            converged=True,
            iterations=5,
            seed=seed,
        )

    def extract(self) -> GaussianHMMArtifact:
        return self._artifact

    def reconstruct(self, artifact: GaussianHMMArtifact) -> None:
        self._artifact = artifact

    def causal_filter(
        self,
        rows: object,
        initial_filtered_probabilities: tuple[float, ...] | None = None,
    ) -> FilterResult:
        raise AssertionError("runner uses backend-independent causal filter")


def source_rows(row_count: int) -> pd.DataFrame:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    timestamps = tuple(start + timedelta(days=index) for index in range(row_count))
    signs = np.where(np.arange(row_count) % 2 == 0, -1.0, 1.0)
    return pd.DataFrame(
        {
            "timestamp_m1": timestamps,
            "f0": signs,
            "f1": signs + np.where(signs > 0.0, 0.05, -0.05),
        }
    )


def evaluate(rows: pd.DataFrame):
    profile = load_profile(PROFILE_CONFIG)
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    return run_walk_forward_candidate(
        rows,
        plan=plan,
        profile=profile,
        candidate=candidate(),
        adapter_factory=DeterministicAdapter,
    )


def test_valid_folds_use_train_only_scaler_continued_test_filter_and_alignment() -> None:
    result = evaluate(source_rows(1386))
    assert len(result.folds) == 2
    assert result.valid_fold_rate == 1.0
    first, second = result.folds
    assert first.valid is True
    assert second.valid is True
    assert first.train_model_observation_count == 1260
    assert first.test_model_observation_count == 63
    assert first.alignment is not None and first.alignment.initial_alignment is True
    assert second.alignment is not None and second.alignment.initial_alignment is False
    assert first.scaler_artifact is not None
    assert first.scaler_artifact.feature_order == ("f0", "f1")
    assert first.oos_predictive_log_likelihood_per_observation is not None
    assert len(first.oos_filtered_probabilities) == 63
    assert all(sum(row) == pytest.approx(1.0) for row in first.oos_filtered_probabilities)
    assert min(first.train_hard_occupancy or ()) >= 0.03
    assert min(first.train_soft_occupancy or ()) >= 0.05


def test_walk_forward_evaluation_accepts_xetra_v2() -> None:
    result = evaluate(source_rows(1323))
    v2 = replace(result, profile_config_version=2)
    assert v2.profile_config_version == 2


def test_source_windowing_precedes_complete_case_filtering_and_records_gaps() -> None:
    rows = source_rows(1323)
    rows.loc[0:9, "f0"] = np.nan
    rows.loc[1260:1269, "f1"] = np.nan
    result = evaluate(rows)
    fold = result.folds[0]
    assert fold.valid is True
    assert fold.train_source_observation_count == 1260
    assert fold.train_model_observation_count == 1250
    assert fold.skipped_train_incomplete_count == 10
    assert fold.test_source_observation_count == 63
    assert fold.test_model_observation_count == 53
    assert fold.skipped_test_incomplete_count == 10
    assert len(fold.oos_timestamps) == 53


def test_below_42_retained_test_rows_invalidates_fold_with_explicit_reason() -> None:
    rows = source_rows(1323)
    rows.loc[1260:1281, "f0"] = np.nan
    result = evaluate(rows)
    fold = result.folds[0]
    assert fold.valid is False
    assert fold.test_model_observation_count == 41
    assert fold.failure_reason is not None
    assert "below pinned minimum 42" in fold.failure_reason
    assert fold.oos_filtered_probabilities == ()


def test_nonfinite_nonnull_selected_value_is_not_silently_dropped() -> None:
    rows = source_rows(1323)
    rows.loc[1300, "f0"] = np.inf
    fold = evaluate(rows).folds[0]
    assert fold.valid is False
    assert fold.failure_reason is not None
    assert "must be finite" in fold.failure_reason


def test_future_source_mutation_cannot_change_earlier_fold_evidence() -> None:
    original = source_rows(1386)
    baseline = evaluate(original)
    mutated = original.copy()
    mutated.loc[1323:, "f0"] = mutated.loc[1323:, "f0"] * 100.0 + 7.0
    mutated.loc[1323:, "f1"] = mutated.loc[1323:, "f1"] * -50.0
    changed = evaluate(mutated)
    assert baseline.folds[0] == changed.folds[0]
    assert baseline.folds[0].scaler_artifact == changed.folds[0].scaler_artifact


def test_plan_source_mismatch_fails_before_model_work() -> None:
    rows = source_rows(1323)
    profile = load_profile(PROFILE_CONFIG)
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    shifted = rows.copy()
    shifted.loc[0, "timestamp_m1"] = shifted.loc[0, "timestamp_m1"] - timedelta(days=1)
    with pytest.raises(ValueError, match="sequence start"):
        run_walk_forward_candidate(
            shifted,
            plan=plan,
            profile=profile,
            candidate=candidate(),
            adapter_factory=DeterministicAdapter,
        )
