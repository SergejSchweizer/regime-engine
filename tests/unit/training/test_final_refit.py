from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluation.walk_forward import run_walk_forward_candidate
from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import FilterResult, FitResult
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile
from market_regime_engine.states.alignment import StateAlignment
from market_regime_engine.training.final_refit import (
    _aligned_artifact,
    _refit_matrix,
    final_production_refit,
)

PROFILE_CONFIG = Path("configs/profiles/xetra_v1.yaml")
FEATURES = ("f0", "f1")


def candidate() -> ResolvedCandidateProfile:
    return ResolvedCandidateProfile(
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        covariance_type="full",
        feature_order=FEATURES,
        feature_dimension=2,
        source_build_id="build-1",
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        original_feature_universe=tuple(f"f{index}" for index in range(48)),
        preliminary_medoids=tuple(f"f{index}" for index in range(8)),
    )


def artifact() -> GaussianHMMArtifact:
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=FEATURES,
        start_probabilities=(0.5, 0.5),
        transition_matrix=((0.8, 0.2), (0.2, 0.8)),
        means=((-1.0, -1.0), (1.0, 1.0)),
        full_covariances=(
            ((0.20, 0.02), (0.02, 0.20)),
            ((0.20, -0.02), (-0.02, 0.20)),
        ),
    )


def test_alignment_preserves_gmm_mixture_emissions_in_persistent_order() -> None:
    base = artifact()
    gmm = replace(
        base,
        model_family="gmm_hmm",
        mixture_weights=((0.7, 0.3), (0.4, 0.6)),
        mixture_means=(
            ((-1.2, -1.1), (-0.8, -0.9)),
            ((0.8, 0.9), (1.2, 1.1)),
        ),
        mixture_full_covariances=(
            (base.full_covariances[0], base.full_covariances[0]),
            (base.full_covariances[1], base.full_covariances[1]),
        ),
    )
    alignment = StateAlignment(
        persistent_state_ids=("state_0", "state_1"),
        persistent_to_fitted=(1, 0),
        aligned_signatures=((1.0,), (-1.0,)),
        matched_rms=(0.0, 0.0),
        total_cost=0.0,
        max_drift=0.0,
        initial_alignment=False,
    )

    aligned = _aligned_artifact(gmm, alignment)

    assert aligned.mixture_weights == (gmm.mixture_weights[1], gmm.mixture_weights[0])
    assert aligned.mixture_means == (gmm.mixture_means[1], gmm.mixture_means[0])


class DeterministicAdapter:
    def __init__(self) -> None:
        self._artifact = artifact()

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

    def reconstruct(self, model_artifact: GaussianHMMArtifact) -> None:
        self._artifact = model_artifact

    def causal_filter(
        self,
        rows: object,
        initial_filtered_probabilities: tuple[float, ...] | None = None,
    ) -> FilterResult:
        raise AssertionError("final refit uses backend-independent causal filter")


def source_rows(row_count: int = 1323) -> pd.DataFrame:
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


def lineage(rows: pd.DataFrame) -> SourceLineage:
    return SourceLineage(
        source_dataset="regime_loader.regime_features_daily",
        source_build_id="build-1",
        data_sha256="d" * 64,
        schema_version=1,
        feature_version=1,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=datetime(2026, 8, 24, tzinfo=UTC),
        row_count=len(rows),
        min_timestamp=rows["timestamp_m1"].iloc[0],
        max_timestamp=rows["timestamp_m1"].iloc[-1],
    )


def winning_evaluation(rows: pd.DataFrame):
    profile = load_profile(PROFILE_CONFIG)
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    return run_walk_forward_candidate(
        rows,
        plan=plan,
        profile=profile,
        candidate=candidate(),
        adapter_factory=DeterministicAdapter,
    )


def test_final_refit_uses_full_sample_aligns_and_persists_filter_boundary() -> None:
    rows = source_rows()
    rows.loc[len(rows) - 1, "f0"] = np.nan
    evaluation = winning_evaluation(rows)
    result = final_production_refit(
        rows,
        lineage=lineage(rows),
        candidate=candidate(),
        winning_evaluation=evaluation,
        adapter_factory_builder=lambda item: DeterministicAdapter,
    )
    assert result.registered_model == "regime-xetra"
    assert result.candidate_id == "gaussian_hmm_k2_full"
    assert result.feature_order == FEATURES
    assert result.source_build_id == "build-1"
    assert result.source_data_sha256 == "d" * 64
    assert result.retained_observation_count == 1322
    assert result.skipped_incomplete_observation_count == 1
    assert result.inference_origin_timestamp == rows["timestamp_m1"].iloc[0]
    assert result.trained_through_timestamp == rows["timestamp_m1"].iloc[-2]
    assert result.trained_through_timestamp < result.evaluation_cutoff
    assert sum(result.terminal_filtered_probabilities) == pytest.approx(1.0)
    assert result.winning_seed == 131
    assert result.hmm.feature_order == result.scaler.feature_order == FEATURES


def test_rows_strictly_after_cutoff_cannot_change_final_refit() -> None:
    rows = source_rows()
    evaluation = winning_evaluation(rows)
    baseline = final_production_refit(
        rows,
        lineage=lineage(rows),
        candidate=candidate(),
        winning_evaluation=evaluation,
        adapter_factory_builder=lambda item: DeterministicAdapter,
    )
    future = rows.copy()
    for index in range(5):
        future.loc[len(future)] = {
            "timestamp_m1": rows["timestamp_m1"].iloc[-1] + timedelta(days=index + 1),
            "f0": 1000.0 + index,
            "f1": -1000.0 - index,
        }
    changed = final_production_refit(
        future,
        lineage=lineage(future),
        candidate=candidate(),
        winning_evaluation=evaluation,
        adapter_factory_builder=lambda item: DeterministicAdapter,
    )
    assert changed.scaler == baseline.scaler
    assert changed.hmm == baseline.hmm
    assert changed.terminal_filtered_probabilities == baseline.terminal_filtered_probabilities
    assert changed.trained_through_timestamp == baseline.trained_through_timestamp


def test_final_refit_rejects_champion_source_and_selection_drift() -> None:
    rows = source_rows()
    evaluation = winning_evaluation(rows)
    with pytest.raises(ValueError, match="statistical champion"):
        final_production_refit(
            rows,
            lineage=lineage(rows),
            candidate=replace(candidate(), candidate_id="gaussian_hmm_k3_full", state_count=3),
            winning_evaluation=evaluation,
            adapter_factory_builder=lambda item: DeterministicAdapter,
        )
    with pytest.raises(ValueError, match="source lineage"):
        final_production_refit(
            rows,
            lineage=replace(lineage(rows), source_build_id="other-build"),
            candidate=candidate(),
            winning_evaluation=evaluation,
            adapter_factory_builder=lambda item: DeterministicAdapter,
        )
    drifted = replace(candidate(), feature_selection_execution_hash="c" * 64)
    with pytest.raises(ValueError, match="selection hashes"):
        final_production_refit(
            rows,
            lineage=lineage(rows),
            candidate=drifted,
            winning_evaluation=evaluation,
            adapter_factory_builder=lambda item: DeterministicAdapter,
        )


def test_refit_matrix_enforces_cutoff_reach_order_finiteness_and_minimum() -> None:
    rows = source_rows(600)
    cutoff = rows["timestamp_m1"].iloc[-1]
    matrix, timestamps, skipped = _refit_matrix(
        rows,
        feature_order=FEATURES,
        evaluation_cutoff=cutoff,
    )
    assert matrix.shape == (600, 2)
    assert len(timestamps) == 600
    assert skipped == 0
    with pytest.raises(ValueError, match="does not reach"):
        _refit_matrix(
            rows,
            feature_order=FEATURES,
            evaluation_cutoff=cutoff + timedelta(days=1),
        )
    reversed_rows = rows.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="strictly increasing"):
        _refit_matrix(reversed_rows, feature_order=FEATURES, evaluation_cutoff=cutoff)
    nonfinite = rows.copy()
    nonfinite.loc[10, "f0"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        _refit_matrix(nonfinite, feature_order=FEATURES, evaluation_cutoff=cutoff)
    short = source_rows(503)
    with pytest.raises(ValueError, match="at least 504"):
        _refit_matrix(
            short,
            feature_order=FEATURES,
            evaluation_cutoff=short["timestamp_m1"].iloc[-1],
        )
