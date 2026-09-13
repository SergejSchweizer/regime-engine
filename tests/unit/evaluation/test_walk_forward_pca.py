from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.walk_forward import (
    WalkForwardEvaluation,
    run_walk_forward_candidate,
)
from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.inference.filtering import causal_filter
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import FilterResult, FitResult
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile

PROFILE_CONFIG = Path("configs/profiles/xetra_v4.yaml")
FEATURES = ("f0", "f1", "pca_pc_001")


def candidate() -> ResolvedCandidateProfile:
    return ResolvedCandidateProfile(
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        covariance_type="full",
        feature_order=FEATURES,
        feature_dimension=len(FEATURES),
        source_build_id="build-pca-1",
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        original_feature_universe=FEATURES,
    )


def model_artifact() -> GaussianHMMArtifact:
    identity = (
        (0.20, 0.0, 0.0),
        (0.0, 0.20, 0.0),
        (0.0, 0.0, 0.20),
    )
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=FEATURES,
        start_probabilities=(0.5, 0.5),
        transition_matrix=((0.8, 0.2), (0.2, 0.8)),
        means=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        full_covariances=(identity, identity),
    )


class DeterministicAdapter:
    def __init__(self) -> None:
        self._artifact = model_artifact()

    def fit(self, train_rows: object, state_count: int, seed: int) -> FitResult:
        assert state_count == 2
        values = np.asarray(train_rows, dtype=np.float64)
        return FitResult(
            artifact=self._artifact,
            train_log_likelihood=causal_filter(values, self._artifact).log_likelihood,
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
    index = np.arange(row_count, dtype=np.float64)
    signs = np.where(index % 2 == 0, -1.0, 1.0)
    return pd.DataFrame(
        {
            "timestamp_m1": tuple(start + timedelta(days=int(value)) for value in index),
            "f0": signs,
            "f1": signs + np.where(signs > 0.0, 0.05, -0.05),
        }
    )


def evaluate(rows: pd.DataFrame) -> WalkForwardEvaluation:
    profile = load_profile(PROFILE_CONFIG)
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    return run_walk_forward_candidate(
        rows,
        plan=plan,
        profile=profile,
        candidate=candidate(),
        adapter_factory=DeterministicAdapter,
        pca_raw_feature_order=("f0", "f1"),
    )


def test_walk_forward_fits_fold_local_pca_then_hmm_scaler() -> None:
    result = evaluate(source_rows(1323))
    fold = result.folds[0]

    assert fold.valid is True
    assert fold.pca_scaler_artifact is not None
    assert fold.scaler_artifact == fold.pca_scaler_artifact.hmm_scaler
    assert fold.pca_scaler_artifact.raw_feature_order == ("f0", "f1")
    assert fold.pca_scaler_artifact.model_feature_order == FEATURES
    assert fold.pca_scaler_artifact.pca_fit.clock.inner_fold_id == "fold_001"
    assert fold.pca_scaler_artifact.pca_fit.selected_row_count == 1260


def test_future_rows_cannot_mutate_earlier_fold_pca_evidence() -> None:
    baseline = evaluate(source_rows(1386))
    mutated = source_rows(1386)
    mutated.loc[1323:, "f0"] *= 100.0
    mutated.loc[1323:, "f1"] *= -50.0

    changed = evaluate(mutated)

    assert baseline.folds[0] == changed.folds[0]
