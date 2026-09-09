from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd

from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan
from market_regime_engine.evaluations.teacher_reference import (
    FrozenTeacherRefit,
    build_provisional_teacher_reference,
    refit_frozen_teacher,
)
from market_regime_engine.feature_discovery.contracts import ProvisionalTeacherReference
from market_regime_engine.inference.filtering import causal_filter
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import FitResult
from market_regime_engine.profiles.loader import load_profile

HASH = "a" * 64
FEATURES = ("prototype_0",)
START = datetime(2020, 1, 1, tzinfo=UTC)


def artifact() -> GaussianHMMArtifact:
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=FEATURES,
        start_probabilities=(0.6, 0.4),
        transition_matrix=((0.8, 0.2), (0.3, 0.7)),
        means=((-1.0,), (1.0,)),
        full_covariances=(((0.5,),), ((0.75,),)),
    )


def teacher_evaluation() -> SimpleNamespace:
    timestamps = tuple(START + timedelta(days=index) for index in range(4))
    valid_fold = SimpleNamespace(
        fold_id="fold_001",
        oos_timestamps=timestamps[:2],
        oos_filtered_probabilities=((0.8, 0.2), (0.1, 0.9)),
        model_artifact=artifact(),
    )
    invalid_fold = SimpleNamespace(
        fold_id="fold_002",
        oos_timestamps=timestamps[2:],
        oos_filtered_probabilities=((0.4, 0.6), (0.5, 0.5)),
        model_artifact=None,
    )
    winning = SimpleNamespace(
        candidate_id="gaussian_hmm_k2_full",
        folds=(valid_fold, invalid_fold),
        valid_folds=(valid_fold,),
        state_count=2,
    )
    return SimpleNamespace(
        selection=SimpleNamespace(
            champion_candidate_id="gaussian_hmm_k2_full", champion_state_count=2
        ),
        candidate_evaluations=(winning,),
        prototype_features=FEATURES,
        source_build_id="build-1",
        inner_plan=WalkForwardPlan((), None, HASH),
    )


def test_reference_contains_only_aligned_causal_valid_inner_test_rows() -> None:
    evaluation = teacher_evaluation()
    reference = build_provisional_teacher_reference(evaluation)

    assert reference.candidate_id == "gaussian_hmm_k2_full"
    assert reference.state_count == 2
    assert reference.timestamps == tuple(START + timedelta(days=index) for index in range(2))
    assert reference.filtered_probabilities == ((0.8, 0.2), (0.1, 0.9))
    assert reference.dominant_states == (0, 1)
    assert reference.valid_inner_fold_ids == ("fold_001",)
    assert reference.prototype_features == FEATURES
    assert reference.reference_hash != ""


class DeterministicAdapter:
    fit_state_counts: ClassVar[list[int]] = []

    def __init__(self) -> None:
        self._artifact = artifact()

    def fit(self, train_rows: object, state_count: int, seed: int) -> FitResult:
        values = np.asarray(train_rows, dtype=np.float64)
        self.fit_state_counts.append(state_count)
        return FitResult(
            artifact=self._artifact,
            train_log_likelihood=causal_filter(values, self._artifact).log_likelihood,
            converged=True,
            iterations=4,
            seed=seed,
        )

    def extract(self) -> GaussianHMMArtifact:
        return self._artifact

    def reconstruct(self, model_artifact: GaussianHMMArtifact) -> None:
        self._artifact = model_artifact

    def causal_filter(self, rows: object, initial_filtered_probabilities=None):
        return causal_filter(
            rows,
            self._artifact,
            initial_filtered_probabilities=initial_filtered_probabilities,
        )


def refit_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_count, test_count = 504, 42
    train_timestamps = tuple(START + timedelta(days=index) for index in range(train_count))
    test_timestamps = tuple(
        START + timedelta(days=train_count + index) for index in range(test_count)
    )
    train_values = np.sin(np.arange(train_count, dtype=np.float64) / 9.0)
    test_values = np.cos(np.arange(test_count, dtype=np.float64) / 7.0)
    return (
        pd.DataFrame({"timestamp_m1": train_timestamps, FEATURES[0]: train_values}),
        pd.DataFrame({"timestamp_m1": test_timestamps, FEATURES[0]: test_values}),
    )


def test_frozen_refit_uses_reference_k_and_continues_from_train_terminal_alpha() -> None:
    train, test = refit_rows()
    reference = ProvisionalTeacherReference(
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        timestamps=(START,),
        filtered_probabilities=((0.5, 0.5),),
        dominant_states=(0,),
        valid_inner_fold_ids=("fold_001",),
        source_build_id="build-1",
        inner_plan_hash=HASH,
        prototype_features=FEATURES,
    )
    DeterministicAdapter.fit_state_counts = []

    result = refit_frozen_teacher(
        train,
        test,
        reference=reference,
        profile=load_profile("configs/profiles/xetra_v4.yaml"),
        adapter_factory=DeterministicAdapter,
    )

    assert isinstance(result, FrozenTeacherRefit)
    assert result.candidate_id == reference.candidate_id
    assert result.state_count == reference.state_count
    assert len(result.train_timestamps) == 504
    assert len(result.test_timestamps) == 42
    assert len(result.train_filtered_probabilities) == 504
    assert len(result.test_filtered_probabilities) == 42
    assert DeterministicAdapter.fit_state_counts == [2] * 8
    assert result.test_log_likelihood_per_observation == (
        result.test_log_likelihood / len(result.test_timestamps)
    )
