"""Causal provisional-teacher reference construction and frozen refitting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from math import isfinite

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluations.provisional_teacher import (
    ProvisionalTeacherEvaluation,
)
from market_regime_engine.feature_discovery.contracts import (
    ProvisionalTeacherReference,
)
from market_regime_engine.inference.filtering import causal_filter
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.gaussian_hmm import HmmlearnGaussianHMMAdapter
from market_regime_engine.models.protocols import GaussianHMMAdapter
from market_regime_engine.preprocessing.scaling import (
    StandardScalerArtifact,
    fit_standard_scaler,
)
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.training.multistart import MultistartResult, run_multistart

_TIMESTAMP_COLUMN = "timestamp_m1"
FrozenTeacherAdapterFactory = Callable[[], GaussianHMMAdapter]


def _utc(value: object, field_name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != UTC.utcoffset(value)
    ):
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    return value


def _validate_reference_evaluation(
    evaluation: ProvisionalTeacherEvaluation,
) -> tuple[GaussianHMMArtifact, tuple[datetime, ...], tuple[tuple[float, ...], ...]]:
    selection = evaluation.selection
    if selection is None:
        raise ValueError(
            "cannot build a teacher reference without a selected provisional Gaussian K"
        )
    winner_id = selection.champion_candidate_id
    winning = next(
        (
            candidate
            for candidate in evaluation.candidate_evaluations
            if candidate.candidate_id == winner_id
        ),
        None,
    )
    if winning is None:
        raise ValueError("selected provisional teacher candidate is missing from evaluations")

    timestamps: list[datetime] = []
    probabilities: list[tuple[float, ...]] = []
    for fold in winning.valid_folds:
        if len(fold.oos_timestamps) != len(fold.oos_filtered_probabilities):
            raise ValueError("teacher fold timestamps and probabilities must have equal length")
        for timestamp, row in zip(
            fold.oos_timestamps, fold.oos_filtered_probabilities, strict=True
        ):
            timestamp = _utc(timestamp, "teacher OOS timestamp")
            if timestamps and timestamp <= timestamps[-1]:
                raise ValueError("teacher OOS timestamps must be unique and strictly increasing")
            if (
                len(row) != winning.state_count
                or any(value < 0.0 or not isfinite(value) for value in row)
                or abs(sum(row) - 1.0) > 1.0e-10
            ):
                raise ValueError("teacher OOS probabilities must be finite and normalized")
            timestamps.append(timestamp)
            probabilities.append(tuple(float(value) for value in row))
    if not timestamps:
        raise ValueError("selected provisional teacher has no valid inner TEST probabilities")
    artifact = next(
        (fold.model_artifact for fold in winning.valid_folds if fold.model_artifact is not None),
        None,
    )
    if artifact is None:
        raise ValueError("selected provisional teacher is missing model artifact evidence")
    return artifact, tuple(timestamps), tuple(probabilities)


def build_provisional_teacher_reference(
    evaluation: ProvisionalTeacherEvaluation,
) -> ProvisionalTeacherReference:
    """Collect only aligned causal filtered probabilities from valid inner TEST rows."""

    _artifact, timestamps, probabilities = _validate_reference_evaluation(evaluation)
    selection = evaluation.selection
    assert selection is not None
    state_count = selection.champion_state_count
    dominant_states = tuple(
        max(range(state_count), key=lambda index: (row[index], -index)) for row in probabilities
    )
    return ProvisionalTeacherReference(
        candidate_id=selection.champion_candidate_id,
        state_count=state_count,
        timestamps=timestamps,
        filtered_probabilities=probabilities,
        dominant_states=dominant_states,
        valid_inner_fold_ids=tuple(
            fold.fold_id
            for fold in next(
                candidate
                for candidate in evaluation.candidate_evaluations
                if candidate.candidate_id == selection.champion_candidate_id
            ).valid_folds
        ),
        source_build_id=evaluation.source_build_id,
        inner_plan_hash=evaluation.inner_plan.plan_hash,
        prototype_features=evaluation.prototype_features,
    )


def _complete_case(
    frame: pd.DataFrame,
    feature_order: tuple[str, ...],
    name: str,
) -> tuple[np.ndarray, tuple[datetime, ...]]:
    if _TIMESTAMP_COLUMN not in frame.columns:
        raise ValueError(f"{name} rows must contain {_TIMESTAMP_COLUMN}")
    missing = tuple(feature for feature in feature_order if feature not in frame.columns)
    if missing:
        raise ValueError(f"{name} rows are missing frozen teacher features: {missing}")
    timestamps = tuple(_utc(value, f"{name} timestamp") for value in frame[_TIMESTAMP_COLUMN])
    if any(left >= right for left, right in pairwise(timestamps)):
        raise ValueError(f"{name} timestamps must be strictly increasing and unique")
    selected = frame.loc[:, list(feature_order)]
    complete_mask = selected.notna().all(axis=1)
    complete = selected.loc[complete_mask]
    try:
        matrix = complete.to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} teacher features must be numeric") from exc
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_order):
        raise ValueError(f"{name} teacher matrix must preserve the frozen feature order")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} teacher features must be finite")
    retained_timestamps = tuple(
        timestamp for timestamp, retained in zip(timestamps, complete_mask, strict=True) if retained
    )
    return matrix, retained_timestamps


@dataclass(frozen=True, slots=True)
class FrozenTeacherRefit:
    """One frozen-K TRAIN fit continued causally once into TEST."""

    candidate_id: str
    state_count: int
    prototype_features: tuple[str, ...]
    scaler: StandardScalerArtifact
    multistart_result: MultistartResult
    model_artifact: GaussianHMMArtifact
    train_timestamps: tuple[datetime, ...]
    test_timestamps: tuple[datetime, ...]
    train_filtered_probabilities: tuple[tuple[float, ...], ...]
    test_filtered_probabilities: tuple[tuple[float, ...], ...]
    train_log_likelihood: float
    test_log_likelihood: float
    test_log_likelihood_per_observation: float

    def __post_init__(self) -> None:
        if self.candidate_id != f"gaussian_hmm_k{self.state_count}_full":
            raise ValueError("frozen teacher candidate identity is invalid")
        if self.state_count not in (2, 3, 4, 5):
            raise ValueError("frozen teacher state count must be K2/K3/K4/K5")
        if not self.prototype_features or len(set(self.prototype_features)) != len(
            self.prototype_features
        ):
            raise ValueError("frozen teacher prototype features must be non-empty and unique")
        if self.model_artifact.state_count != self.state_count:
            raise ValueError("frozen teacher artifact state count differs")
        if self.model_artifact.feature_order != self.prototype_features:
            raise ValueError("frozen teacher artifact feature order differs")
        if self.scaler.feature_order != self.prototype_features:
            raise ValueError("frozen teacher scaler feature order differs")
        if not self.train_timestamps or not self.test_timestamps:
            raise ValueError("frozen teacher refit requires retained TRAIN and TEST rows")
        if any(
            current <= previous
            for previous, current in pairwise((*self.train_timestamps, *self.test_timestamps))
        ):
            raise ValueError("frozen teacher TRAIN/TEST timestamps must be strictly increasing")
        if len(self.train_filtered_probabilities) != len(self.train_timestamps):
            raise ValueError("frozen teacher TRAIN probabilities do not match timestamps")
        if len(self.test_filtered_probabilities) != len(self.test_timestamps):
            raise ValueError("frozen teacher TEST probabilities do not match timestamps")
        for rows, name in (
            (self.train_filtered_probabilities, "TRAIN"),
            (self.test_filtered_probabilities, "TEST"),
        ):
            for row in rows:
                if (
                    len(row) != self.state_count
                    or any(value < 0.0 or not isfinite(value) for value in row)
                    or abs(sum(row) - 1.0) > 1.0e-10
                ):
                    raise ValueError(f"frozen teacher {name} probabilities are invalid")
        for value, name in (
            (self.train_log_likelihood, "TRAIN log likelihood"),
            (self.test_log_likelihood, "TEST log likelihood"),
            (self.test_log_likelihood_per_observation, "TEST per-observation likelihood"),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.test_log_likelihood_per_observation != (
            self.test_log_likelihood / len(self.test_timestamps)
        ):
            raise ValueError("TEST per-observation likelihood does not reconcile")


def refit_frozen_teacher(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    *,
    reference: ProvisionalTeacherReference,
    profile: ModelProfile,
    adapter_factory: FrozenTeacherAdapterFactory | None = None,
) -> FrozenTeacherRefit:
    """Refit the reference's fixed Gaussian K/features and continue once into TEST.

    The reference supplies the state count and prototype tuple.  No selection,
    candidate ranking, prototype selection, or outer TEST access is performed.
    """

    if profile.profile_id != "xetra" or profile.profile_config_version != 4:
        raise ValueError("frozen teacher refit requires the canonical Xetra v4 profile")
    if not reference.prototype_features:
        raise ValueError("frozen teacher refit requires prototype feature lineage")
    train_matrix, train_timestamps = _complete_case(
        train_rows, reference.prototype_features, "TRAIN"
    )
    test_matrix, test_timestamps = _complete_case(test_rows, reference.prototype_features, "TEST")
    if train_matrix.shape[0] < profile.walk_forward.minimum_model_train_observations:
        raise ValueError("frozen teacher TRAIN rows are below the model minimum")
    if test_matrix.shape[0] < profile.walk_forward.minimum_model_test_observations:
        raise ValueError("frozen teacher TEST rows are below the model minimum")
    if train_timestamps[-1] >= test_timestamps[0]:
        raise ValueError("frozen teacher TRAIN and TEST windows must not overlap")

    scaler = fit_standard_scaler(train_matrix, reference.prototype_features)
    scaled_train = scaler.transform(train_matrix)
    scaled_test = scaler.transform(test_matrix)
    active_factory = (
        adapter_factory
        if adapter_factory is not None
        else lambda: HmmlearnGaussianHMMAdapter(reference.prototype_features)
    )
    multistart = run_multistart(
        scaled_train,
        state_count=reference.state_count,
        adapter_factory=active_factory,
    )
    artifact = multistart.winner.artifact
    if artifact.feature_order != reference.prototype_features:
        raise ValueError("frozen teacher artifact feature order differs")
    if artifact.state_count != reference.state_count:
        raise ValueError("frozen teacher artifact state count differs")
    train_filter = causal_filter(scaled_train, artifact)
    test_filter = causal_filter(
        scaled_test,
        artifact,
        initial_filtered_probabilities=train_filter.terminal_probabilities,
    )
    return FrozenTeacherRefit(
        candidate_id=reference.candidate_id,
        state_count=reference.state_count,
        prototype_features=reference.prototype_features,
        scaler=scaler,
        multistart_result=multistart,
        model_artifact=artifact,
        train_timestamps=train_timestamps,
        test_timestamps=test_timestamps,
        train_filtered_probabilities=tuple(
            tuple(float(value) for value in row) for row in train_filter.filtered_probabilities
        ),
        test_filtered_probabilities=tuple(
            tuple(float(value) for value in row) for row in test_filter.filtered_probabilities
        ),
        train_log_likelihood=train_filter.log_likelihood,
        test_log_likelihood=test_filter.log_likelihood,
        test_log_likelihood_per_observation=test_filter.log_likelihood / len(test_timestamps),
    )


create_provisional_teacher_reference = build_provisional_teacher_reference
refit_frozen_provisional_teacher = refit_frozen_teacher


__all__ = [
    "FrozenTeacherAdapterFactory",
    "FrozenTeacherRefit",
    "build_provisional_teacher_reference",
    "create_provisional_teacher_reference",
    "refit_frozen_provisional_teacher",
    "refit_frozen_teacher",
]
