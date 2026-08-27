"""Leak-free walk-forward evaluation over the frozen complete-case HMM clock."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from math import isfinite

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.diagnostics import (
    InformationCriteria,
    OccupancyDiagnostics,
    dominant_state_durations,
    information_criteria,
    occupancy,
    switches_per_year,
    uncertainty,
    validate_full_covariances,
    validate_train_occupancy,
)
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardFold, WalkForwardPlan
from market_regime_engine.inference.filtering import causal_filter
from market_regime_engine.inference.predictive_likelihood import (
    continued_test_predictive_likelihood,
)
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import GaussianHMMAdapter
from market_regime_engine.preprocessing.scaling import StandardScalerArtifact, fit_standard_scaler
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile
from market_regime_engine.states.alignment import (
    StateAlignment,
    align_first_fold,
    align_to_reference,
)
from market_regime_engine.states.signatures import StateSignature
from market_regime_engine.training.multistart import MultistartResult, run_multistart

AdapterFactory = Callable[[], GaussianHMMAdapter]
_TIMESTAMP_COLUMN = "timestamp_m1"


def _require_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    return value


def _finite_optional(value: float | None, field_name: str) -> None:
    if value is not None and not isfinite(value):
        raise ValueError(f"{field_name} must be finite when present")


@dataclass(frozen=True, slots=True)
class WalkForwardFoldResult:
    fold_id: str
    fold_index: int
    valid: bool
    failure_reason: str | None
    train_source_observation_count: int
    test_source_observation_count: int
    train_model_observation_count: int
    test_model_observation_count: int
    skipped_train_incomplete_count: int
    skipped_test_incomplete_count: int
    scaler_artifact: StandardScalerArtifact | None = None
    multistart_result: MultistartResult | None = None
    model_artifact: GaussianHMMArtifact | None = None
    alignment: StateAlignment | None = None
    train_log_likelihood: float | None = None
    oos_predictive_log_likelihood: float | None = None
    oos_predictive_log_likelihood_per_observation: float | None = None
    aic: float | None = None
    bic: float | None = None
    multistart_success_rate: float | None = None
    train_hard_occupancy: tuple[float, ...] | None = None
    train_soft_occupancy: tuple[float, ...] | None = None
    oos_hard_occupancy: tuple[float, ...] | None = None
    oos_soft_occupancy: tuple[float, ...] | None = None
    max_state_signature_drift: float | None = None
    mean_state_duration: float | None = None
    switches_per_year: float | None = None
    oos_entropy_mean: float | None = None
    oos_confidence_mean: float | None = None
    oos_timestamps: tuple[datetime, ...] = ()
    oos_filtered_probabilities: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.fold_index < 1 or self.fold_id != f"fold_{self.fold_index:03d}":
            raise ValueError("fold result identity must be deterministic and one-based")
        counts = (
            self.train_source_observation_count,
            self.test_source_observation_count,
            self.train_model_observation_count,
            self.test_model_observation_count,
            self.skipped_train_incomplete_count,
            self.skipped_test_incomplete_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("fold observation counts cannot be negative")
        if self.train_model_observation_count + self.skipped_train_incomplete_count != (
            self.train_source_observation_count
        ):
            raise ValueError("TRAIN model/skipped counts must reconcile to source rows")
        if self.test_model_observation_count + self.skipped_test_incomplete_count != (
            self.test_source_observation_count
        ):
            raise ValueError("TEST model/skipped counts must reconcile to source rows")
        if self.valid == (self.failure_reason is not None):
            raise ValueError("valid fold must have no failure reason; invalid fold must have one")
        for field_name in (
            "train_log_likelihood",
            "oos_predictive_log_likelihood",
            "oos_predictive_log_likelihood_per_observation",
            "aic",
            "bic",
            "multistart_success_rate",
            "max_state_signature_drift",
            "mean_state_duration",
            "switches_per_year",
            "oos_entropy_mean",
            "oos_confidence_mean",
        ):
            _finite_optional(getattr(self, field_name), field_name)
        for timestamp in self.oos_timestamps:
            _require_utc(timestamp, "OOS timestamp")
        if self.valid:
            required = (
                self.scaler_artifact,
                self.multistart_result,
                self.model_artifact,
                self.alignment,
                self.train_log_likelihood,
                self.oos_predictive_log_likelihood,
                self.oos_predictive_log_likelihood_per_observation,
                self.aic,
                self.bic,
                self.multistart_success_rate,
                self.train_hard_occupancy,
                self.train_soft_occupancy,
                self.oos_hard_occupancy,
                self.oos_soft_occupancy,
                self.max_state_signature_drift,
                self.mean_state_duration,
                self.oos_entropy_mean,
                self.oos_confidence_mean,
            )
            if any(value is None for value in required):
                raise ValueError("valid fold is missing required evaluation evidence")
            if len(self.oos_timestamps) != self.test_model_observation_count:
                raise ValueError("valid fold OOS timestamps must match retained TEST observations")
            if len(self.oos_filtered_probabilities) != self.test_model_observation_count:
                raise ValueError(
                    "valid fold OOS probabilities must match retained TEST observations"
                )


@dataclass(frozen=True, slots=True)
class WalkForwardEvaluation:
    profile_id: str
    profile_config_version: int
    candidate_id: str
    state_count: int
    source_build_id: str
    feature_order: tuple[str, ...]
    feature_selection_definition_hash: str
    feature_selection_execution_hash: str
    evaluation_plan_hash: str
    evaluation_cutoff: datetime
    folds: tuple[WalkForwardFoldResult, ...]

    def __post_init__(self) -> None:
        if self.profile_id != "xetra" or self.profile_config_version not in {1, 2}:
            raise ValueError("walk-forward evaluation requires a supported xetra profile version")
        if self.state_count not in (2, 3, 4, 5):
            raise ValueError("walk-forward evaluation supports exactly K2/K3/K4/K5")
        expected_candidates = {
            f"gaussian_hmm_k{self.state_count}_full",
            "gmm_hmm_k2_m2_full",
        }
        if self.candidate_id not in expected_candidates:
            raise ValueError("candidate identity is unsupported")
        if not self.folds:
            raise ValueError("walk-forward evaluation requires at least one planned fold")
        if tuple(result.fold_index for result in self.folds) != tuple(
            range(1, len(self.folds) + 1)
        ):
            raise ValueError("walk-forward results must preserve complete planned fold order")
        _require_utc(self.evaluation_cutoff, "evaluation_cutoff")
        if (
            self.folds[-1].oos_timestamps
            and self.folds[-1].oos_timestamps[-1] > self.evaluation_cutoff
        ):
            raise ValueError("OOS evidence cannot extend beyond evaluation cutoff")

    @property
    def valid_folds(self) -> tuple[WalkForwardFoldResult, ...]:
        return tuple(fold for fold in self.folds if fold.valid)

    @property
    def valid_fold_rate(self) -> float:
        return len(self.valid_folds) / len(self.folds)


def _validate_source_rows(
    source_rows: pd.DataFrame,
    feature_order: tuple[str, ...],
) -> tuple[datetime, ...]:
    if _TIMESTAMP_COLUMN not in source_rows.columns:
        raise ValueError(f"source rows must contain {_TIMESTAMP_COLUMN}")
    missing = tuple(feature for feature in feature_order if feature not in source_rows.columns)
    if missing:
        raise ValueError(f"source rows are missing resolved features: {', '.join(missing)}")
    timestamps = tuple(
        _require_utc(value, _TIMESTAMP_COLUMN) for value in source_rows[_TIMESTAMP_COLUMN]
    )
    if any(left >= right for left, right in pairwise(timestamps)):
        raise ValueError("source timestamps must be strictly increasing and unique")
    return timestamps


def _fold_source_frames(
    source_rows: pd.DataFrame,
    fold: WalkForwardFold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_stop = fold.train_source_observations
    test_stop = train_stop + fold.test_source_observations
    if test_stop > len(source_rows):
        raise ValueError(f"{fold.fold_id} source bounds exceed supplied source rows")
    train = source_rows.iloc[:train_stop]
    test = source_rows.iloc[train_stop:test_stop]
    if len(train) != fold.train_source_observations or len(test) != fold.test_source_observations:
        raise ValueError(f"{fold.fold_id} source-row counts do not match planned fold")
    observed_bounds = (
        train[_TIMESTAMP_COLUMN].iloc[0],
        train[_TIMESTAMP_COLUMN].iloc[-1],
        test[_TIMESTAMP_COLUMN].iloc[0],
        test[_TIMESTAMP_COLUMN].iloc[-1],
    )
    planned_bounds = (fold.train_start, fold.train_end, fold.test_start, fold.test_end)
    if observed_bounds != planned_bounds:
        raise ValueError(f"{fold.fold_id} source timestamps do not match evaluation plan")
    return train, test


def _complete_case(
    frame: pd.DataFrame,
    feature_order: tuple[str, ...],
) -> tuple[np.ndarray, tuple[datetime, ...], int]:
    selected = frame.loc[:, list(feature_order)]
    complete_mask = selected.notna().all(axis=1)
    complete = selected.loc[complete_mask]
    try:
        matrix = complete.to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("selected non-null feature values must be numeric") from exc
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_order):
        raise ValueError("complete-case matrix must preserve exact resolved feature order")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("selected non-null feature values must be finite")
    timestamps = tuple(
        _require_utc(value, "retained timestamp")
        for value in frame.loc[complete_mask, _TIMESTAMP_COLUMN]
    )
    skipped = len(frame) - len(complete)
    return matrix, timestamps, skipped


def _aligned_occupancy(
    diagnostics: OccupancyDiagnostics,
    alignment: StateAlignment,
) -> OccupancyDiagnostics:
    mapping = alignment.persistent_to_fitted
    return OccupancyDiagnostics(
        hard=tuple(diagnostics.hard[index] for index in mapping),
        soft=tuple(diagnostics.soft[index] for index in mapping),
    )


def _aligned_probabilities(
    probabilities: np.ndarray,
    alignment: StateAlignment,
) -> np.ndarray:
    result = probabilities[:, list(alignment.persistent_to_fitted)]
    if not np.all(np.isfinite(result)):
        raise ValueError("aligned filtered probabilities must be finite")
    return result


def _valid_fold_result(
    *,
    fold: WalkForwardFold,
    train_model_count: int,
    test_model_count: int,
    skipped_train: int,
    skipped_test: int,
    scaler: StandardScalerArtifact,
    multistart: MultistartResult,
    artifact: GaussianHMMArtifact,
    alignment: StateAlignment,
    criteria: InformationCriteria,
    train_occupancy: OccupancyDiagnostics,
    oos_occupancy: OccupancyDiagnostics,
    train_log_likelihood: float,
    oos_log_likelihood: float,
    oos_per_observation: float,
    oos_timestamps: tuple[datetime, ...],
    oos_probabilities: np.ndarray,
) -> WalkForwardFoldResult:
    uncertainty_diagnostics = uncertainty(oos_probabilities)
    durations = dominant_state_durations(oos_probabilities)
    return WalkForwardFoldResult(
        fold_id=fold.fold_id,
        fold_index=fold.fold_index,
        valid=True,
        failure_reason=None,
        train_source_observation_count=fold.train_source_observations,
        test_source_observation_count=fold.test_source_observations,
        train_model_observation_count=train_model_count,
        test_model_observation_count=test_model_count,
        skipped_train_incomplete_count=skipped_train,
        skipped_test_incomplete_count=skipped_test,
        scaler_artifact=scaler,
        multistart_result=multistart,
        model_artifact=artifact,
        alignment=alignment,
        train_log_likelihood=train_log_likelihood,
        oos_predictive_log_likelihood=oos_log_likelihood,
        oos_predictive_log_likelihood_per_observation=oos_per_observation,
        aic=criteria.aic,
        bic=criteria.bic,
        multistart_success_rate=multistart.success_rate,
        train_hard_occupancy=train_occupancy.hard,
        train_soft_occupancy=train_occupancy.soft,
        oos_hard_occupancy=oos_occupancy.hard,
        oos_soft_occupancy=oos_occupancy.soft,
        max_state_signature_drift=alignment.max_drift,
        mean_state_duration=float(np.mean(durations)),
        switches_per_year=switches_per_year(oos_timestamps, oos_probabilities),
        oos_entropy_mean=float(np.mean(uncertainty_diagnostics.entropy)),
        oos_confidence_mean=float(np.mean(uncertainty_diagnostics.confidence)),
        oos_timestamps=oos_timestamps,
        oos_filtered_probabilities=tuple(
            tuple(float(value) for value in row) for row in oos_probabilities
        ),
    )


def _invalid_fold_result(
    fold: WalkForwardFold,
    *,
    train_model_count: int,
    test_model_count: int,
    skipped_train: int,
    skipped_test: int,
    failure_reason: str,
) -> WalkForwardFoldResult:
    return WalkForwardFoldResult(
        fold_id=fold.fold_id,
        fold_index=fold.fold_index,
        valid=False,
        failure_reason=failure_reason,
        train_source_observation_count=fold.train_source_observations,
        test_source_observation_count=fold.test_source_observations,
        train_model_observation_count=train_model_count,
        test_model_observation_count=test_model_count,
        skipped_train_incomplete_count=skipped_train,
        skipped_test_incomplete_count=skipped_test,
    )


def run_walk_forward_candidate(
    source_rows: pd.DataFrame,
    *,
    plan: WalkForwardPlan,
    profile: ModelProfile,
    candidate: ResolvedCandidateProfile,
    adapter_factory: AdapterFactory,
) -> WalkForwardEvaluation:
    """Evaluate one frozen-feature K candidate without rerunning feature selection."""

    if profile.profile_id != "xetra" or profile.profile_config_version not in {1, 2}:
        raise ValueError("walk-forward runner supports xetra profile configurations 1 and 2")
    if candidate.model_family == "gaussian_hmm":
        if candidate.state_count not in profile.gaussian_hmm.candidate_states:
            raise ValueError("resolved Gaussian candidate state count is absent from model profile")
    elif (
        profile.gmm_hmm is None
        or candidate.state_count != profile.gmm_hmm.state_count
        or candidate.mixture_count != profile.gmm_hmm.mixture_count
    ):
        raise ValueError("resolved GMM-HMM candidate differs from the model profile")
    if candidate.feature_order == ():
        raise ValueError("resolved candidate feature order cannot be empty")
    if candidate.feature_dimension != len(candidate.feature_order):
        raise ValueError("resolved candidate feature dimension is inconsistent")
    if plan.evaluation_cutoff is None or not plan.folds:
        raise ValueError("walk-forward plan must contain at least one complete fold")
    timestamps = _validate_source_rows(source_rows, candidate.feature_order)
    if timestamps[0] != plan.folds[0].train_start:
        raise ValueError("source sequence start does not match walk-forward plan")
    if plan.evaluation_cutoff != plan.folds[-1].test_end:
        raise ValueError("evaluation cutoff must equal final planned complete-fold TEST end")

    results: list[WalkForwardFoldResult] = []
    reference_signatures: tuple[StateSignature, ...] | None = None
    for fold in plan.folds:
        train_model_count = 0
        test_model_count = 0
        skipped_train = fold.train_source_observations
        skipped_test = fold.test_source_observations
        try:
            train_source, test_source = _fold_source_frames(source_rows, fold)
            train_rows, _, skipped_train = _complete_case(train_source, candidate.feature_order)
            train_model_count = int(train_rows.shape[0])
            test_rows, test_timestamps, skipped_test = _complete_case(
                test_source,
                candidate.feature_order,
            )
            test_model_count = int(test_rows.shape[0])
            if train_model_count < profile.walk_forward.minimum_model_train_observations:
                raise ValueError(
                    f"retained TRAIN observations are below pinned minimum 504: {train_model_count}"
                )
            if test_model_count < profile.walk_forward.minimum_model_test_observations:
                raise ValueError(
                    f"retained TEST observations are below pinned minimum 42: {test_model_count}"
                )

            scaler = fit_standard_scaler(train_rows, candidate.feature_order)
            scaled_train = scaler.transform(train_rows)
            scaled_test = scaler.transform(test_rows)
            multistart = run_multistart(
                scaled_train,
                state_count=candidate.state_count,
                adapter_factory=adapter_factory,
            )
            artifact = multistart.winner.artifact
            if artifact.feature_order != candidate.feature_order:
                raise ValueError("fitted model feature order differs from frozen resolved order")
            validate_full_covariances(artifact)

            train_filter = causal_filter(scaled_train, artifact)
            train_occupancy_raw = validate_train_occupancy(train_filter.filtered_probabilities)
            continued = continued_test_predictive_likelihood(
                scaled_train,
                scaled_test,
                artifact,
            )
            test_filter = causal_filter(
                scaled_test,
                artifact,
                initial_filtered_probabilities=train_filter.terminal_probabilities,
            )
            if abs(test_filter.log_likelihood - continued.test_log_likelihood) > 1e-10:
                raise ValueError("continued TEST likelihood/filter evidence disagree")

            if reference_signatures is None:
                alignment = align_first_fold(artifact)
            else:
                alignment = align_to_reference(artifact, reference_signatures)
            aligned_train_occupancy = _aligned_occupancy(train_occupancy_raw, alignment)
            aligned_oos_probabilities = _aligned_probabilities(
                test_filter.filtered_probabilities,
                alignment,
            )
            aligned_oos_occupancy = _aligned_occupancy(
                occupancy(test_filter.filtered_probabilities),
                alignment,
            )
            criteria = information_criteria(
                multistart.winner.train_log_likelihood,
                train_model_count,
                candidate.state_count,
                candidate.feature_dimension,
                candidate.mixture_count,
            )
            result = _valid_fold_result(
                fold=fold,
                train_model_count=train_model_count,
                test_model_count=test_model_count,
                skipped_train=skipped_train,
                skipped_test=skipped_test,
                scaler=scaler,
                multistart=multistart,
                artifact=artifact,
                alignment=alignment,
                criteria=criteria,
                train_occupancy=aligned_train_occupancy,
                oos_occupancy=aligned_oos_occupancy,
                train_log_likelihood=multistart.winner.train_log_likelihood,
                oos_log_likelihood=continued.test_log_likelihood,
                oos_per_observation=continued.test_log_likelihood_per_observation,
                oos_timestamps=test_timestamps,
                oos_probabilities=aligned_oos_probabilities,
            )
            results.append(result)
            reference_signatures = alignment.aligned_signatures
        except Exception as exc:
            results.append(
                _invalid_fold_result(
                    fold,
                    train_model_count=train_model_count,
                    test_model_count=test_model_count,
                    skipped_train=skipped_train,
                    skipped_test=skipped_test,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
            )

    return WalkForwardEvaluation(
        profile_id=profile.profile_id,
        profile_config_version=profile.profile_config_version,
        candidate_id=candidate.candidate_id,
        state_count=candidate.state_count,
        source_build_id=candidate.source_build_id,
        feature_order=candidate.feature_order,
        feature_selection_definition_hash=candidate.feature_selection_definition_hash,
        feature_selection_execution_hash=candidate.feature_selection_execution_hash,
        evaluation_plan_hash=plan.plan_hash,
        evaluation_cutoff=plan.evaluation_cutoff,
        folds=tuple(results),
    )
