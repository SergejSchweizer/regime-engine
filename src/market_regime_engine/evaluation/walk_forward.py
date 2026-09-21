"""Leak-free walk-forward evaluation over the frozen complete-case HMM clock."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from itertools import pairwise
from math import isfinite
from typing import TYPE_CHECKING, Any, Protocol

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
from market_regime_engine.evaluation.errors import RecoverableEvaluationInvalidity
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardFold, WalkForwardPlan
from market_regime_engine.evaluations.process_parallel import is_pickleable
from market_regime_engine.evaluations.task_frontier import SharedTaskFrontier
from market_regime_engine.inference.filtering import causal_filter
from market_regime_engine.inference.predictive_likelihood import (
    continued_test_predictive_likelihood,
)
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import GaussianHMMAdapter
from market_regime_engine.preprocessing.pca_policy import validate_pca_source_universe
from market_regime_engine.preprocessing.scaling import StandardScalerArtifact
from market_regime_engine.preprocessing.two_stage import (
    PCATwoStageScalerArtifact,
    fit_pca_hmm_scaler,
)
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.runtime.cpu import cpu_worker_count
from market_regime_engine.states.alignment import (
    StateAlignment,
    align_first_fold,
    align_to_reference,
)
from market_regime_engine.states.signatures import StateSignature
from market_regime_engine.training.multistart import (
    MultistartBatchJob,
    MultistartResult,
    run_multistart,
    run_multistart_batch,
)

if TYPE_CHECKING:
    from market_regime_engine.evaluation_runs.hmm_units import HMMSeedCheckpoint

AdapterFactory = Callable[[], GaussianHMMAdapter]
_TIMESTAMP_COLUMN = "timestamp_m1"


class WalkForwardCandidate(Protocol):
    @property
    def candidate_id(self) -> str: ...

    @property
    def model_family(self) -> str: ...

    @property
    def state_count(self) -> int: ...

    @property
    def mixture_count(self) -> int: ...

    @property
    def feature_order(self) -> tuple[str, ...]: ...

    @property
    def feature_dimension(self) -> int: ...

    @property
    def source_build_id(self) -> str: ...

    @property
    def feature_selection_definition_hash(self) -> str: ...

    @property
    def feature_selection_execution_hash(self) -> str: ...

    @property
    def original_feature_universe(self) -> tuple[str, ...]: ...


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
    pca_scaler_artifact: PCATwoStageScalerArtifact | None = None
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
        if self.pca_scaler_artifact is not None and (
            self.scaler_artifact is None
            or self.pca_scaler_artifact.hmm_scaler != self.scaler_artifact
        ):
            raise ValueError("PCA two-stage artifact must contain the fold HMM scaler")
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
                self.pca_scaler_artifact,
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
    alignment_reference_scaler: StandardScalerArtifact | None = None

    def __post_init__(self) -> None:
        if self.profile_id != "xetra" or self.profile_config_version != 4:
            raise ValueError("walk-forward evaluation requires the Xetra v4 profile")
        if self.state_count not in (2, 3, 4, 5):
            raise ValueError("walk-forward evaluation supports exactly K2/K3/K4/K5")
        expected_candidates = {
            f"gaussian_hmm_k{self.state_count}_full",
            "gmm_hmm_k2_m2_full",
            "gmm_hmm_k3_m2_full",
            "gmm_hmm_k4_m2_full",
            "gmm_hmm_k5_m2_full",
            f"student_t_hmm_k{self.state_count}_full",
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


def _validate_candidate_contract(
    candidate: WalkForwardCandidate,
    profile: ModelProfile,
) -> None:
    """Validate generic candidate identity before any source/model work."""

    families = {"gaussian_hmm", "gmm_hmm", "student_t_hmm"}
    if candidate.model_family not in families:
        raise ValueError("resolved candidate model family is unsupported")
    expected_id = (
        f"gmm_hmm_k{candidate.state_count}_m{candidate.mixture_count}_full"
        if candidate.model_family == "gmm_hmm"
        else f"{candidate.model_family}_k{candidate.state_count}_full"
    )
    if candidate.candidate_id != expected_id:
        raise ValueError("resolved candidate identity is inconsistent with family/state/mixture")
    if candidate.state_count not in (2, 3, 4, 5):
        raise ValueError("resolved candidate state count must be one of 2, 3, 4, 5")
    expected_mixture_count = 2 if candidate.model_family == "gmm_hmm" else 1
    if candidate.mixture_count != expected_mixture_count:
        raise ValueError("resolved candidate mixture count is inconsistent with family")
    if (
        not isinstance(candidate.feature_order, tuple)
        or not candidate.feature_order
        or len(set(candidate.feature_order)) != len(candidate.feature_order)
        or any(
            not isinstance(feature, str) or not feature.strip()
            for feature in candidate.feature_order
        )
    ):
        raise ValueError("resolved candidate feature order must be non-empty and duplicate-free")
    if candidate.feature_dimension != len(candidate.feature_order):
        raise ValueError("resolved candidate feature dimension is inconsistent")
    if (
        not isinstance(candidate.source_build_id, str)
        or not candidate.source_build_id.strip() == candidate.source_build_id
    ):
        raise ValueError("resolved candidate source build must be a non-empty trimmed string")
    for field_name in (
        "feature_selection_definition_hash",
        "feature_selection_execution_hash",
    ):
        value = getattr(candidate, field_name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or value != value.lower()
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"resolved candidate {field_name} must be a lowercase SHA-256 digest")
    if getattr(candidate, "feature_contract_version", None) != 4:
        raise ValueError("Xetra v4 walk-forward candidates require feature contract version 4")


def validate_mandatory_pca_feature_universe(
    raw_feature_order: tuple[str, ...] | None,
    original_feature_universe: tuple[str, ...],
) -> tuple[str, ...]:
    if raw_feature_order is None:
        raise ValueError("Xetra v4 evaluation requires a PCA raw feature order")
    if (
        not raw_feature_order
        or len(set(raw_feature_order)) != len(raw_feature_order)
        or any(not name or name.strip() != name for name in raw_feature_order)
    ):
        raise ValueError("PCA raw feature order must be non-empty and duplicate-free")
    validate_pca_source_universe(raw_feature_order, original_feature_universe)
    if not any(name.startswith("pca_pc_") for name in original_feature_universe):
        raise ValueError("Xetra v4 feature universe must contain generated PCA features")
    return raw_feature_order


def _validate_pca_raw_feature_order(
    raw_feature_order: tuple[str, ...] | None,
    candidate_feature_order: tuple[str, ...],
    original_feature_universe: tuple[str, ...],
) -> tuple[str, ...]:
    validated = validate_mandatory_pca_feature_universe(
        raw_feature_order,
        original_feature_universe,
    )
    if any(name not in original_feature_universe for name in candidate_feature_order):
        raise ValueError("PCA candidate feature order contains an unknown feature")
    return validated


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
        raise RecoverableEvaluationInvalidity(
            "selected non-null feature values must be numeric"
        ) from exc
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_order):
        raise RecoverableEvaluationInvalidity(
            "complete-case matrix must preserve exact resolved feature order"
        )
    if not np.all(np.isfinite(matrix)):
        raise RecoverableEvaluationInvalidity("selected non-null feature values must be finite")
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
    pca_scaler: PCATwoStageScalerArtifact | None,
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
        pca_scaler_artifact=pca_scaler,
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


def _fitted_order_values(
    values: tuple[float, ...],
    alignment: StateAlignment,
) -> tuple[float, ...]:
    """Undo a fold-local persistent-state permutation into fitted order."""

    inverse = [0] * len(alignment.persistent_to_fitted)
    for persistent_index, fitted_index in enumerate(alignment.persistent_to_fitted):
        inverse[fitted_index] = persistent_index
    return tuple(values[index] for index in inverse)


def _fitted_order_probabilities(
    values: tuple[tuple[float, ...], ...],
    alignment: StateAlignment,
) -> np.ndarray:
    """Undo a fold-local persistent-state permutation for OOS probabilities."""

    inverse = [0] * len(alignment.persistent_to_fitted)
    for persistent_index, fitted_index in enumerate(alignment.persistent_to_fitted):
        inverse[fitted_index] = persistent_index
    result = np.asarray(values, dtype=np.float64)[:, inverse]
    if not np.all(np.isfinite(result)):
        raise ValueError("fold-local OOS probabilities must be finite")
    return result


def _reconcile_parallel_fold(
    fold_result: WalkForwardFoldResult,
    *,
    reference_signatures: tuple[StateSignature, ...] | None,
    reference_scaler: StandardScalerArtifact | None,
) -> tuple[WalkForwardFoldResult, tuple[StateSignature, ...] | None, StandardScalerArtifact | None]:
    """Reconcile one independent fold into the canonical sequential state space."""

    if not fold_result.valid:
        return fold_result, reference_signatures, reference_scaler
    if fold_result.model_artifact is None or fold_result.scaler_artifact is None:
        raise ValueError("valid parallel fold is missing model/scaler evidence")
    local_alignment = fold_result.alignment
    if local_alignment is None:
        raise ValueError("valid parallel fold is missing local state alignment")
    resolved_reference_scaler = reference_scaler or fold_result.scaler_artifact
    if reference_signatures is None:
        alignment = align_first_fold(
            fold_result.model_artifact,
            fold_result.scaler_artifact,
            resolved_reference_scaler,
        )
    else:
        alignment = align_to_reference(
            fold_result.model_artifact,
            reference_signatures,
            fold_result.scaler_artifact,
            resolved_reference_scaler,
        )
    if fold_result.train_hard_occupancy is None or fold_result.train_soft_occupancy is None:
        raise ValueError("valid parallel fold is missing TRAIN occupancy evidence")
    fitted_train_hard = _fitted_order_values(fold_result.train_hard_occupancy, local_alignment)
    fitted_train_soft = _fitted_order_values(fold_result.train_soft_occupancy, local_alignment)
    fitted_oos = _fitted_order_probabilities(
        fold_result.oos_filtered_probabilities,
        local_alignment,
    )
    aligned_train = _aligned_occupancy(
        OccupancyDiagnostics(hard=fitted_train_hard, soft=fitted_train_soft),
        alignment,
    )
    aligned_oos = _aligned_probabilities(fitted_oos, alignment)
    aligned_oos_occupancy = _aligned_occupancy(occupancy(fitted_oos), alignment)
    return (
        replace(
            fold_result,
            alignment=alignment,
            train_hard_occupancy=aligned_train.hard,
            train_soft_occupancy=aligned_train.soft,
            oos_hard_occupancy=aligned_oos_occupancy.hard,
            oos_soft_occupancy=aligned_oos_occupancy.soft,
            max_state_signature_drift=alignment.max_drift,
            oos_filtered_probabilities=tuple(
                tuple(float(value) for value in row) for row in aligned_oos
            ),
        ),
        alignment.aligned_signatures,
        resolved_reference_scaler,
    )


@dataclass(frozen=True, slots=True)
class _PreparedFrontierFold:
    fold: WalkForwardFold
    train_model_count: int
    test_model_count: int
    skipped_train: int
    skipped_test: int
    train_rows: np.ndarray
    test_rows: np.ndarray
    train_timestamps: tuple[datetime, ...]
    test_timestamps: tuple[datetime, ...]
    pca_scaler: PCATwoStageScalerArtifact
    scaled_train: np.ndarray
    scaled_test: np.ndarray


def _prepare_frontier_fold(
    source_rows: pd.DataFrame,
    fold: WalkForwardFold,
    *,
    profile: ModelProfile,
    candidate: WalkForwardCandidate,
    pca_raw_order: tuple[str, ...],
    pca_variance_threshold: float,
) -> _PreparedFrontierFold:
    train_source, test_source = _fold_source_frames(source_rows, fold)
    train_rows, train_timestamps, skipped_train = _complete_case(train_source, pca_raw_order)
    test_rows, test_timestamps, skipped_test = _complete_case(test_source, pca_raw_order)
    train_model_count = int(train_rows.shape[0])
    test_model_count = int(test_rows.shape[0])
    if train_model_count < profile.walk_forward.minimum_model_train_observations:
        raise RecoverableEvaluationInvalidity(
            f"retained TRAIN observations are below pinned minimum 504: {train_model_count}"
        )
    minimum_test_observations = (
        1
        if hasattr(fold, "test_calendar_month")
        else profile.walk_forward.minimum_model_test_observations
    )
    if test_model_count < minimum_test_observations:
        minimum_label = (
            f"the model-clock minimum {minimum_test_observations}"
            if minimum_test_observations != profile.walk_forward.minimum_model_test_observations
            else f"pinned minimum {minimum_test_observations}"
        )
        raise RecoverableEvaluationInvalidity(
            f"retained TEST observations are below {minimum_label}: {test_model_count}"
        )
    pca_scaler = fit_pca_hmm_scaler(
        train_timestamps,
        train_rows,
        raw_feature_order=pca_raw_order,
        inner_fold_id=fold.fold_id,
        fit_start=fold.train_start,
        fit_end=fold.train_end,
        variance_threshold=pca_variance_threshold,
        component_count=profile.pca.component_count,
        model_feature_order=candidate.feature_order,
    )
    if pca_scaler.model_feature_order != candidate.feature_order:
        raise RecoverableEvaluationInvalidity(
            "fold-local PCA generated feature order differs from candidate order"
        )
    scaled_train = pca_scaler.transform(train_rows)
    scaled_test = pca_scaler.transform(test_rows)
    return _PreparedFrontierFold(
        fold,
        train_model_count,
        test_model_count,
        skipped_train,
        skipped_test,
        train_rows,
        test_rows,
        train_timestamps,
        test_timestamps,
        pca_scaler,
        scaled_train,
        scaled_test,
    )


def _evaluate_prepared_frontier_fold(
    prepared: _PreparedFrontierFold,
    multistart: MultistartResult,
    *,
    candidate: WalkForwardCandidate,
    reference_signatures: tuple[StateSignature, ...] | None,
    reference_scaler: StandardScalerArtifact | None,
) -> tuple[WalkForwardFoldResult, tuple[StateSignature, ...], StandardScalerArtifact]:
    artifact = multistart.winner.artifact
    if artifact.feature_order != candidate.feature_order:
        raise RecoverableEvaluationInvalidity(
            "fitted model feature order differs from frozen resolved order"
        )
    validate_full_covariances(artifact)
    scaler = prepared.pca_scaler.hmm_scaler
    resolved_reference_scaler = reference_scaler or scaler
    train_filter = causal_filter(prepared.scaled_train, artifact)
    fit_train_log_likelihood = multistart.winner.train_log_likelihood
    filter_train_log_likelihood = train_filter.log_likelihood
    parity_tolerance = 1e-10 * max(
        1.0,
        abs(fit_train_log_likelihood),
        abs(filter_train_log_likelihood),
    )
    if abs(fit_train_log_likelihood - filter_train_log_likelihood) > parity_tolerance:
        raise RecoverableEvaluationInvalidity(
            "TRAIN likelihood parity failed: "
            f"fit={fit_train_log_likelihood:.17g}, "
            f"filter={filter_train_log_likelihood:.17g}, "
            f"tolerance={parity_tolerance:.17g}"
        )
    train_occupancy_raw = validate_train_occupancy(train_filter.filtered_probabilities)
    continued = continued_test_predictive_likelihood(
        prepared.scaled_train,
        prepared.scaled_test,
        artifact,
    )
    test_filter = causal_filter(
        prepared.scaled_test,
        artifact,
        initial_filtered_probabilities=train_filter.terminal_probabilities,
    )
    if abs(test_filter.log_likelihood - continued.test_log_likelihood) > 1e-10:
        raise RecoverableEvaluationInvalidity("continued TEST likelihood/filter evidence disagree")
    if reference_signatures is None:
        alignment = align_first_fold(artifact, scaler, resolved_reference_scaler)
    else:
        alignment = align_to_reference(
            artifact,
            reference_signatures,
            scaler,
            resolved_reference_scaler,
        )
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
        filter_train_log_likelihood,
        prepared.train_model_count,
        candidate.state_count,
        candidate.feature_dimension,
        candidate.mixture_count,
        candidate.model_family,
    )
    result = _valid_fold_result(
        fold=prepared.fold,
        train_model_count=prepared.train_model_count,
        test_model_count=prepared.test_model_count,
        skipped_train=prepared.skipped_train,
        skipped_test=prepared.skipped_test,
        scaler=scaler,
        pca_scaler=prepared.pca_scaler,
        multistart=multistart,
        artifact=artifact,
        alignment=alignment,
        criteria=criteria,
        train_occupancy=aligned_train_occupancy,
        oos_occupancy=aligned_oos_occupancy,
        train_log_likelihood=filter_train_log_likelihood,
        oos_log_likelihood=continued.test_log_likelihood,
        oos_per_observation=continued.test_log_likelihood_per_observation,
        oos_timestamps=prepared.test_timestamps,
        oos_probabilities=aligned_oos_probabilities,
    )
    return result, alignment.aligned_signatures, resolved_reference_scaler


def run_walk_forward_candidate(
    source_rows: pd.DataFrame,
    *,
    plan: WalkForwardPlan,
    profile: ModelProfile,
    candidate: WalkForwardCandidate,
    adapter_factory: AdapterFactory,
    max_workers: int | None = None,
    seed_checkpoint_factory: Callable[[str], HMMSeedCheckpoint] | None = None,
    frontier: SharedTaskFrontier[Any, Any] | None = None,
    pca_raw_feature_order: tuple[str, ...] | None = None,
    pca_variance_threshold: float = 0.90,
) -> WalkForwardEvaluation:
    """Evaluate one frozen-feature K candidate without rerunning feature selection.

    Each fold fits PCA exclusively on that fold's complete raw TRAIN rows,
    then fits HMM standardization on the selected raw-plus-PCA TRAIN columns.
    TEST rows are transformed only with those fold-local artifacts. Candidate
    features may be any ordered subset of the complete raw-plus-generated PCA
    universe; omitting the PCA source contract is invalid.
    """

    if profile.profile_id != "xetra" or profile.profile_config_version != 4:
        raise ValueError("walk-forward runner supports only the Xetra v4 profile")
    _validate_candidate_contract(candidate, profile)
    if candidate.model_family == "gaussian_hmm":
        if candidate.state_count not in profile.gaussian_hmm.candidate_states:
            raise ValueError("resolved Gaussian candidate state count is absent from model profile")
    elif candidate.model_family == "gmm_hmm" and (
        candidate.state_count,
        candidate.mixture_count,
    ) not in {(item.state_count, item.mixture_count) for item in profile.gmm_hmms}:
        raise ValueError("resolved GMM-HMM candidate differs from the model profile")
    elif candidate.model_family == "student_t_hmm" and (
        profile.student_t_hmm is None
        or candidate.state_count not in profile.student_t_hmm.candidate_states
    ):
        raise ValueError("resolved Student-t candidate differs from the model profile")
    pca_raw_order = _validate_pca_raw_feature_order(
        pca_raw_feature_order,
        candidate.feature_order,
        candidate.original_feature_universe,
    )
    if plan.evaluation_cutoff is None or not plan.folds:
        raise ValueError("walk-forward plan must contain at least one complete fold")
    source_feature_order = pca_raw_order
    timestamps = _validate_source_rows(source_rows, source_feature_order)
    if timestamps[0] != plan.folds[0].train_start:
        raise ValueError("source sequence start does not match walk-forward plan")
    if plan.evaluation_cutoff != plan.folds[-1].test_end:
        raise ValueError("evaluation cutoff must equal final planned complete-fold TEST end")

    fold_worker_limit = cpu_worker_count(max_workers, task_count=len(plan.folds))
    if (
        frontier is None
        and seed_checkpoint_factory is None
        and fold_worker_limit > 1
        and is_pickleable(adapter_factory)
    ):
        # A direct caller owns one frontier at the process boundary.  The
        # recursive call then flattens every fold's multistart seeds into it;
        # no fold worker can create a child pool of its own.
        with SharedTaskFrontier[Any, Any](fold_worker_limit) as active_frontier:
            return run_walk_forward_candidate(
                source_rows,
                plan=plan,
                profile=profile,
                candidate=candidate,
                adapter_factory=adapter_factory,
                max_workers=fold_worker_limit,
                frontier=active_frontier,
                pca_raw_feature_order=pca_raw_order,
                pca_variance_threshold=pca_variance_threshold,
            )

    results: list[WalkForwardFoldResult] = []
    reference_signatures: tuple[StateSignature, ...] | None = None
    reference_scaler: StandardScalerArtifact | None = None

    if frontier is not None and seed_checkpoint_factory is None:
        prepared: list[_PreparedFrontierFold] = []
        fold_results: dict[str, WalkForwardFoldResult] = {}
        for fold in plan.folds:
            try:
                prepared.append(
                    _prepare_frontier_fold(
                        source_rows,
                        fold,
                        profile=profile,
                        candidate=candidate,
                        pca_raw_order=pca_raw_order,
                        pca_variance_threshold=pca_variance_threshold,
                    )
                )
            except RecoverableEvaluationInvalidity as exc:
                fold_results[fold.fold_id] = _invalid_fold_result(
                    fold,
                    train_model_count=0,
                    test_model_count=0,
                    skipped_train=fold.train_source_observations,
                    skipped_test=fold.test_source_observations,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
        jobs = tuple(
            MultistartBatchJob(
                prepared_fold.fold.fold_id,
                prepared_fold.scaled_train,
                candidate.state_count,
                adapter_factory,
            )
            for prepared_fold in prepared
        )
        multistarts = run_multistart_batch(
            jobs,
            max_workers=max_workers,
            allow_invalid=True,
            frontier=frontier,
        )
        for prepared_fold, multistart in zip(prepared, multistarts, strict=True):
            if multistart is None:
                fold_results[prepared_fold.fold.fold_id] = _invalid_fold_result(
                    prepared_fold.fold,
                    train_model_count=prepared_fold.train_model_count,
                    test_model_count=prepared_fold.test_model_count,
                    skipped_train=prepared_fold.skipped_train,
                    skipped_test=prepared_fold.skipped_test,
                    failure_reason="RecoverableEvaluationInvalidity: multistart gate failed",
                )
                continue
            try:
                result, reference_signatures, reference_scaler = _evaluate_prepared_frontier_fold(
                    prepared_fold,
                    multistart,
                    candidate=candidate,
                    reference_signatures=reference_signatures,
                    reference_scaler=reference_scaler,
                )
                fold_results[prepared_fold.fold.fold_id] = result
            except RecoverableEvaluationInvalidity as exc:
                fold_results[prepared_fold.fold.fold_id] = _invalid_fold_result(
                    prepared_fold.fold,
                    train_model_count=prepared_fold.train_model_count,
                    test_model_count=prepared_fold.test_model_count,
                    skipped_train=prepared_fold.skipped_train,
                    skipped_test=prepared_fold.skipped_test,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
        evaluation = WalkForwardEvaluation(
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
            folds=tuple(fold_results[fold.fold_id] for fold in plan.folds),
            alignment_reference_scaler=reference_scaler,
        )
        return evaluation

    for fold in plan.folds:
        train_model_count = 0
        test_model_count = 0
        skipped_train = fold.train_source_observations
        skipped_test = fold.test_source_observations
        pca_scaler: PCATwoStageScalerArtifact | None = None
        try:
            train_source, test_source = _fold_source_frames(source_rows, fold)
            train_order = pca_raw_order
            train_rows, train_timestamps, skipped_train = _complete_case(train_source, train_order)
            train_model_count = int(train_rows.shape[0])
            test_rows, test_timestamps, skipped_test = _complete_case(
                test_source,
                train_order,
            )
            test_model_count = int(test_rows.shape[0])
            if train_model_count < profile.walk_forward.minimum_model_train_observations:
                raise RecoverableEvaluationInvalidity(
                    f"retained TRAIN observations are below pinned minimum 504: {train_model_count}"
                )
            minimum_test_observations = (
                1
                if hasattr(fold, "test_calendar_month")
                else profile.walk_forward.minimum_model_test_observations
            )
            if test_model_count < minimum_test_observations:
                minimum_label = (
                    f"the model-clock minimum {minimum_test_observations}"
                    if (
                        minimum_test_observations
                        != profile.walk_forward.minimum_model_test_observations
                    )
                    else f"pinned minimum {minimum_test_observations}"
                )
                raise RecoverableEvaluationInvalidity(
                    f"retained TEST observations are below {minimum_label}: {test_model_count}"
                )

            pca_scaler = fit_pca_hmm_scaler(
                train_timestamps,
                train_rows,
                raw_feature_order=pca_raw_order,
                inner_fold_id=fold.fold_id,
                fit_start=fold.train_start,
                fit_end=fold.train_end,
                variance_threshold=pca_variance_threshold,
                component_count=profile.pca.component_count,
                model_feature_order=candidate.feature_order,
            )
            if pca_scaler.model_feature_order != candidate.feature_order:
                raise RecoverableEvaluationInvalidity(
                    "fold-local PCA generated feature order differs from candidate order"
                )
            scaler = pca_scaler.hmm_scaler
            scaled_train = pca_scaler.transform(train_rows)
            scaled_test = pca_scaler.transform(test_rows)
            if reference_scaler is None:
                reference_scaler = scaler
            assert reference_scaler is not None
            checkpoint = (
                None if seed_checkpoint_factory is None else seed_checkpoint_factory(fold.fold_id)
            )
            multistart_kwargs: dict[str, Any] = {}
            if max_workers is not None:
                multistart_kwargs["max_workers"] = max_workers
            if checkpoint is not None:
                multistart_kwargs["checkpoint"] = checkpoint
            if frontier is not None:
                multistart_kwargs["frontier"] = frontier
            multistart = run_multistart(
                scaled_train,
                state_count=candidate.state_count,
                adapter_factory=adapter_factory,
                **multistart_kwargs,
            )
            artifact = multistart.winner.artifact
            if artifact.feature_order != candidate.feature_order:
                raise RecoverableEvaluationInvalidity(
                    "fitted model feature order differs from frozen resolved order"
                )
            validate_full_covariances(artifact)

            train_filter = causal_filter(scaled_train, artifact)
            fit_train_log_likelihood = multistart.winner.train_log_likelihood
            filter_train_log_likelihood = train_filter.log_likelihood
            parity_tolerance = 1e-10 * max(
                1.0,
                abs(fit_train_log_likelihood),
                abs(filter_train_log_likelihood),
            )
            if abs(fit_train_log_likelihood - filter_train_log_likelihood) > parity_tolerance:
                raise RecoverableEvaluationInvalidity(
                    "TRAIN likelihood parity failed: "
                    f"fit={fit_train_log_likelihood:.17g}, "
                    f"filter={filter_train_log_likelihood:.17g}, "
                    f"tolerance={parity_tolerance:.17g}"
                )
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
                raise RecoverableEvaluationInvalidity(
                    "continued TEST likelihood/filter evidence disagree"
                )

            if reference_signatures is None:
                alignment = align_first_fold(artifact, scaler, reference_scaler)
            else:
                alignment = align_to_reference(
                    artifact,
                    reference_signatures,
                    scaler,
                    reference_scaler,
                )
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
                filter_train_log_likelihood,
                train_model_count,
                candidate.state_count,
                candidate.feature_dimension,
                candidate.mixture_count,
                candidate.model_family,
            )
            result = _valid_fold_result(
                fold=fold,
                train_model_count=train_model_count,
                test_model_count=test_model_count,
                skipped_train=skipped_train,
                skipped_test=skipped_test,
                scaler=scaler,
                pca_scaler=pca_scaler,
                multistart=multistart,
                artifact=artifact,
                alignment=alignment,
                criteria=criteria,
                train_occupancy=aligned_train_occupancy,
                oos_occupancy=aligned_oos_occupancy,
                train_log_likelihood=filter_train_log_likelihood,
                oos_log_likelihood=continued.test_log_likelihood,
                oos_per_observation=continued.test_log_likelihood_per_observation,
                oos_timestamps=test_timestamps,
                oos_probabilities=aligned_oos_probabilities,
            )
            results.append(result)
            reference_signatures = alignment.aligned_signatures
        except RecoverableEvaluationInvalidity as exc:
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
        alignment_reference_scaler=reference_scaler,
    )
