"""Mandatory fresh final production refit after statistical champion selection."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from itertools import pairwise

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluation.diagnostics import (
    validate_full_covariances,
    validate_train_occupancy,
)
from market_regime_engine.evaluation.walk_forward import WalkForwardEvaluation
from market_regime_engine.inference.filtering import causal_filter
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.gaussian_hmm import HmmlearnGaussianHMMAdapter
from market_regime_engine.models.production_artifact import ProductionModelArtifact
from market_regime_engine.preprocessing.scaling import fit_standard_scaler
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile
from market_regime_engine.states.alignment import StateAlignment, align_to_reference
from market_regime_engine.training.multistart import AdapterFactory, run_multistart

_TIMESTAMP_COLUMN = "timestamp_m1"
_MINIMUM_RETAINED = 504
AdapterFactoryBuilder = Callable[[ResolvedCandidateProfile], AdapterFactory]


def _require_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    return value


def _default_adapter_builder(candidate: ResolvedCandidateProfile) -> AdapterFactory:
    def factory() -> HmmlearnGaussianHMMAdapter:
        return HmmlearnGaussianHMMAdapter(candidate.feature_order)

    return factory


def _refit_matrix(
    source_rows: pd.DataFrame,
    *,
    feature_order: tuple[str, ...],
    evaluation_cutoff: datetime,
) -> tuple[np.ndarray, tuple[datetime, ...], int]:
    if _TIMESTAMP_COLUMN not in source_rows.columns:
        raise ValueError(f"source rows must contain {_TIMESTAMP_COLUMN}")
    missing = tuple(feature for feature in feature_order if feature not in source_rows.columns)
    if missing:
        raise ValueError(f"source rows are missing frozen features: {', '.join(missing)}")
    timestamps = tuple(
        _require_utc(value, _TIMESTAMP_COLUMN) for value in source_rows[_TIMESTAMP_COLUMN]
    )
    if any(left >= right for left, right in pairwise(timestamps)):
        raise ValueError("source timestamps must be strictly increasing and unique")
    if timestamps and timestamps[-1] < evaluation_cutoff:
        raise ValueError("source snapshot does not reach the evaluation cutoff")
    bounded = source_rows.loc[source_rows[_TIMESTAMP_COLUMN] <= evaluation_cutoff]
    if bounded.empty:
        raise ValueError("final refit has no source rows through evaluation cutoff")
    selected = bounded.loc[:, list(feature_order)]
    complete_mask = selected.notna().all(axis=1)
    complete = selected.loc[complete_mask]
    try:
        matrix = complete.to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("selected non-null feature values must be numeric") from exc
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_order):
        raise ValueError("final-refit matrix must preserve exact frozen feature order")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("selected non-null feature values must be finite")
    retained_timestamps = tuple(
        _require_utc(value, "retained timestamp")
        for value in bounded.loc[complete_mask, _TIMESTAMP_COLUMN]
    )
    if len(retained_timestamps) < _MINIMUM_RETAINED:
        raise ValueError("final production refit requires at least 504 usable observations")
    skipped = len(bounded) - len(retained_timestamps)
    return matrix, retained_timestamps, skipped


def _last_valid_reference(evaluation: WalkForwardEvaluation) -> tuple[tuple[float, ...], ...]:
    valid = evaluation.valid_folds
    if not valid:
        raise ValueError("winning candidate has no valid evaluation fold for final alignment")
    alignment = valid[-1].alignment
    if alignment is None:
        raise ValueError("last valid winning-K fold is missing persistent alignment evidence")
    return alignment.aligned_signatures


def _aligned_artifact(
    artifact: GaussianHMMArtifact,
    alignment: StateAlignment,
) -> GaussianHMMArtifact:
    mapping = alignment.persistent_to_fitted
    return GaussianHMMArtifact(
        state_count=artifact.state_count,
        feature_order=artifact.feature_order,
        start_probabilities=tuple(artifact.start_probabilities[index] for index in mapping),
        transition_matrix=tuple(
            tuple(artifact.transition_matrix[row][column] for column in mapping)
            for row in mapping
        ),
        means=tuple(artifact.means[index] for index in mapping),
        full_covariances=tuple(artifact.full_covariances[index] for index in mapping),
    )


def final_production_refit(
    source_rows: pd.DataFrame,
    *,
    lineage: SourceLineage,
    candidate: ResolvedCandidateProfile,
    winning_evaluation: WalkForwardEvaluation,
    adapter_factory_builder: AdapterFactoryBuilder = _default_adapter_builder,
) -> ProductionModelArtifact:
    """Fit one fresh production artifact without rerunning selection or candidate ranking."""

    if candidate.candidate_id != winning_evaluation.candidate_id:
        raise ValueError("final-refit candidate must equal the statistical champion evaluation")
    if candidate.state_count != winning_evaluation.state_count:
        raise ValueError("final-refit candidate K differs from champion evaluation")
    if candidate.feature_order != winning_evaluation.feature_order:
        raise ValueError("final-refit features differ from frozen champion evaluation")
    if candidate.source_build_id != winning_evaluation.source_build_id:
        raise ValueError("final-refit candidate and evaluation source builds differ")
    if lineage.source_build_id != winning_evaluation.source_build_id:
        raise ValueError("final-refit source lineage differs from evaluation source build")
    if (
        candidate.feature_selection_definition_hash
        != winning_evaluation.feature_selection_definition_hash
        or candidate.feature_selection_execution_hash
        != winning_evaluation.feature_selection_execution_hash
    ):
        raise ValueError("final-refit selection hashes differ from champion evaluation")

    cutoff = _require_utc(winning_evaluation.evaluation_cutoff, "evaluation_cutoff")
    matrix, retained_timestamps, skipped = _refit_matrix(
        source_rows,
        feature_order=candidate.feature_order,
        evaluation_cutoff=cutoff,
    )
    scaler = fit_standard_scaler(matrix, candidate.feature_order)
    scaled = scaler.transform(matrix)
    multistart = run_multistart(
        scaled,
        state_count=candidate.state_count,
        adapter_factory=adapter_factory_builder(candidate),
    )
    raw_artifact = multistart.winner.artifact
    validate_full_covariances(raw_artifact)
    filtered = causal_filter(scaled, raw_artifact)
    validate_train_occupancy(filtered.filtered_probabilities)
    alignment = align_to_reference(raw_artifact, _last_valid_reference(winning_evaluation))
    production_hmm = _aligned_artifact(raw_artifact, alignment)
    terminal = tuple(
        filtered.terminal_probabilities[index] for index in alignment.persistent_to_fitted
    )
    return ProductionModelArtifact(
        profile_id=winning_evaluation.profile_id,
        profile_config_version=winning_evaluation.profile_config_version,
        registered_model="regime-xetra",
        candidate_id=candidate.candidate_id,
        state_count=candidate.state_count,
        source_build_id=lineage.source_build_id,
        source_data_sha256=lineage.data_sha256,
        source_schema_version=lineage.schema_version,
        source_feature_version=lineage.feature_version,
        data_time_semantics=lineage.data_time_semantics,
        feature_selection_definition_hash=winning_evaluation.feature_selection_definition_hash,
        feature_selection_execution_hash=winning_evaluation.feature_selection_execution_hash,
        evaluation_plan_hash=winning_evaluation.evaluation_plan_hash,
        evaluation_cutoff=cutoff,
        feature_order=candidate.feature_order,
        scaler=scaler,
        hmm=production_hmm,
        winning_seed=multistart.winner.seed,
        inference_origin_timestamp=retained_timestamps[0],
        trained_through_timestamp=retained_timestamps[-1],
        terminal_filtered_probabilities=terminal,
        retained_observation_count=len(retained_timestamps),
        skipped_incomplete_observation_count=skipped,
    )
