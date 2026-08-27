"""Immutable final-refit production artifact; fold models are never registrable."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from math import isclose, isfinite

from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.preprocessing.scaling import StandardScalerArtifact

_PROBABILITY_TOLERANCE = 1e-10


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field_name} must be timezone-aware UTC")


@dataclass(frozen=True, slots=True)
class ProductionModelArtifact:
    """Complete state required to continue or replay one registered final refit."""

    profile_id: str
    profile_config_version: int
    registered_model: str
    candidate_id: str
    state_count: int
    source_build_id: str
    source_data_sha256: str
    source_schema_version: int
    source_feature_version: int
    data_time_semantics: str
    feature_selection_definition_hash: str
    feature_selection_execution_hash: str
    evaluation_plan_hash: str
    evaluation_cutoff: datetime
    feature_order: tuple[str, ...]
    scaler: StandardScalerArtifact
    hmm: GaussianHMMArtifact
    winning_seed: int
    inference_origin_timestamp: datetime
    trained_through_timestamp: datetime
    terminal_filtered_probabilities: tuple[float, ...]
    retained_observation_count: int
    skipped_incomplete_observation_count: int

    def __post_init__(self) -> None:
        if self.profile_id != "xetra" or self.profile_config_version not in {1, 2}:
            raise ValueError("production artifact requires a supported xetra profile configuration")
        if self.registered_model != "regime-xetra":
            raise ValueError("production artifact registered model must be exactly regime-xetra")
        if self.state_count not in (2, 3, 4, 5):
            raise ValueError("production artifact supports exactly K2/K3/K4/K5")
        expected_candidates = {
            f"gaussian_hmm_k{self.state_count}_full",
            "gmm_hmm_k2_m2_full",
            "gmm_hmm_k3_m2_full",
            "gmm_hmm_k5_m2_full",
        }
        if self.candidate_id not in expected_candidates:
            raise ValueError("production candidate identity is unsupported")
        if self.candidate_id.startswith("gmm_hmm_") and self.hmm.model_family != "gmm_hmm":
            raise ValueError("GMM-HMM production candidate requires a GMM-HMM artifact")
        if not self.source_build_id or not self.source_data_sha256:
            raise ValueError("production source identity cannot be empty")
        if len(self.source_data_sha256) != 64:
            raise ValueError("source_data_sha256 must contain a SHA-256 digest")
        if self.source_schema_version < 1 or self.source_feature_version < 1:
            raise ValueError("source schema/feature versions must be positive")
        if self.data_time_semantics != "current_vintage_observation_day":
            raise ValueError("unsupported production data_time_semantics")
        for field_name in (
            "feature_selection_definition_hash",
            "feature_selection_execution_hash",
            "evaluation_plan_hash",
        ):
            if len(getattr(self, field_name)) != 64:
                raise ValueError(f"{field_name} must contain a SHA-256 digest")
        for field_name in (
            "evaluation_cutoff",
            "inference_origin_timestamp",
            "trained_through_timestamp",
        ):
            _require_utc(getattr(self, field_name), field_name)
        if self.inference_origin_timestamp > self.trained_through_timestamp:
            raise ValueError("inference origin cannot be after trained-through timestamp")
        if self.trained_through_timestamp > self.evaluation_cutoff:
            raise ValueError("trained-through timestamp cannot exceed evaluation cutoff")
        if self.retained_observation_count < 504:
            raise ValueError("production refit requires at least 504 retained observations")
        if self.skipped_incomplete_observation_count < 0:
            raise ValueError("skipped incomplete observation count cannot be negative")
        feature_order_mismatch = (
            self.scaler.feature_order != self.feature_order
            or self.hmm.feature_order != self.feature_order
        )
        if feature_order_mismatch:
            raise ValueError("scaler/HMM feature order must equal frozen production feature order")
        if self.hmm.state_count != self.state_count:
            raise ValueError("HMM state count must equal production state count")
        if len(self.terminal_filtered_probabilities) != self.state_count:
            raise ValueError("terminal filtered probability dimension differs from state count")
        if any(
            not isfinite(value) or value < 0.0 for value in self.terminal_filtered_probabilities
        ):
            raise ValueError("terminal filtered probabilities must be finite and nonnegative")
        if not isclose(
            sum(self.terminal_filtered_probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=_PROBABILITY_TOLERANCE,
        ):
            raise ValueError("terminal filtered probabilities must sum to one within 1e-10")
        if self.winning_seed not in (11, 23, 37, 53, 71, 89, 107, 131):
            raise ValueError("production winning seed must come from the pinned eight-seed set")
