"""Resolve one frozen selected-feature contract shared by all Gaussian candidates."""

from __future__ import annotations

from dataclasses import dataclass

from market_regime_engine.feature_selection.contracts import (
    FeatureSelectionPolicy,
    FeatureSelectionResult,
)
from market_regime_engine.profiles.config import ModelProfile

EXPECTED_XETRA_CANDIDATE_STATES = (2, 3, 4)
_PROFILE_CONTRACTS = {1: (48, 8), 2: (45, 7)}


def _require_lower_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ResolvedCandidateProfile:
    """Candidate-local view over one immutable shared feature-selection contract."""

    candidate_id: str
    state_count: int
    covariance_type: str
    feature_order: tuple[str, ...]
    feature_dimension: int
    source_build_id: str
    feature_selection_definition_hash: str
    feature_selection_execution_hash: str
    original_feature_universe: tuple[str, ...]
    preliminary_medoids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.state_count not in EXPECTED_XETRA_CANDIDATE_STATES:
            raise ValueError("resolved Xetra candidate state count must be one of 2, 3, 4")
        expected_id = f"gaussian_hmm_k{self.state_count}_full"
        if self.candidate_id != expected_id:
            raise ValueError(f"candidate_id must be exactly {expected_id}")
        if self.covariance_type != "full":
            raise ValueError("resolved Gaussian candidate covariance_type must be full")
        if not self.feature_order or len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("resolved feature_order must be non-empty and duplicate-free")
        if self.feature_dimension != len(self.feature_order):
            raise ValueError("feature_dimension must equal len(feature_order)")
        if not self.source_build_id or self.source_build_id.strip() != self.source_build_id:
            raise ValueError("source_build_id must be a non-empty trimmed string")
        _require_lower_sha256(
            self.feature_selection_definition_hash,
            "feature_selection_definition_hash",
        )
        _require_lower_sha256(
            self.feature_selection_execution_hash,
            "feature_selection_execution_hash",
        )
        expected_universe, expected_medoids = _PROFILE_CONTRACTS[
            1 if len(self.original_feature_universe) == 48 else 2
        ]
        if (
            len(self.original_feature_universe) != expected_universe
            or len(set(self.original_feature_universe)) != expected_universe
        ):
            raise ValueError(
                "original Xetra feature universe must contain exactly 48 unique features"
            )
        if (
            len(self.preliminary_medoids) != expected_medoids
            or len(set(self.preliminary_medoids)) != expected_medoids
        ):
            raise ValueError("preliminary medoids must contain exactly eight unique features")
        universe = set(self.original_feature_universe)
        if any(feature not in universe for feature in self.preliminary_medoids):
            raise ValueError(
                "every preliminary medoid must belong to the original feature universe"
            )
        medoids = set(self.preliminary_medoids)
        if any(feature not in medoids for feature in self.feature_order):
            raise ValueError("final feature_order must be a subset of preliminary medoids")
        expected_order = tuple(
            feature for feature in self.preliminary_medoids if feature in set(self.feature_order)
        )
        if self.feature_order != expected_order:
            raise ValueError("final feature_order must preserve preliminary-medoid order")


@dataclass(frozen=True, slots=True)
class ResolvedSelectedFeatureProfile:
    """Profile identity plus the frozen feature contract used by every candidate."""

    profile_id: str
    profile_config_version: int
    registered_model: str
    source_build_id: str
    original_feature_universe: tuple[str, ...]
    preliminary_medoids: tuple[str, ...]
    final_features: tuple[str, ...]
    feature_selection_definition_hash: str
    feature_selection_execution_hash: str
    candidates: tuple[ResolvedCandidateProfile, ...]

    def __post_init__(self) -> None:
        if self.profile_id != "xetra" or self.profile_config_version not in _PROFILE_CONTRACTS:
            raise ValueError("resolved public profile must use a supported xetra configuration")
        if self.registered_model != "regime-xetra":
            raise ValueError("resolved Xetra registered model must be regime-xetra")
        if not self.source_build_id or self.source_build_id.strip() != self.source_build_id:
            raise ValueError("source_build_id must be a non-empty trimmed string")
        _require_lower_sha256(
            self.feature_selection_definition_hash,
            "feature_selection_definition_hash",
        )
        _require_lower_sha256(
            self.feature_selection_execution_hash,
            "feature_selection_execution_hash",
        )
        validate_candidate_comparison_inputs(self.candidates)
        first = self.candidates[0]
        expected_shared = (
            first.source_build_id,
            first.original_feature_universe,
            first.preliminary_medoids,
            first.feature_order,
            first.feature_selection_definition_hash,
            first.feature_selection_execution_hash,
        )
        actual_shared = (
            self.source_build_id,
            self.original_feature_universe,
            self.preliminary_medoids,
            self.final_features,
            self.feature_selection_definition_hash,
            self.feature_selection_execution_hash,
        )
        if actual_shared != expected_shared:
            raise ValueError("resolved profile identity must match its shared candidate contract")


def _candidate_shared_contract(candidate: ResolvedCandidateProfile) -> tuple[object, ...]:
    return (
        candidate.feature_order,
        candidate.feature_dimension,
        candidate.source_build_id,
        candidate.feature_selection_definition_hash,
        candidate.feature_selection_execution_hash,
        candidate.original_feature_universe,
        candidate.preliminary_medoids,
    )


def validate_candidate_comparison_inputs(
    candidates: tuple[ResolvedCandidateProfile, ...],
) -> None:
    """Fail before candidate comparison unless K2/K3/K4 share one exact feature contract."""

    if len(candidates) != 3:
        raise ValueError("candidate comparison requires exactly K2, K3, and K4")
    states = tuple(candidate.state_count for candidate in candidates)
    if states != EXPECTED_XETRA_CANDIDATE_STATES:
        raise ValueError("candidate comparison order must be exactly K2, K3, K4")
    expected_ids = tuple(f"gaussian_hmm_k{state_count}_full" for state_count in (2, 3, 4))
    if tuple(candidate.candidate_id for candidate in candidates) != expected_ids:
        raise ValueError("candidate comparison IDs must be exact full-covariance K2/K3/K4 IDs")
    shared = _candidate_shared_contract(candidates[0])
    if any(_candidate_shared_contract(candidate) != shared for candidate in candidates[1:]):
        raise ValueError(
            "K2/K3/K4 candidates must share exact feature order, dimension, source build, "
            "selection hashes, original universe, and preliminary medoids"
        )


def _validate_selection_against_policy(
    policy: FeatureSelectionPolicy,
    selection: FeatureSelectionResult,
) -> None:
    if policy.policy_id != selection.policy_id:
        raise ValueError("feature-selection policy/result policy_id mismatch")
    expected_universe, _ = _PROFILE_CONTRACTS[1 if policy.policy_id.endswith("v1") else 2]
    if len(policy.feature_universe) != expected_universe:
        raise ValueError("Xetra feature-selection policy has an invalid feature-universe size")
    evidence = selection.evidence
    if tuple(block.block_id for block in evidence.block_evidence) != tuple(
        block.block_id for block in policy.blocks
    ):
        raise ValueError("selection block evidence must preserve canonical policy block order")
    for policy_block, evidence_block in zip(policy.blocks, evidence.block_evidence, strict=True):
        if evidence_block.winner not in policy_block.features:
            raise ValueError("each preliminary medoid must belong to its canonical semantic block")
    if evidence.preliminary_medoids != tuple(block.winner for block in evidence.block_evidence):
        raise ValueError("selection preliminary medoids do not match block evidence")
    if selection.final_features != evidence.final_features:
        raise ValueError("selection final features do not match immutable evidence")


def resolve_selected_feature_profile(
    profile: ModelProfile,
    policy: FeatureSelectionPolicy,
    selection: FeatureSelectionResult,
    *,
    source_build_id: str,
) -> ResolvedSelectedFeatureProfile:
    """Bind the frozen first-TRAIN selection to every configured Gaussian candidate."""

    if profile.profile_id != "xetra" or profile.profile_config_version not in _PROFILE_CONTRACTS:
        raise ValueError("only supported public xetra profile configurations are allowed")
    if profile.feature_selection.policy_id != policy.policy_id:
        raise ValueError("model profile feature-selection policy_id mismatch")
    if profile.feature_selection.static_features:
        raise ValueError("selected-feature profile cannot also configure static features")
    if profile.gaussian_hmm.candidate_states != EXPECTED_XETRA_CANDIDATE_STATES:
        raise ValueError("Xetra Gaussian candidate states must be exactly K2/K3/K4")
    if profile.gaussian_hmm.covariance_type != "full":
        raise ValueError("Xetra Gaussian candidates must use full covariance")
    if not source_build_id or source_build_id.strip() != source_build_id:
        raise ValueError("source_build_id must be a non-empty trimmed string")

    _validate_selection_against_policy(policy, selection)
    candidates = tuple(
        ResolvedCandidateProfile(
            candidate_id=f"gaussian_hmm_k{state_count}_full",
            state_count=state_count,
            covariance_type="full",
            feature_order=selection.final_features,
            feature_dimension=len(selection.final_features),
            source_build_id=source_build_id,
            feature_selection_definition_hash=selection.feature_selection_definition_hash,
            feature_selection_execution_hash=selection.feature_selection_execution_hash,
            original_feature_universe=policy.feature_universe,
            preliminary_medoids=selection.evidence.preliminary_medoids,
        )
        for state_count in profile.gaussian_hmm.candidate_states
    )
    validate_candidate_comparison_inputs(candidates)
    return ResolvedSelectedFeatureProfile(
        profile_id=profile.profile_id,
        profile_config_version=profile.profile_config_version,
        registered_model=profile.registered_model,
        source_build_id=source_build_id,
        original_feature_universe=policy.feature_universe,
        preliminary_medoids=selection.evidence.preliminary_medoids,
        final_features=selection.final_features,
        feature_selection_definition_hash=selection.feature_selection_definition_hash,
        feature_selection_execution_hash=selection.feature_selection_execution_hash,
        candidates=candidates,
    )
