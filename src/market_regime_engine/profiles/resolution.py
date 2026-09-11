"""Resolve the single supported Xetra v4 candidate contract."""

from __future__ import annotations

from dataclasses import dataclass

from market_regime_engine.feature_discovery.contracts import FINAL_CANDIDATE_IDS
from market_regime_engine.profiles.config import ModelProfile

EXPECTED_XETRA_CANDIDATE_STATES = (2, 3, 4, 5)


def expected_candidate_ids(profile_config_version: int = 4) -> tuple[str, ...]:
    """Return the immutable v4 candidate universe."""

    if profile_config_version != 4:
        raise ValueError("only Xetra profile configuration version 4 is supported")
    return FINAL_CANDIDATE_IDS


def _require_lower_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ResolvedCandidateProfile:
    """Candidate-local view over one immutable v4 feature-discovery contract."""

    candidate_id: str
    state_count: int
    covariance_type: str
    feature_order: tuple[str, ...]
    feature_dimension: int
    source_build_id: str
    feature_selection_definition_hash: str
    feature_selection_execution_hash: str
    original_feature_universe: tuple[str, ...]
    model_family: str = "gaussian_hmm"
    mixture_count: int = 1
    feature_contract_version: int = 4

    def __post_init__(self) -> None:
        if self.feature_contract_version != 4:
            raise ValueError("only feature-discovery contract version 4 is supported")
        if self.state_count not in EXPECTED_XETRA_CANDIDATE_STATES:
            raise ValueError("resolved Xetra candidate state count must be one of 2, 3, 4, 5")
        expected_id = (
            f"gmm_hmm_k{self.state_count}_m{self.mixture_count}_full"
            if self.model_family == "gmm_hmm"
            else f"{self.model_family}_k{self.state_count}_full"
        )
        if self.candidate_id != expected_id:
            raise ValueError(f"candidate_id must be exactly {expected_id}")
        if self.model_family == "gaussian_hmm" and self.mixture_count != 1:
            raise ValueError("Gaussian HMM candidate must have exactly one mixture")
        if self.model_family == "gmm_hmm" and self.mixture_count != 2:
            raise ValueError("GMM-HMM candidate must have exactly two mixtures")
        if self.model_family == "student_t_hmm" and self.mixture_count != 1:
            raise ValueError("Student-t HMM candidate must have one emission per state")
        if self.model_family not in {"gaussian_hmm", "gmm_hmm", "student_t_hmm"}:
            raise ValueError("candidate model_family is unsupported")
        if self.covariance_type != "full":
            raise ValueError("resolved candidate covariance_type must be full")
        if not self.feature_order or len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("resolved feature_order must be non-empty and duplicate-free")
        if self.feature_dimension != len(self.feature_order):
            raise ValueError("feature_dimension must equal len(feature_order)")
        if not self.source_build_id or self.source_build_id.strip() != self.source_build_id:
            raise ValueError("source_build_id must be a non-empty trimmed string")
        _require_lower_sha256(
            self.feature_selection_definition_hash, "feature_selection_definition_hash"
        )
        _require_lower_sha256(
            self.feature_selection_execution_hash, "feature_selection_execution_hash"
        )
        if not self.original_feature_universe or len(set(self.original_feature_universe)) != len(
            self.original_feature_universe
        ):
            raise ValueError("v4 candidates require a dynamic feature universe")
        universe = set(self.original_feature_universe)
        if any(feature not in universe for feature in self.feature_order):
            raise ValueError("v4 final feature_order must belong to the dynamic universe")


@dataclass(frozen=True, slots=True)
class ResolvedSelectedFeatureProfile:
    """Profile identity plus the frozen v4 feature contract used by every candidate."""

    profile_id: str
    profile_config_version: int
    registered_model: str
    source_build_id: str
    original_feature_universe: tuple[str, ...]
    final_features: tuple[str, ...]
    feature_selection_definition_hash: str
    feature_selection_execution_hash: str
    candidates: tuple[ResolvedCandidateProfile, ...]

    def __post_init__(self) -> None:
        if self.profile_id != "xetra" or self.profile_config_version != 4:
            raise ValueError("resolved public profile must use Xetra v4")
        if self.registered_model != "regime-xetra":
            raise ValueError("resolved Xetra registered model must be regime-xetra")
        if not self.source_build_id or self.source_build_id.strip() != self.source_build_id:
            raise ValueError("source_build_id must be a non-empty trimmed string")
        _require_lower_sha256(
            self.feature_selection_definition_hash, "feature_selection_definition_hash"
        )
        _require_lower_sha256(
            self.feature_selection_execution_hash, "feature_selection_execution_hash"
        )
        if not self.original_feature_universe or len(set(self.original_feature_universe)) != len(
            self.original_feature_universe
        ):
            raise ValueError("v4 resolved profiles require a dynamic feature universe")
        if not self.final_features or len(set(self.final_features)) != len(self.final_features):
            raise ValueError("resolved final features must be non-empty and duplicate-free")
        if any(feature not in self.original_feature_universe for feature in self.final_features):
            raise ValueError("resolved final features must belong to the source universe")
        validate_candidate_comparison_inputs(self.candidates, profile_config_version=4)
        first = self.candidates[0]
        expected_shared = (
            first.source_build_id,
            first.original_feature_universe,
            first.feature_order,
            first.feature_selection_definition_hash,
            first.feature_selection_execution_hash,
        )
        actual_shared = (
            self.source_build_id,
            self.original_feature_universe,
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
        candidate.feature_contract_version,
    )


def validate_candidate_comparison_inputs(
    candidates: tuple[ResolvedCandidateProfile, ...],
    *,
    profile_config_version: int = 4,
) -> None:
    """Fail before comparison unless all configured candidates share one v4 contract."""

    expected_ids = expected_candidate_ids(profile_config_version)
    if tuple(candidate.candidate_id for candidate in candidates) != expected_ids:
        raise ValueError("candidate comparison IDs/order do not match the v4 candidate grid")
    shared = _candidate_shared_contract(candidates[0])
    if any(_candidate_shared_contract(candidate) != shared for candidate in candidates[1:]):
        raise ValueError("all candidates must share the exact v4 feature contract")


def resolve_global_feature_profile(
    profile: ModelProfile,
    *,
    source_build_id: str,
    original_feature_universe: tuple[str, ...],
    final_features: tuple[str, ...],
    feature_discovery_definition_hash: str,
    feature_discovery_execution_hash: str,
) -> ResolvedSelectedFeatureProfile:
    """Bind one v4 global-discovery result to the complete candidate universe."""

    if profile.profile_id != "xetra" or profile.profile_config_version != 4:
        raise ValueError("global feature resolution requires Xetra v4")
    if not source_build_id or source_build_id.strip() != source_build_id:
        raise ValueError("source_build_id must be a non-empty trimmed string")
    if not original_feature_universe or len(set(original_feature_universe)) != len(
        original_feature_universe
    ):
        raise ValueError("v4 feature universe must be non-empty and duplicate-free")
    if not final_features or len(set(final_features)) != len(final_features):
        raise ValueError("v4 final feature_order must be non-empty and duplicate-free")
    if any(feature not in original_feature_universe for feature in final_features):
        raise ValueError("v4 final feature_order must belong to the dynamic universe")
    _require_lower_sha256(feature_discovery_definition_hash, "feature_discovery_definition_hash")
    _require_lower_sha256(feature_discovery_execution_hash, "feature_discovery_execution_hash")

    def candidate(
        candidate_id: str,
        state_count: int,
        model_family: str = "gaussian_hmm",
        mixture_count: int = 1,
    ) -> ResolvedCandidateProfile:
        return ResolvedCandidateProfile(
            candidate_id=candidate_id,
            state_count=state_count,
            covariance_type="full",
            feature_order=final_features,
            feature_dimension=len(final_features),
            source_build_id=source_build_id,
            feature_selection_definition_hash=feature_discovery_definition_hash,
            feature_selection_execution_hash=feature_discovery_execution_hash,
            original_feature_universe=original_feature_universe,
            model_family=model_family,
            mixture_count=mixture_count,
        )

    candidates = (
        *(candidate(f"gaussian_hmm_k{k}_full", k) for k in EXPECTED_XETRA_CANDIDATE_STATES),
        *(
            candidate(f"gmm_hmm_k{k}_m2_full", k, "gmm_hmm", 2)
            for k in EXPECTED_XETRA_CANDIDATE_STATES
        ),
        *(
            candidate(f"student_t_hmm_k{k}_full", k, "student_t_hmm")
            for k in EXPECTED_XETRA_CANDIDATE_STATES
        ),
    )
    return ResolvedSelectedFeatureProfile(
        profile_id="xetra",
        profile_config_version=4,
        registered_model=profile.registered_model,
        source_build_id=source_build_id,
        original_feature_universe=original_feature_universe,
        final_features=final_features,
        feature_selection_definition_hash=feature_discovery_definition_hash,
        feature_selection_execution_hash=feature_discovery_execution_hash,
        candidates=candidates,
    )
