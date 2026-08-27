"""Strict, versioned model-profile configuration contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isclose
from typing import Any


@dataclass(frozen=True, slots=True)
class FeatureSelectionConfig:
    policy_id: str | None
    static_features: tuple[str, ...]
    within_block_method: str
    cross_block_method: str
    minimum_feature_coverage: float
    minimum_nonzero_variance: float
    minimum_block_complete_observations: int
    maximum_cross_block_abs_spearman: float
    numeric_tie_abs_tolerance: float

    def __post_init__(self) -> None:
        if (self.policy_id is None) == (len(self.static_features) == 0):
            raise ValueError("exactly one of policy_id or static_features must be configured")
        if len(set(self.static_features)) != len(self.static_features):
            raise ValueError("static_features contains duplicates")
        if self.within_block_method != "absolute_spearman_medoid":
            raise ValueError("unsupported within-block feature-selection method")
        if self.cross_block_method != "absolute_spearman_prune":
            raise ValueError("unsupported cross-block feature-selection method")
        if not 0.0 < self.minimum_feature_coverage <= 1.0:
            raise ValueError("minimum_feature_coverage must be in (0, 1]")
        if self.minimum_nonzero_variance <= 0.0:
            raise ValueError("minimum_nonzero_variance must be positive")
        if self.minimum_block_complete_observations < 1:
            raise ValueError("minimum_block_complete_observations must be positive")
        if not 0.0 < self.maximum_cross_block_abs_spearman < 1.0:
            raise ValueError("maximum_cross_block_abs_spearman must be in (0, 1)")
        if self.numeric_tie_abs_tolerance <= 0.0:
            raise ValueError("numeric_tie_abs_tolerance must be positive")


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    minimum_train_source_observations: int
    test_source_observations: int
    step_source_observations: int
    allow_partial_final_test: bool
    minimum_model_train_observations: int
    minimum_model_test_observations: int
    ranking_abs_tolerance: float

    def __post_init__(self) -> None:
        counts = (
            self.minimum_train_source_observations,
            self.test_source_observations,
            self.step_source_observations,
            self.minimum_model_train_observations,
            self.minimum_model_test_observations,
        )
        if any(value < 1 for value in counts):
            raise ValueError("walk-forward observation counts must be positive")
        if self.ranking_abs_tolerance <= 0.0:
            raise ValueError("ranking_abs_tolerance must be positive")


@dataclass(frozen=True, slots=True)
class GaussianHMMConfig:
    candidate_states: tuple[int, ...]
    backend: str
    covariance_type: str
    implementation: str
    seeds: tuple[int, ...]
    minimum_valid_starts: int
    minimum_multistart_success_rate: float
    n_iter: int
    tol: float
    min_covar: float
    startprob_prior: float
    transmat_prior: float
    means_prior: float
    means_weight: float
    covars_prior: float
    covars_weight: float
    params: str
    init_params: str

    def __post_init__(self) -> None:
        if self.candidate_states != (2, 3, 4, 5):
            raise ValueError("Gaussian MVP candidates must be exactly K=2,3,4,5")
        if self.backend != "hmmlearn==0.3.3":
            raise ValueError("unsupported Gaussian HMM backend")
        if self.covariance_type != "full":
            raise ValueError("Gaussian HMM covariance_type must be exactly full")
        if self.implementation != "log":
            raise ValueError("Gaussian HMM implementation must be log")
        if len(self.seeds) != len(set(self.seeds)) or not self.seeds:
            raise ValueError("multistart seeds must be unique and non-empty")
        if not 1 <= self.minimum_valid_starts <= len(self.seeds):
            raise ValueError("minimum_valid_starts is inconsistent with seeds")
        if not 0.0 < self.minimum_multistart_success_rate <= 1.0:
            raise ValueError("minimum_multistart_success_rate must be in (0,1]")
        if self.n_iter < 1 or self.tol <= 0.0 or self.min_covar <= 0.0:
            raise ValueError("HMM iteration/tolerance/covariance settings must be positive")
        if self.params != "stmc" or self.init_params != "stmc":
            raise ValueError("Gaussian HMM params/init_params must be stmc")


@dataclass(frozen=True, slots=True)
class GMMHMMConfig:
    state_count: int
    mixture_count: int
    backend: str
    covariance_type: str
    implementation: str

    def __post_init__(self) -> None:
        if self.state_count not in (2, 5) or self.mixture_count != 2:
            raise ValueError("GMM-HMM comparison candidate must be K=2 or K=5 with two mixtures")
        if self.backend != "hmmlearn==0.3.3":
            raise ValueError("unsupported GMM-HMM backend")
        if self.covariance_type != "full" or self.implementation != "log":
            raise ValueError("GMM-HMM requires full covariance and log implementation")


@dataclass(frozen=True, slots=True)
class EvaluationGates:
    minimum_train_hard_occupancy: float
    minimum_train_soft_occupancy: float
    candidate_minimum_valid_fold_rate: float
    low_confidence_threshold: float
    state_alignment_ambiguity_abs_tolerance: float
    covariance_asymmetry_abs_tolerance: float
    probability_normalization_abs_tolerance: float
    minimum_covariance_diagonal_variance: float

    def __post_init__(self) -> None:
        rates = (
            self.minimum_train_hard_occupancy,
            self.minimum_train_soft_occupancy,
            self.candidate_minimum_valid_fold_rate,
            self.low_confidence_threshold,
        )
        if any(not 0.0 < value <= 1.0 for value in rates):
            raise ValueError("gate rates must be in (0,1]")
        tolerances = (
            self.state_alignment_ambiguity_abs_tolerance,
            self.covariance_asymmetry_abs_tolerance,
            self.probability_normalization_abs_tolerance,
            self.minimum_covariance_diagonal_variance,
        )
        if any(value <= 0.0 for value in tolerances):
            raise ValueError("numerical gate tolerances must be positive")


@dataclass(frozen=True, slots=True)
class ModelProfile:
    profile_id: str
    profile_config_version: int
    registered_model: str
    production_alias: str
    challenger_alias: str
    feature_selection: FeatureSelectionConfig
    walk_forward: WalkForwardConfig
    gaussian_hmm: GaussianHMMConfig
    gates: EvaluationGates
    gmm_hmms: tuple[GMMHMMConfig, ...] = ()

    def __post_init__(self) -> None:
        identity_fields = (
            "profile_id",
            "registered_model",
            "production_alias",
            "challenger_alias",
        )
        for field_name in identity_fields:
            value = getattr(self, field_name)
            if not value or value.strip() != value:
                raise ValueError(f"{field_name} must be a non-empty trimmed string")
        if self.profile_config_version < 1:
            raise ValueError("profile_config_version must be positive")
        if self.production_alias != "champion" or self.challenger_alias != "challenger":
            raise ValueError("registry aliases must be champion/challenger")
        gmm_identities = tuple(
            (candidate.state_count, candidate.mixture_count) for candidate in self.gmm_hmms
        )
        if len(set(gmm_identities)) != len(gmm_identities):
            raise ValueError("GMM-HMM candidates must be unique by state and mixture count")

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def profile_hash(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return sha256(payload).hexdigest()


def assert_xetra_v1_pins(profile: ModelProfile) -> None:
    """Fail closed unless every evaluation-contract constant is exactly pinned."""
    fs = profile.feature_selection
    wf = profile.walk_forward
    hmm = profile.gaussian_hmm
    gates = profile.gates
    exact = {
        "profile_id": profile.profile_id == "xetra",
        "profile_config_version": profile.profile_config_version == 1,
        "registered_model": profile.registered_model == "regime-xetra",
        "policy_id": fs.policy_id == "xetra_semantic_medoid_v1",
        "coverage": fs.minimum_feature_coverage == 0.90,
        "variance": fs.minimum_nonzero_variance == 1e-12,
        "block_rows": fs.minimum_block_complete_observations == 504,
        "cross_block": fs.maximum_cross_block_abs_spearman == 0.85,
        "feature_tie": fs.numeric_tie_abs_tolerance == 1e-12,
        "min_source_train": wf.minimum_train_source_observations == 1260,
        "test_rows": wf.test_source_observations == 63,
        "step_rows": wf.step_source_observations == 63,
        "no_partial": wf.allow_partial_final_test is False,
        "model_train": wf.minimum_model_train_observations == 504,
        "model_test": wf.minimum_model_test_observations == 42,
        "ranking_tie": wf.ranking_abs_tolerance == 1e-12,
        "seeds": hmm.seeds == (11, 23, 37, 53, 71, 89, 107, 131),
        "valid_starts": hmm.minimum_valid_starts == 6,
        "success_rate": hmm.minimum_multistart_success_rate == 0.75,
        "n_iter": hmm.n_iter == 1000,
        "tol": hmm.tol == 1e-4,
        "min_covar": hmm.min_covar == 1e-6,
        "startprob_prior": hmm.startprob_prior == 1.0,
        "transmat_prior": hmm.transmat_prior == 1.0,
        "means_prior": hmm.means_prior == 0.0,
        "means_weight": hmm.means_weight == 0.0,
        "covars_prior": hmm.covars_prior == 0.01,
        "covars_weight": hmm.covars_weight == 1.0,
        "hard_occ": gates.minimum_train_hard_occupancy == 0.03,
        "soft_occ": gates.minimum_train_soft_occupancy == 0.05,
        "valid_fold_rate": gates.candidate_minimum_valid_fold_rate == 0.80,
        "confidence": gates.low_confidence_threshold == 0.60,
        "alignment": gates.state_alignment_ambiguity_abs_tolerance == 1e-10,
        "asymmetry": gates.covariance_asymmetry_abs_tolerance == 1e-10,
        "normalization": gates.probability_normalization_abs_tolerance == 1e-10,
        "min_variance": gates.minimum_covariance_diagonal_variance == 1e-12,
    }
    failures = [name for name, matches in exact.items() if not matches]
    if failures:
        raise ValueError(f"xetra v1 profile differs from pinned evaluation contract: {failures}")
    if not isclose(hmm.minimum_valid_starts / len(hmm.seeds), 0.75, abs_tol=0.0):
        raise ValueError("xetra v1 multistart minimum is not exactly 6/8")
