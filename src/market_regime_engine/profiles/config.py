"""Strict, versioned model-profile configuration contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isclose
from typing import Any

from market_regime_engine.feature_discovery.contracts import (
    CLUSTER_COUNT_MAX,
    CLUSTER_COUNT_MIN,
    CLUSTER_COUNT_TIE_BREAK,
    FEATURE_SCORE_BIN_COUNT,
    FEATURE_SCORE_TIE_TOLERANCE,
    FINAL_CANDIDATE_IDS,
    INNER_ALLOW_PARTIAL_FINAL_TEST,
    INNER_STEP_SOURCE_OBSERVATIONS,
    INNER_TEST_SOURCE_OBSERVATIONS,
    INNER_TRAIN_SOURCE_OBSERVATIONS,
    MAX_PREFIX_LENGTH,
    MIN_ELIGIBLE_FEATURES,
    MIN_FEATURE_COVERAGE,
    MIN_FEATURE_SCORE_COVERAGE,
    MIN_FEATURE_SCORE_OBSERVATIONS,
    MIN_FEATURE_VARIANCE,
    MIN_MODEL_CLOCK_VALID_FOLD_RATE,
    MIN_MODEL_TEST_OBSERVATIONS,
    MIN_MODEL_TRAIN_OBSERVATIONS,
    MIN_OUTER_VALID_FOLD_RATE,
    MIN_OUTER_VALID_FOLDS,
    MIN_PAIRWISE_OBSERVATIONS,
    MIN_PREFIX_LENGTH,
    MIN_SILHOUETTE,
    MIN_TEACHER_SHARED_SUPPORT,
    OUTER_ALLOW_PARTIAL_FINAL_TEST,
    OUTER_STEP_SOURCE_OBSERVATIONS,
    OUTER_TEST_SOURCE_OBSERVATIONS,
    OUTER_TRAIN_SOURCE_OBSERVATIONS,
    POPULATION_VARIANCE_DDOF,
    PREFIX_NMI_TIE_TOLERANCE,
    RHO_CLIP_TOLERANCE,
    SILHOUETTE_SINGLETON_VALUE,
    SILHOUETTE_TIE_TOLERANCE,
    V4_CLUSTER_ID_ORDERING,
    V4_CLUSTERING_METHOD,
    V4_CROSS_L_LIKELIHOOD_FORBIDDEN,
    V4_CROSS_L_TIE_BREAKS,
    V4_DEPLOYMENT_SELECTION_SCOPE,
    V4_DISTANCE_METHOD,
    V4_EVALUATION_ID,
    V4_EXCLUDED_SOURCE_COLUMN,
    V4_FEATURE_DISCOVERY_POLICY,
    V4_FEATURE_ORDERING,
    V4_FEATURE_REGIME_SCORE,
    V4_FEATURE_SCORE_DIAGNOSTIC,
    V4_FEATURE_UNIVERSE_MODE,
    V4_GAUSSIAN_PARAMETER_BOUND,
    V4_MISSING_VALUE_POLICY,
    V4_OUTER_STATE_IDENTITY,
    V4_PREFIX_LIKELIHOOD_SCOPE,
    V4_PREFIX_MODEL_FAMILY,
    V4_PREFIX_SELECTION_TARGET,
    V4_PREFIX_STATE_COUNTS,
    V4_PRODUCTION_STATE_IDENTITY,
    V4_PROVISIONAL_MODEL_FAMILY,
    V4_PROVISIONAL_STATE_COUNTS,
    V4_REDUNDANCY_MEASURE,
    V4_SCORE_TIE_ORDER,
    V4_SOURCE_NAN_INF_POLICY,
    V4_TEMPORARY_PROTOTYPE_METHOD,
)


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
        if (
            not self.candidate_states
            or any(state_count < 2 for state_count in self.candidate_states)
            or tuple(sorted(self.candidate_states)) != self.candidate_states
            or len(set(self.candidate_states)) != len(self.candidate_states)
        ):
            raise ValueError(
                "Gaussian candidate_states must be unique ascending integers with K >= 2"
            )
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
        if self.state_count not in (2, 3, 4, 5) or self.mixture_count != 2:
            raise ValueError(
                "GMM-HMM comparison candidate must be K=2, K=3, K=4, or K=5 with two mixtures"
            )
        if self.backend != "hmmlearn==0.3.3":
            raise ValueError("unsupported GMM-HMM backend")
        if self.covariance_type != "full" or self.implementation != "log":
            raise ValueError("GMM-HMM requires full covariance and log implementation")


@dataclass(frozen=True, slots=True)
class StudentTHMMConfig:
    candidate_states: tuple[int, ...]
    backend: str
    covariance_type: str
    minimum_nu: float
    maximum_nu: float
    initial_nu: float
    n_iter: int
    tol: float
    min_covar: float

    def __post_init__(self) -> None:
        if self.candidate_states != (2, 3, 4, 5):
            raise ValueError("Student-t candidates must be exactly K=2,3,4,5")
        if self.backend != "native-baum-welch-v1" or self.covariance_type != "full":
            raise ValueError("unsupported Student-t HMM backend or covariance type")
        if not 2.0 < self.minimum_nu < self.initial_nu < self.maximum_nu:
            raise ValueError("Student-t degrees-of-freedom bounds are inconsistent")
        if self.n_iter < 1 or self.tol <= 0.0 or self.min_covar <= 0.0:
            raise ValueError("Student-t optimizer settings must be positive")


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
class FeatureDiscoveryConfig:
    """Explicit, semantic-free configuration for global discovery v4."""

    evaluation_id: str
    policy_id: str
    feature_universe_mode: str
    excluded_source_column: str
    feature_ordering: str
    minimum_feature_coverage: float
    population_variance_ddof: int
    minimum_feature_variance: float
    minimum_eligible_features: int
    minimum_pairwise_observations: int
    redundancy_measure: str
    distance_method: str
    clustering_method: str
    rho_clip_tolerance: float
    cluster_count_min: int
    cluster_count_max: int
    silhouette_singleton_value: float
    silhouette_tie_tolerance: float
    minimum_silhouette: float
    cluster_count_tie_break: str
    temporary_prototype_method: str
    cluster_id_ordering: str
    provisional_model_family: str
    provisional_state_counts: tuple[int, ...]
    gaussian_parameter_bound: str
    feature_score_coverage: float
    feature_score_observations: int
    feature_score_bin_count: int
    feature_regime_score: str
    feature_score_diagnostic: str
    feature_score_tie_tolerance: float
    score_tie_order: tuple[str, ...]
    minimum_prefix_length: int
    maximum_prefix_length: int
    prefix_model_family: str
    prefix_state_counts: tuple[int, ...]
    prefix_selection_target: str
    minimum_teacher_shared_support: float
    prefix_nmi_tie_tolerance: float
    cross_l_tie_breaks: tuple[str, ...]
    cross_l_likelihood_forbidden: bool
    prefix_likelihood_scope: str
    inner_train_source_observations: int
    inner_test_source_observations: int
    inner_step_source_observations: int
    inner_partial_final_test: bool
    minimum_model_train_observations: int
    minimum_model_test_observations: int
    minimum_model_clock_valid_fold_rate: float
    outer_train_source_observations: int
    outer_test_source_observations: int
    outer_step_source_observations: int
    outer_partial_final_test: bool
    minimum_outer_valid_fold_rate: float
    minimum_outer_valid_folds: int
    outer_state_identity: str
    production_state_identity: str
    deployment_selection_scope: str
    source_nan_inf_policy: str
    missing_value_policy: str
    final_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        string_fields = (
            "evaluation_id",
            "policy_id",
            "feature_universe_mode",
            "excluded_source_column",
            "feature_ordering",
            "redundancy_measure",
            "distance_method",
            "clustering_method",
            "cluster_count_tie_break",
            "temporary_prototype_method",
            "cluster_id_ordering",
            "provisional_model_family",
            "gaussian_parameter_bound",
            "feature_regime_score",
            "feature_score_diagnostic",
            "prefix_model_family",
            "prefix_selection_target",
            "prefix_likelihood_scope",
            "outer_state_identity",
            "production_state_identity",
            "deployment_selection_scope",
            "source_nan_inf_policy",
            "missing_value_policy",
        )
        for field_name in string_fields:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{field_name} must be a non-empty trimmed string")
        expected = {
            "evaluation_id": V4_EVALUATION_ID,
            "policy_id": V4_FEATURE_DISCOVERY_POLICY,
            "feature_universe_mode": V4_FEATURE_UNIVERSE_MODE,
            "excluded_source_column": V4_EXCLUDED_SOURCE_COLUMN,
            "feature_ordering": V4_FEATURE_ORDERING,
            "minimum_feature_coverage": MIN_FEATURE_COVERAGE,
            "population_variance_ddof": POPULATION_VARIANCE_DDOF,
            "minimum_feature_variance": MIN_FEATURE_VARIANCE,
            "minimum_eligible_features": MIN_ELIGIBLE_FEATURES,
            "minimum_pairwise_observations": MIN_PAIRWISE_OBSERVATIONS,
            "redundancy_measure": V4_REDUNDANCY_MEASURE,
            "distance_method": V4_DISTANCE_METHOD,
            "clustering_method": V4_CLUSTERING_METHOD,
            "rho_clip_tolerance": RHO_CLIP_TOLERANCE,
            "cluster_count_min": CLUSTER_COUNT_MIN,
            "cluster_count_max": CLUSTER_COUNT_MAX,
            "silhouette_singleton_value": SILHOUETTE_SINGLETON_VALUE,
            "silhouette_tie_tolerance": SILHOUETTE_TIE_TOLERANCE,
            "minimum_silhouette": MIN_SILHOUETTE,
            "cluster_count_tie_break": CLUSTER_COUNT_TIE_BREAK,
            "temporary_prototype_method": V4_TEMPORARY_PROTOTYPE_METHOD,
            "cluster_id_ordering": V4_CLUSTER_ID_ORDERING,
            "provisional_model_family": V4_PROVISIONAL_MODEL_FAMILY,
            "provisional_state_counts": V4_PROVISIONAL_STATE_COUNTS,
            "gaussian_parameter_bound": V4_GAUSSIAN_PARAMETER_BOUND,
            "feature_score_coverage": MIN_FEATURE_SCORE_COVERAGE,
            "feature_score_observations": MIN_FEATURE_SCORE_OBSERVATIONS,
            "feature_score_bin_count": FEATURE_SCORE_BIN_COUNT,
            "feature_regime_score": V4_FEATURE_REGIME_SCORE,
            "feature_score_diagnostic": V4_FEATURE_SCORE_DIAGNOSTIC,
            "feature_score_tie_tolerance": FEATURE_SCORE_TIE_TOLERANCE,
            "score_tie_order": V4_SCORE_TIE_ORDER,
            "minimum_prefix_length": MIN_PREFIX_LENGTH,
            "maximum_prefix_length": MAX_PREFIX_LENGTH,
            "prefix_model_family": V4_PREFIX_MODEL_FAMILY,
            "prefix_state_counts": V4_PREFIX_STATE_COUNTS,
            "prefix_selection_target": V4_PREFIX_SELECTION_TARGET,
            "minimum_teacher_shared_support": MIN_TEACHER_SHARED_SUPPORT,
            "prefix_nmi_tie_tolerance": PREFIX_NMI_TIE_TOLERANCE,
            "cross_l_tie_breaks": V4_CROSS_L_TIE_BREAKS,
            "cross_l_likelihood_forbidden": V4_CROSS_L_LIKELIHOOD_FORBIDDEN,
            "prefix_likelihood_scope": V4_PREFIX_LIKELIHOOD_SCOPE,
            "inner_train_source_observations": INNER_TRAIN_SOURCE_OBSERVATIONS,
            "inner_test_source_observations": INNER_TEST_SOURCE_OBSERVATIONS,
            "inner_step_source_observations": INNER_STEP_SOURCE_OBSERVATIONS,
            "inner_partial_final_test": INNER_ALLOW_PARTIAL_FINAL_TEST,
            "minimum_model_train_observations": MIN_MODEL_TRAIN_OBSERVATIONS,
            "minimum_model_test_observations": MIN_MODEL_TEST_OBSERVATIONS,
            "minimum_model_clock_valid_fold_rate": MIN_MODEL_CLOCK_VALID_FOLD_RATE,
            "outer_train_source_observations": OUTER_TRAIN_SOURCE_OBSERVATIONS,
            "outer_test_source_observations": OUTER_TEST_SOURCE_OBSERVATIONS,
            "outer_step_source_observations": OUTER_STEP_SOURCE_OBSERVATIONS,
            "outer_partial_final_test": OUTER_ALLOW_PARTIAL_FINAL_TEST,
            "minimum_outer_valid_fold_rate": MIN_OUTER_VALID_FOLD_RATE,
            "minimum_outer_valid_folds": MIN_OUTER_VALID_FOLDS,
            "outer_state_identity": V4_OUTER_STATE_IDENTITY,
            "production_state_identity": V4_PRODUCTION_STATE_IDENTITY,
            "deployment_selection_scope": V4_DEPLOYMENT_SELECTION_SCOPE,
            "source_nan_inf_policy": V4_SOURCE_NAN_INF_POLICY,
            "missing_value_policy": V4_MISSING_VALUE_POLICY,
            "final_candidate_ids": FINAL_CANDIDATE_IDS,
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(
                    f"v4 discovery field {field_name} differs from the pinned contract"
                )
        if not self.final_candidate_ids:
            raise ValueError("v4 discovery requires a non-empty final candidate universe")


@dataclass(frozen=True, slots=True)
class ModelProfile:
    profile_id: str
    profile_config_version: int
    registered_model: str
    production_alias: str
    challenger_alias: str
    walk_forward: WalkForwardConfig
    gaussian_hmm: GaussianHMMConfig
    gates: EvaluationGates
    feature_selection: FeatureSelectionConfig | None = None
    feature_discovery: FeatureDiscoveryConfig | None = None
    gmm_hmms: tuple[GMMHMMConfig, ...] = ()
    student_t_hmm: StudentTHMMConfig | None = None

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
        if (self.feature_selection is None) == (self.feature_discovery is None):
            raise ValueError(
                "configure exactly one of legacy feature_selection or feature_discovery"
            )
        gmm_identities = tuple(
            (candidate.state_count, candidate.mixture_count) for candidate in self.gmm_hmms
        )
        if len(set(gmm_identities)) != len(gmm_identities):
            raise ValueError("GMM-HMM candidates must be unique by state and mixture count")
        if self.profile_config_version == 1:
            if self.feature_selection is None:
                raise ValueError("xetra v1 requires legacy feature_selection")
            if self.gaussian_hmm.candidate_states != (2, 3, 4):
                raise ValueError("xetra v1 Gaussian candidates must be exactly K=2,3,4")
            if self.gmm_hmms or self.student_t_hmm is not None:
                raise ValueError("xetra v1 supports only its immutable Gaussian candidate set")
        elif self.profile_config_version in (2, 3):
            if self.feature_selection is None:
                raise ValueError("xetra v2/v3 require legacy feature_selection")
            expected_policy = f"xetra_semantic_medoid_v{self.profile_config_version}"
            if self.feature_selection.policy_id != expected_policy:
                raise ValueError("Xetra profile version and feature-selection policy differ")
            if self.gaussian_hmm.candidate_states != (2, 3, 4, 5):
                raise ValueError("Xetra v2/v3 Gaussian candidates must be exactly K=2,3,4,5")
            expected_gmm = ((2, 2), (3, 2), (4, 2), (5, 2))
            if gmm_identities != expected_gmm:
                raise ValueError("Xetra v2/v3 GMM candidates must be exactly K2-K5 with M=2")
            if self.student_t_hmm is None:
                raise ValueError("Xetra v2/v3 requires the Student-t K2-K5 candidate family")
        elif self.profile_config_version == 4:
            if self.feature_discovery is None:
                raise ValueError("xetra v4 requires dedicated feature_discovery")
            if self.gaussian_hmm.candidate_states != (2, 3, 4, 5):
                raise ValueError("xetra v4 Gaussian candidates must be exactly K=2,3,4,5")
            expected_gmm = ((2, 2), (3, 2), (4, 2), (5, 2))
            if gmm_identities != expected_gmm:
                raise ValueError("xetra v4 GMM candidates must be exactly K2-K5 with M=2")
            if self.student_t_hmm is None:
                raise ValueError("xetra v4 requires the Student-t K2-K5 candidate family")
        else:
            raise ValueError("unsupported Xetra profile configuration version")

    def canonical_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.feature_selection is None:
            payload.pop("feature_selection", None)
        if self.feature_discovery is None:
            payload.pop("feature_discovery", None)
        return payload

    @property
    def profile_hash(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return sha256(payload).hexdigest()


def assert_xetra_v1_pins(profile: ModelProfile) -> None:
    """Fail closed unless every evaluation-contract constant is exactly pinned."""
    if profile.feature_selection is None:
        raise ValueError("xetra v1 pin audit requires legacy feature_selection")
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
        "gaussian_candidates": hmm.candidate_states == (2, 3, 4),
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
