"""Immutable, fail-closed contracts for Xetra global regime discovery v4."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from types import MappingProxyType
from typing import Any, cast

V4_PROFILE_ID = "xetra"
V4_PROFILE_CONFIG_VERSION = 4
V4_FEATURE_DISCOVERY_POLICY = "xetra_global_regime_v4"
V4_FEATURE_UNIVERSE_MODE = "all_numeric_source_features"
V4_EXCLUDED_SOURCE_COLUMN = "timestamp_m1"
MIN_FEATURE_COVERAGE = 0.90
MIN_FEATURE_VARIANCE = 1.0e-12
MIN_PAIRWISE_OBSERVATIONS = 504
V4_REDUNDANCY_MEASURE = "absolute_spearman_rank_correlation"
V4_DISTANCE_METHOD = "one_minus_absolute_spearman"
V4_CLUSTERING_METHOD = "agglomerative_average_linkage_precomputed"
CLUSTER_COUNT_MIN = 2
CLUSTER_COUNT_MAX = 25
SILHOUETTE_SINGLETON_VALUE = 0.0
SILHOUETTE_TIE_TOLERANCE = 1.0e-12
MIN_SILHOUETTE = 0.0
CLUSTER_COUNT_TIE_BREAK = "smaller_m"
V4_TEMPORARY_PROTOTYPE_METHOD = "within_cluster_correlation_medoid"
V4_PROVISIONAL_MODEL_FAMILY = "gaussian_hmm"
V4_PROVISIONAL_STATE_COUNTS = (2, 3, 4, 5)
MIN_FEATURE_SCORE_COVERAGE = 0.90
MIN_FEATURE_SCORE_OBSERVATIONS = 126
V4_FEATURE_REGIME_SCORE = "posterior_weighted_eta_squared"
FEATURE_SCORE_TIE_TOLERANCE = 1.0e-12
MIN_PREFIX_LENGTH = 2
V4_PREFIX_MODEL_FAMILY = "gaussian_hmm"
V4_PREFIX_STATE_COUNTS = (2, 3, 4, 5)
V4_PREFIX_SELECTION_TARGET = "dominant_state_nmi_to_common_causal_teacher"
MIN_TEACHER_SHARED_SUPPORT = 0.90
PREFIX_NMI_TIE_TOLERANCE = 1.0e-12
V4_CROSS_L_TIE_BREAKS = ("higher_shared_timestamp_count", "smaller_L")
V4_CROSS_L_LIKELIHOOD_FORBIDDEN = True
OUTER_TRAIN_SOURCE_OBSERVATIONS = 1260
OUTER_TEST_SOURCE_OBSERVATIONS = 63
OUTER_STEP_SOURCE_OBSERVATIONS = 63
OUTER_ALLOW_PARTIAL_FINAL_TEST = False
INNER_TRAIN_SOURCE_OBSERVATIONS = 756
INNER_TEST_SOURCE_OBSERVATIONS = 63
INNER_STEP_SOURCE_OBSERVATIONS = 63
INNER_ALLOW_PARTIAL_FINAL_TEST = False
MIN_MODEL_TRAIN_OBSERVATIONS = 504
MIN_MODEL_TEST_OBSERVATIONS = 42
V4_OUTER_STATE_IDENTITY = "outer_fold_local"

FINAL_CANDIDATE_IDS = (
    "gaussian_hmm_k2_full",
    "gaussian_hmm_k3_full",
    "gaussian_hmm_k4_full",
    "gaussian_hmm_k5_full",
    "gmm_hmm_k2_m2_full",
    "gmm_hmm_k3_m2_full",
    "gmm_hmm_k4_m2_full",
    "gmm_hmm_k5_m2_full",
    "student_t_hmm_k2_full",
    "student_t_hmm_k3_full",
    "student_t_hmm_k4_full",
    "student_t_hmm_k5_full",
)

# An immutable, serializable inventory makes the section-2 contract available to
# profile/schema code without requiring that code to duplicate constants.
V4_STATISTICAL_CONSTANTS: Mapping[str, object] = MappingProxyType(
    {
        "feature_universe_mode": V4_FEATURE_UNIVERSE_MODE,
        "excluded_source_column": V4_EXCLUDED_SOURCE_COLUMN,
        "minimum_feature_coverage": MIN_FEATURE_COVERAGE,
        "minimum_nonzero_population_variance": MIN_FEATURE_VARIANCE,
        "minimum_pairwise_complete_observations": MIN_PAIRWISE_OBSERVATIONS,
        "redundancy_measure": V4_REDUNDANCY_MEASURE,
        "distance": V4_DISTANCE_METHOD,
        "clustering": V4_CLUSTERING_METHOD,
        "candidate_cluster_count_minimum": CLUSTER_COUNT_MIN,
        "candidate_cluster_count_maximum": CLUSTER_COUNT_MAX,
        "silhouette_singleton_value": SILHOUETTE_SINGLETON_VALUE,
        "silhouette_tie_tolerance": SILHOUETTE_TIE_TOLERANCE,
        "minimum_accepted_best_silhouette": MIN_SILHOUETTE,
        "cluster_count_tie_break": CLUSTER_COUNT_TIE_BREAK,
        "temporary_prototype": V4_TEMPORARY_PROTOTYPE_METHOD,
        "provisional_model_family": V4_PROVISIONAL_MODEL_FAMILY,
        "provisional_candidate_state_counts": V4_PROVISIONAL_STATE_COUNTS,
        "inner_minimum_train_source_observations": INNER_TRAIN_SOURCE_OBSERVATIONS,
        "inner_test_source_observations": INNER_TEST_SOURCE_OBSERVATIONS,
        "inner_step_source_observations": INNER_STEP_SOURCE_OBSERVATIONS,
        "inner_partial_final_test": INNER_ALLOW_PARTIAL_FINAL_TEST,
        "inner_minimum_model_train_observations": MIN_MODEL_TRAIN_OBSERVATIONS,
        "inner_minimum_model_test_observations": MIN_MODEL_TEST_OBSERVATIONS,
        "feature_regime_score": V4_FEATURE_REGIME_SCORE,
        "minimum_feature_score_coverage": MIN_FEATURE_SCORE_COVERAGE,
        "minimum_feature_score_observations": MIN_FEATURE_SCORE_OBSERVATIONS,
        "feature_score_tie_tolerance": FEATURE_SCORE_TIE_TOLERANCE,
        "minimum_prefix_length": MIN_PREFIX_LENGTH,
        "prefix_model_family": V4_PREFIX_MODEL_FAMILY,
        "prefix_candidate_state_counts": V4_PREFIX_STATE_COUNTS,
        "prefix_selection_target": V4_PREFIX_SELECTION_TARGET,
        "minimum_shared_teacher_support": MIN_TEACHER_SHARED_SUPPORT,
        "prefix_nmi_tie_tolerance": PREFIX_NMI_TIE_TOLERANCE,
        "cross_l_tie_breaks": V4_CROSS_L_TIE_BREAKS,
        "cross_l_likelihood_forbidden": V4_CROSS_L_LIKELIHOOD_FORBIDDEN,
        "outer_walk_forward": (
            OUTER_TRAIN_SOURCE_OBSERVATIONS,
            OUTER_TEST_SOURCE_OBSERVATIONS,
            OUTER_STEP_SOURCE_OBSERVATIONS,
        ),
        "outer_partial_final_test": OUTER_ALLOW_PARTIAL_FINAL_TEST,
        "outer_fold_state_identity": V4_OUTER_STATE_IDENTITY,
        "final_candidate_universe": FINAL_CANDIDATE_IDS,
    }
)


class DiscoveryStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


def _finite(value: float, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _digest(value: str, name: str) -> None:
    if (
        len(value) != 64
        or value != value.lower()
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _feature_names(values: tuple[str, ...], name: str) -> None:
    if (
        not values
        or len(values) != len(set(values))
        or any(not isinstance(value, str) or not value.strip() for value in values)
    ):
        raise ValueError(f"{name} must be non-empty and duplicate-free")


def _validate(value: object, field_name: str = "") -> None:
    lowered = field_name.lower()
    if "semantic" in lowered or "economic" in lowered or "portfolio" in lowered:
        raise ValueError(f"forbidden statistical decision field: {field_name}")
    if any(token in lowered for token in ("raw_row", "raw_rows", "password", "dsn", "secret")):
        raise ValueError(f"forbidden contract field: {field_name}")
    if isinstance(value, float):
        _finite(value, field_name or "value")
    elif isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(f"{field_name} must be timezone-aware")
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("contract mapping keys must be strings")
            _validate(item, key)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _validate(item, field_name)
    elif value is not None and not isinstance(value, (str, int, bool, StrEnum)):
        raise ValueError(f"unsupported contract value in {field_name}: {type(value).__name__}")


def canonical_json(value: object) -> bytes:
    """Serialize a contract with stable ordering and strict nonfinite rejection."""
    payload: Any = asdict(cast(Any, value)) if hasattr(value, "__dataclass_fields__") else value
    _validate(payload)

    def normalize(item: object) -> object:
        if isinstance(item, StrEnum):
            return item.value
        if isinstance(item, datetime):
            return item.isoformat()
        if isinstance(item, dict):
            return {key: normalize(value) for key, value in sorted(item.items())}
        if isinstance(item, (tuple, list)):
            return [normalize(value) for value in item]
        return item

    return (
        json.dumps(normalize(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def content_hash(value: object) -> str:
    return sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureQuality:
    feature_name: str
    source_position: int
    train_observation_count: int
    finite_observation_count: int
    coverage: float
    population_variance: float
    eligible: bool
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.feature_name, str)
            or not self.feature_name.strip()
            or self.source_position < 0
        ):
            raise ValueError("feature quality identity is invalid")
        if (
            self.train_observation_count < 1
            or not 0 <= self.finite_observation_count <= self.train_observation_count
        ):
            raise ValueError("feature quality observation counts are invalid")
        _finite(self.coverage, "coverage")
        _finite(self.population_variance, "population_variance")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must be in [0,1]")
        if self.population_variance < 0.0:
            raise ValueError("population_variance must be non-negative")
        if (
            abs(self.coverage - self.finite_observation_count / self.train_observation_count)
            > 1.0e-12
        ):
            raise ValueError("coverage does not reconcile to observation counts")
        if self.eligible == (self.rejection_reason is not None):
            raise ValueError("eligible quality has no reason; rejected quality requires a reason")


@dataclass(frozen=True, slots=True)
class QualityFilterResult:
    source_build_id: str
    train_source_observation_count: int
    catalog_hash: str
    features: tuple[FeatureQuality, ...]
    eligible_features: tuple[str, ...]
    status: DiscoveryStatus = DiscoveryStatus.VALID

    def __post_init__(self) -> None:
        _digest(self.catalog_hash, "catalog_hash")
        if self.train_source_observation_count < 1:
            raise ValueError("quality filter needs positive TRAIN observations")
        if not isinstance(self.source_build_id, str) or not self.source_build_id.strip():
            raise ValueError("source_build_id cannot be empty")
        names = tuple(item.feature_name for item in self.features)
        _feature_names(names, "quality feature order")
        _feature_names(self.eligible_features, "eligible_features")
        if any(name not in names for name in self.eligible_features):
            raise ValueError("eligible feature is absent from quality evidence")
        if self.eligible_features != tuple(
            item.feature_name for item in self.features if item.eligible
        ):
            raise ValueError("eligible_features must preserve quality evidence order")
        if any(
            item.train_observation_count != self.train_source_observation_count
            for item in self.features
        ):
            raise ValueError("feature quality denominators must equal the TRAIN source count")
        if len(self.eligible_features) < 3:
            raise ValueError("v4 requires at least three eligible features")

    @property
    def result_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class DistanceMatrixResult:
    feature_order: tuple[str, ...]
    distances: tuple[tuple[float, ...], ...]
    pairwise_support: tuple[tuple[int, ...], ...]
    minimum_pairwise_observations: int = MIN_PAIRWISE_OBSERVATIONS

    def __post_init__(self) -> None:
        _feature_names(self.feature_order, "distance feature_order")
        if self.minimum_pairwise_observations < 1:
            raise ValueError("minimum pairwise support must be positive")
        size = len(self.feature_order)
        if len(self.distances) != size or any(len(row) != size for row in self.distances):
            raise ValueError("distance matrix must be square")
        if len(self.pairwise_support) != size or any(
            len(row) != size for row in self.pairwise_support
        ):
            raise ValueError("pairwise support matrix must be square")
        for row_index, row in enumerate(self.distances):
            for column_index, value in enumerate(row):
                _finite(value, "distance")
                if not 0.0 <= value <= 1.0:
                    raise ValueError("distance must be in [0,1]")
                if abs(value - self.distances[column_index][row_index]) > 1.0e-12:
                    raise ValueError("distance matrix must be symmetric")
                if row_index == column_index and value != 0.0:
                    raise ValueError("distance diagonal must be exactly zero")
                support = self.pairwise_support[row_index][column_index]
                if (
                    not isinstance(support, int)
                    or isinstance(support, bool)
                    or support < self.minimum_pairwise_observations
                ):
                    raise ValueError("distance pair lacks minimum support")
                if support != self.pairwise_support[column_index][row_index]:
                    raise ValueError("pairwise support matrix must be symmetric")
        if any(
            self.pairwise_support[index][index] < self.minimum_pairwise_observations
            for index in range(size)
        ):
            raise ValueError("pairwise support diagonal lacks minimum support")

    @property
    def matrix_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class ClusterSolution:
    candidate_count: int
    selected_count: int
    silhouette_curve: tuple[tuple[int, float], ...]
    memberships: tuple[tuple[str, tuple[str, ...]], ...]
    selected_silhouette: float
    singleton_count: int

    def __post_init__(self) -> None:
        if self.candidate_count < 3 or not CLUSTER_COUNT_MIN <= self.selected_count <= min(
            CLUSTER_COUNT_MAX, self.candidate_count - 1
        ):
            raise ValueError("cluster count is outside the pinned v4 bounds")
        expected_counts = tuple(
            range(CLUSTER_COUNT_MIN, min(CLUSTER_COUNT_MAX, self.candidate_count - 1) + 1)
        )
        if tuple(count for count, _ in self.silhouette_curve) != expected_counts:
            raise ValueError("silhouette curve is required")
        for _count, value in self.silhouette_curve:
            _finite(value, "silhouette")
            if not 0.0 <= value <= 1.0:
                raise ValueError("silhouette must be in [0,1]")
        _finite(self.selected_silhouette, "selected_silhouette")
        if self.selected_silhouette <= MIN_SILHOUETTE:
            raise ValueError("selected silhouette must be strictly positive")
        selected_curve_value = dict(self.silhouette_curve).get(self.selected_count)
        if (
            selected_curve_value is None
            or abs(selected_curve_value - self.selected_silhouette) > SILHOUETTE_TIE_TOLERANCE
        ):
            raise ValueError("selected silhouette must match the selected curve point")
        if len(self.memberships) != self.selected_count or self.singleton_count < 0:
            raise ValueError("cluster membership evidence is inconsistent")
        members = [feature for _, features in self.memberships for feature in features]
        _feature_names(tuple(members), "cluster members")
        if self.singleton_count != sum(len(features) == 1 for _, features in self.memberships):
            raise ValueError("singleton_count does not reconcile to memberships")
        if tuple(sorted(cluster_id for cluster_id, _ in self.memberships)) != tuple(
            f"cluster_{index:03d}" for index in range(self.selected_count)
        ):
            raise ValueError("cluster IDs must be canonical")

    @property
    def solution_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class PrototypeSet:
    cluster_ids: tuple[str, ...]
    prototypes: tuple[str, ...]
    mean_distances: tuple[tuple[str, float], ...]
    temporary: bool = True

    def __post_init__(self) -> None:
        if not self.temporary or not self.cluster_ids:
            raise ValueError("prototypes must be explicitly temporary")
        _feature_names(self.cluster_ids, "prototype cluster_ids")
        _feature_names(self.prototypes, "prototype features")
        if self.cluster_ids != tuple(
            f"cluster_{index:03d}" for index in range(len(self.cluster_ids))
        ):
            raise ValueError("prototype cluster IDs must be canonical")
        if len(self.cluster_ids) != len(self.prototypes):
            raise ValueError("one prototype is required per cluster")
        if len(self.mean_distances) < len(self.prototypes):
            raise ValueError("prototype tie evidence must cover every prototype")
        evidence_names = tuple(name for name, _ in self.mean_distances)
        _feature_names(evidence_names, "prototype tie evidence features")
        if not set(self.prototypes).issubset(evidence_names):
            raise ValueError("prototype tie evidence must include every prototype")
        for _, value in self.mean_distances:
            _finite(value, "prototype mean distance")


@dataclass(frozen=True, slots=True)
class ProvisionalTeacherReference:
    candidate_id: str
    state_count: int
    timestamps: tuple[datetime, ...]
    filtered_probabilities: tuple[tuple[float, ...], ...]
    dominant_states: tuple[int, ...]
    valid_inner_fold_ids: tuple[str, ...]
    source_build_id: str
    inner_plan_hash: str

    def __post_init__(self) -> None:
        if (
            self.state_count not in (2, 3, 4, 5)
            or not isinstance(self.candidate_id, str)
            or not self.candidate_id.strip()
        ):
            raise ValueError("teacher identity is invalid")
        _digest(self.inner_plan_hash, "inner_plan_hash")
        if (
            not len(self.timestamps)
            == len(self.filtered_probabilities)
            == len(self.dominant_states)
        ):
            raise ValueError("teacher timestamps, probabilities and states must align")
        if not self.timestamps or any(
            current <= previous
            for previous, current in zip(self.timestamps, self.timestamps[1:], strict=False)
        ):
            raise ValueError("teacher timestamps must be strictly increasing")
        for row, state in zip(self.filtered_probabilities, self.dominant_states, strict=True):
            if len(row) != self.state_count or state not in range(self.state_count):
                raise ValueError("teacher probability row has invalid state dimension")
            if (
                any(value < 0.0 or not isfinite(value) for value in row)
                or abs(sum(row) - 1.0) > 1.0e-10
            ):
                raise ValueError("teacher probability rows must be normalized and finite")

    @property
    def reference_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class FeatureRegimeScore:
    feature_name: str
    coverage: float
    observation_count: int
    state_weights: tuple[float, ...]
    state_means: tuple[float, ...]
    state_variances: tuple[float, ...]
    between_variance: float
    within_variance: float
    eta_squared: float | None
    eligible: bool
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.feature_name, str)
            or not self.feature_name.strip()
            or self.observation_count < 0
        ):
            raise ValueError("feature score identity/count is invalid")
        if len(self.state_weights) != len(self.state_means) or len(self.state_means) != len(
            self.state_variances
        ):
            raise ValueError("feature score state vectors must align")
        if any(value < 0.0 or not isfinite(value) for value in self.state_weights):
            raise ValueError("feature score weights must be finite and non-negative")
        for value in self.state_means:
            _finite(value, "feature score state mean")
        for value in (*self.state_variances, self.between_variance, self.within_variance):
            _finite(value, "feature score variance")
            if value < 0.0:
                raise ValueError("feature score variances must be non-negative")
        _finite(self.coverage, "feature score coverage")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("feature score coverage must be in [0,1]")
        if self.eligible:
            if self.exclusion_reason is not None or self.eta_squared is None:
                raise ValueError("eligible feature score requires eta_squared and no exclusion")
            _finite(self.eta_squared, "eta_squared")
            if not 0.0 <= self.eta_squared <= 1.0:
                raise ValueError("eta_squared must be in [0,1]")
            if (
                self.coverage < MIN_FEATURE_SCORE_COVERAGE
                or self.observation_count < MIN_FEATURE_SCORE_OBSERVATIONS
            ):
                raise ValueError("eligible feature score is below support gates")
        elif not self.exclusion_reason:
            raise ValueError("ineligible feature score requires exclusion_reason")


@dataclass(frozen=True, slots=True)
class ClusterWinner:
    cluster_id: str
    member_features: tuple[str, ...]
    winner_feature: str
    winner_score: FeatureRegimeScore

    def __post_init__(self) -> None:
        _feature_names(self.member_features, "cluster member features")
        if (
            not self.cluster_id.startswith("cluster_")
            or self.winner_feature not in self.member_features
        ):
            raise ValueError("cluster winner identity is invalid")
        if self.winner_score.feature_name != self.winner_feature or not self.winner_score.eligible:
            raise ValueError("cluster winner must reference an eligible score")


@dataclass(frozen=True, slots=True)
class RankedWinnerSet:
    winners: tuple[ClusterWinner, ...]
    ranked_features: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.winners or len(self.winners) != len(self.ranked_features):
            raise ValueError("ranked winners must contain one feature per cluster")
        if self.ranked_features != tuple(item.winner_feature for item in self.winners):
            raise ValueError("ranked_features must preserve winner order")
        if len(set(self.ranked_features)) != len(self.ranked_features):
            raise ValueError("ranked winner features must be unique")
        if len({item.cluster_id for item in self.winners}) != len(self.winners):
            raise ValueError("ranked winners must come from distinct clusters")


@dataclass(frozen=True, slots=True)
class PrefixEvaluation:
    prefix_length: int
    feature_order: tuple[str, ...]
    candidate_id: str
    shared_timestamp_count: int
    shared_teacher_coverage: float
    dominant_state_nmi: float

    def __post_init__(self) -> None:
        if self.prefix_length < MIN_PREFIX_LENGTH or len(self.feature_order) != self.prefix_length:
            raise ValueError("prefix length and feature order are inconsistent")
        _feature_names(self.feature_order, "prefix feature order")
        if self.candidate_id not in tuple(
            f"gaussian_hmm_k{state_count}_full" for state_count in V4_PREFIX_STATE_COUNTS
        ):
            raise ValueError("prefix search permits Gaussian candidates only")
        if self.shared_timestamp_count < 0:
            raise ValueError("shared timestamp count cannot be negative")
        for value, name in (
            (self.shared_teacher_coverage, "shared_teacher_coverage"),
            (self.dominant_state_nmi, "dominant_state_nmi"),
        ):
            _finite(value, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")


@dataclass(frozen=True, slots=True)
class PrefixSearchResult:
    ranked_features: tuple[str, ...]
    evaluations: tuple[PrefixEvaluation, ...]
    selected_prefix_length: int
    selected_candidate_id: str

    def __post_init__(self) -> None:
        _feature_names(self.ranked_features, "ranked_features")
        if not MIN_PREFIX_LENGTH <= self.selected_prefix_length <= len(self.ranked_features):
            raise ValueError("selected prefix length is outside ranked winner bounds")
        if tuple(range(MIN_PREFIX_LENGTH, len(self.ranked_features) + 1)) != tuple(
            item.prefix_length for item in self.evaluations
        ):
            raise ValueError("prefix evaluations must cover every exact nested prefix")
        selected = self.evaluations[self.selected_prefix_length - MIN_PREFIX_LENGTH]
        if (
            selected.candidate_id != self.selected_candidate_id
            or selected.feature_order != self.ranked_features[: self.selected_prefix_length]
        ):
            raise ValueError("selected prefix is not an exact ranked prefix")


@dataclass(frozen=True, slots=True)
class FinalSelectedConfiguration:
    feature_order: tuple[str, ...]
    candidate_id: str
    state_count: int
    model_family: str
    selected_prefix_length: int
    feature_discovery_hash: str

    def __post_init__(self) -> None:
        _feature_names(self.feature_order, "final feature_order")
        _digest(self.feature_discovery_hash, "feature_discovery_hash")
        if (
            len(self.feature_order) != self.selected_prefix_length
            or self.selected_prefix_length < MIN_PREFIX_LENGTH
        ):
            raise ValueError("final feature count is inconsistent")
        if self.candidate_id not in FINAL_CANDIDATE_IDS:
            raise ValueError("final candidate is outside the exact 12-candidate universe")
        if self.state_count not in (2, 3, 4, 5) or self.model_family not in {
            "gaussian_hmm",
            "gmm_hmm",
            "student_t_hmm",
        }:
            raise ValueError("final candidate model identity is invalid")
        expected_family = (
            "gmm_hmm"
            if self.candidate_id.startswith("gmm_hmm_")
            else self.candidate_id.rsplit("_k", 1)[0]
        )
        expected_id = (
            f"gmm_hmm_k{self.state_count}_m2_full"
            if expected_family == "gmm_hmm"
            else f"{expected_family}_k{self.state_count}_full"
        )
        if self.model_family != expected_family or self.candidate_id != expected_id:
            raise ValueError("final candidate identity does not match family/state")


@dataclass(frozen=True, slots=True)
class OuterFoldResult:
    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    final_configuration: FinalSelectedConfiguration
    oos_predictive_loglik_per_observation: float
    oos_timestamps: tuple[datetime, ...]
    oos_filtered_probabilities: tuple[tuple[float, ...], ...]
    state_identity: str = "outer_fold_local"

    def __post_init__(self) -> None:
        if (
            self.fold_index < 1
            or not self.train_start <= self.train_end < self.test_start <= self.test_end
        ):
            raise ValueError("outer fold bounds are invalid")
        if self.state_identity != "outer_fold_local":
            raise ValueError("outer fold state identity is pinned as fold-local")
        _finite(self.oos_predictive_loglik_per_observation, "oos_predictive_loglik_per_observation")
        if len(self.oos_timestamps) != len(self.oos_filtered_probabilities):
            raise ValueError("outer OOS timestamps and probabilities must align")
        if any(
            current <= previous
            for previous, current in zip(self.oos_timestamps, self.oos_timestamps[1:], strict=False)
        ):
            raise ValueError("outer OOS timestamps must be strictly increasing")
        if not self.oos_filtered_probabilities:
            raise ValueError("outer OOS evidence cannot be empty")
        for row in self.oos_filtered_probabilities:
            if len(row) != self.final_configuration.state_count:
                raise ValueError("outer OOS probability rows have inconsistent state dimension")
            if (
                any(not isfinite(value) or value < 0.0 for value in row)
                or abs(sum(row) - 1.0) > 1.0e-10
            ):
                raise ValueError("outer OOS probabilities must be normalized and finite")

    @property
    def result_hash(self) -> str:
        return content_hash(self)
