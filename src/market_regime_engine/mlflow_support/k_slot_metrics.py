"""Contracts for K-slot Model Metrics and comparison payloads.

This module deliberately does not start an MLflow run and does not accept an
evaluation object.  It validates and canonicalizes already projected Model
Metrics.  Consequently it can be prepared in independent worker processes;
the caller remains responsible for ordered, idempotent MLflow side effects.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256

from market_regime_engine.contracts.core import KChampionSlot
from market_regime_engine.mlflow_support.metric_catalog import (
    METRIC_CATALOG_VERSION,
    require_metric_definition,
    validate_metric_points,
)
from market_regime_engine.mlflow_support.ports import MetricPoint

LEGAL_K = (2, 3, 4, 5)
K_SLOT_POLICY_VERSION = "k_champion_policy.v1"
K_COMPARISON_DOMAIN_ID = "k_specific_shared_feature_vector.v1"
K_CROSS_COMPARISON_DOMAIN_ID = "cross_k_dimension_independent.v1"
MODEL_FAMILIES = ("gaussian_hmm", "gmm_hmm", "student_t_hmm")

TAG_SLOT_ID = "regime_engine.slot_id"
TAG_STATE_COUNT = "regime_engine.state_count"
TAG_FEATURE_ORDER_SHA256 = "regime_engine.feature_order_sha256"
TAG_FEATURE_DIMENSION = "regime_engine.feature_dimension"
TAG_POLICY_VERSION = "regime_engine.policy_version"
TAG_COMPARISON_DOMAIN_ID = "regime_engine.comparison_domain_id"
TAG_SOURCE_BUILD_ID = "regime_engine.source_build_id"
TAG_SOURCE_DATA_SHA256 = "regime_engine.source_data_sha256"
TAG_EVALUATION_PLAN_HASH = "regime_engine.evaluation_plan_hash"

# Only these catalogued quantities are dimension-independent enough for a
# cross-K plot.  In particular, likelihood and information criteria remain
# same-feature-vector metrics even when their numeric values happen to exist.
CROSS_K_METRIC_KEYS = frozenset(
    {
        "valid_fold_rate",
        "valid_fold_count",
        "invalid_fold_count",
        "soft_regime_nmi",
        "shared_timestamp_count",
        "outer_soft_regime_nmi",
        "outer_shared_timestamp_count",
        "prefix_soft_regime_nmi",
    }
)


def _text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty trimmed string")


def _sha256(value: str, field: str) -> None:
    _text(value, field)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase SHA-256")


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def feature_order_hash(feature_order: Sequence[str]) -> str:
    """Hash the exact ordered feature tuple used by one K slot."""

    order = tuple(feature_order)
    if not order or any(not isinstance(item, str) or not item for item in order):
        raise ValueError("feature_order must contain non-empty feature names")
    if len(set(order)) != len(order):
        raise ValueError("feature_order must not contain duplicates")
    return _canonical_hash(list(order))


@dataclass(frozen=True, slots=True)
class KSlotMetadata:
    """Lineage identity shared by every candidate in one K slot."""

    slot_id: str
    state_count: int
    feature_order: tuple[str, ...]
    policy_version: str
    comparison_domain_id: str
    source_build_id: str
    source_data_sha256: str
    evaluation_plan_hash: str

    def __post_init__(self) -> None:
        if self.state_count not in LEGAL_K or isinstance(self.state_count, bool):
            raise ValueError("K slot state_count must be one of 2, 3, 4 or 5")
        try:
            slot = KChampionSlot(self.slot_id)
        except ValueError as exc:
            raise ValueError("slot_id must be one of k2, k3, k4 or k5") from exc
        if slot.state_count != self.state_count:
            raise ValueError("slot_id state count does not match state_count")
        if not self.feature_order or len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("K slot feature_order must be non-empty and unique")
        if any(not isinstance(item, str) or not item.strip() for item in self.feature_order):
            raise ValueError("K slot feature_order contains an invalid name")
        _text(self.policy_version, "policy_version")
        _text(self.comparison_domain_id, "comparison_domain_id")
        _text(self.source_build_id, "source_build_id")
        _sha256(self.source_data_sha256, "source_data_sha256")
        _sha256(self.evaluation_plan_hash, "evaluation_plan_hash")
        if self.policy_version != K_SLOT_POLICY_VERSION:
            raise ValueError("unsupported K-slot policy version")
        if self.comparison_domain_id != K_COMPARISON_DOMAIN_ID:
            raise ValueError("unsupported K-slot comparison domain")

    @property
    def feature_order_sha256(self) -> str:
        return feature_order_hash(self.feature_order)

    @property
    def feature_dimension(self) -> int:
        return len(self.feature_order)

    def tags(self, *, scope: str) -> dict[str, str]:
        _text(scope, "scope")
        return {
            TAG_SLOT_ID: self.slot_id,
            "regime_engine.alias": KChampionSlot(self.slot_id).alias,
            TAG_STATE_COUNT: str(self.state_count),
            TAG_FEATURE_ORDER_SHA256: self.feature_order_sha256,
            TAG_FEATURE_DIMENSION: str(self.feature_dimension),
            TAG_POLICY_VERSION: self.policy_version,
            TAG_COMPARISON_DOMAIN_ID: self.comparison_domain_id,
            TAG_SOURCE_BUILD_ID: self.source_build_id,
            TAG_SOURCE_DATA_SHA256: self.source_data_sha256,
            TAG_EVALUATION_PLAN_HASH: self.evaluation_plan_hash,
            "regime_engine.scope": scope,
            "regime_engine.metric_catalog_version": str(METRIC_CATALOG_VERSION),
        }


@dataclass(frozen=True, slots=True)
class KCandidateMetricProjection:
    """One candidate LoggedModel's raw and aggregate Model Metrics."""

    logged_model_id: str
    model_family: str
    metadata: KSlotMetadata
    metric_points: tuple[MetricPoint, ...]

    def __post_init__(self) -> None:
        _text(self.logged_model_id, "logged_model_id")
        if self.model_family not in MODEL_FAMILIES:
            raise ValueError("K candidate model_family is not one of the three permitted families")
        if not self.metric_points:
            raise ValueError("K candidate must preserve at least one Model Metric point")
        validate_metric_points(self.metric_points)
        ordered = tuple(sorted(self.metric_points, key=lambda item: (item.key, item.step)))
        object.__setattr__(self, "metric_points", ordered)

    @property
    def tags(self) -> dict[str, str]:
        return self.metadata.tags(scope="candidate")

    def canonical_dict(self) -> dict[str, object]:
        return {
            "logged_model_id": self.logged_model_id,
            "model_family": self.model_family,
            "metadata": {
                "slot_id": self.metadata.slot_id,
                "state_count": self.metadata.state_count,
                "feature_order": list(self.metadata.feature_order),
                "feature_order_sha256": self.metadata.feature_order_sha256,
                "policy_version": self.metadata.policy_version,
                "comparison_domain_id": self.metadata.comparison_domain_id,
                "source_build_id": self.metadata.source_build_id,
                "source_data_sha256": self.metadata.source_data_sha256,
                "evaluation_plan_hash": self.metadata.evaluation_plan_hash,
            },
            "metric_points": [asdict(point) for point in self.metric_points],
        }


@dataclass(frozen=True, slots=True)
class KSlotMetricProjection:
    """Complete candidate/selected projection for exactly one K slot."""

    metadata: KSlotMetadata
    candidates: tuple[KCandidateMetricProjection, ...]
    eligible: bool
    selected_logged_model_id: str | None = None
    selected_metric_points: tuple[MetricPoint, ...] = ()
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if len(self.candidates) != len(MODEL_FAMILIES):
            raise ValueError("each K slot must contain exactly the three model families")
        families = tuple(item.model_family for item in self.candidates)
        if set(families) != set(MODEL_FAMILIES):
            raise ValueError("K slot must contain Gaussian, GMM and Student-t candidates")
        if any(item.metadata != self.metadata for item in self.candidates):
            raise ValueError("all K candidates must share one feature/source/policy identity")
        ids = tuple(item.logged_model_id for item in self.candidates)
        if len(set(ids)) != len(ids):
            raise ValueError("K candidate LoggedModel IDs must be unique")
        if not isinstance(self.eligible, bool):
            raise ValueError("K slot eligibility must be boolean")
        if self.eligible:
            if not self.selected_logged_model_id:
                raise ValueError("eligible K slot requires a selected LoggedModel")
            if self.selected_logged_model_id not in ids:
                raise ValueError("selected K LoggedModel is not one of the candidates")
            if not self.selected_metric_points:
                raise ValueError("eligible K slot requires selected/final Model Metrics")
            if self.unavailable_reason is not None:
                raise ValueError("eligible K slot cannot have an unavailable reason")
        else:
            if self.selected_logged_model_id is not None or self.selected_metric_points:
                raise ValueError("ineligible K slot cannot contain a selected projection")
            _text(self.unavailable_reason or "", "unavailable_reason")
        if self.selected_metric_points:
            validate_metric_points(self.selected_metric_points)
            object.__setattr__(
                self,
                "selected_metric_points",
                tuple(sorted(self.selected_metric_points, key=lambda item: (item.key, item.step))),
            )
        object.__setattr__(
            self,
            "candidates",
            tuple(sorted(self.candidates, key=lambda item: item.model_family)),
        )

    @property
    def slot_id(self) -> str:
        return self.metadata.slot_id

    @property
    def canonical_payload_hash(self) -> str:
        return _canonical_hash(self.canonical_dict())

    def canonical_dict(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "state_count": self.metadata.state_count,
            "eligible": self.eligible,
            "selected_logged_model_id": self.selected_logged_model_id,
            "unavailable_reason": self.unavailable_reason,
            "candidates": [item.canonical_dict() for item in self.candidates],
            "selected_metric_points": [asdict(point) for point in self.selected_metric_points],
        }


def build_k_slot_projection(
    candidates: Sequence[KCandidateMetricProjection],
    *,
    eligible: bool,
    selected_logged_model_id: str | None = None,
    selected_metric_points: Sequence[MetricPoint] = (),
    unavailable_reason: str | None = None,
) -> KSlotMetricProjection:
    """Validate one slot without deriving or inventing any Model Metrics."""

    candidate_tuple = tuple(candidates)
    if not candidate_tuple:
        raise ValueError("K slot requires candidate projections")
    metadata = candidate_tuple[0].metadata
    return KSlotMetricProjection(
        metadata=metadata,
        candidates=candidate_tuple,
        eligible=eligible,
        selected_logged_model_id=selected_logged_model_id,
        selected_metric_points=tuple(selected_metric_points),
        unavailable_reason=unavailable_reason,
    )


def build_four_k_slot_projection(
    slots: Sequence[KSlotMetricProjection],
) -> tuple[KSlotMetricProjection, ...]:
    """Return all four canonical slots in K order, rejecting omissions/duplicates."""

    by_k = {slot.metadata.state_count: slot for slot in slots}
    if len(by_k) != len(tuple(slots)) or tuple(sorted(by_k)) != LEGAL_K:
        raise ValueError("the K-slot portfolio must contain exactly K=2,3,4,5")
    return tuple(by_k[state_count] for state_count in LEGAL_K)


def validate_k_metric_points(
    metric_points: Mapping[str, Sequence[MetricPoint]],
    tags: Mapping[str, Mapping[str, str]],
) -> None:
    """Validate a LoggedModel collection before any plot payload is built."""

    if not metric_points:
        raise ValueError("K metric collection cannot be empty")
    for model_id, points in metric_points.items():
        if not model_id or model_id not in tags:
            raise ValueError(f"missing tags for LoggedModel {model_id!r}")
        point_tuple = tuple(points)
        validate_metric_points(point_tuple)
        if not point_tuple:
            raise ValueError(f"LoggedModel {model_id!r} has no Model Metrics")
        if tags[model_id].get(TAG_FEATURE_ORDER_SHA256, "") == "":
            raise ValueError(f"LoggedModel {model_id!r} is missing feature-order hash")
        _sha256(tags[model_id][TAG_FEATURE_ORDER_SHA256], "feature_order_sha256")


def validate_k_metric_comparison(
    metric_key: str,
    metric_points: Mapping[str, Sequence[MetricPoint]],
    tags: Mapping[str, Mapping[str, str]],
    *,
    state_count: int,
    cross_k: bool = False,
) -> None:
    """Apply the K-specific comparison domain without reading evaluation data."""

    definition = require_metric_definition(metric_key)
    validate_k_metric_points(metric_points, tags)
    if definition.comparison_domain == "same_feature_vector_source_plan":
        hashes = {tags[model_id].get(TAG_FEATURE_ORDER_SHA256, "") for model_id in metric_points}
        dimensions = {tags[model_id].get(TAG_FEATURE_DIMENSION, "") for model_id in metric_points}
        if len(hashes) != 1 or len(dimensions) != 1:
            raise ValueError(
                f"{metric_key} cannot compare K-specific feature dimensions or feature hashes"
            )
    if cross_k:
        if metric_key not in CROSS_K_METRIC_KEYS:
            raise ValueError(
                f"{metric_key} is not explicitly dimension-independent for cross-K comparison"
            )
        if definition.comparison_domain != "same_source_plan":
            raise ValueError(f"{metric_key} has no cross-K comparison domain")
    if not 2 <= state_count <= 5:
        raise ValueError("state_count must be one of 2, 3, 4 or 5")
    for model_id, points in metric_points.items():
        if any(point.key != metric_key for point in points):
            raise ValueError(f"LoggedModel {model_id!r} contains a different metric key")


def slot_metric_tags(slot: KSlotMetricProjection, *, scope: str) -> dict[str, str]:
    """Return canonical tags for candidate or selected LoggedModel creation."""

    return slot.metadata.tags(scope=scope)


__all__ = [
    "CROSS_K_METRIC_KEYS",
    "K_SLOT_POLICY_VERSION",
    "LEGAL_K",
    "MODEL_FAMILIES",
    "KCandidateMetricProjection",
    "KSlotMetadata",
    "KSlotMetricProjection",
    "build_four_k_slot_projection",
    "build_k_slot_projection",
    "feature_order_hash",
    "slot_metric_tags",
    "validate_k_metric_comparison",
    "validate_k_metric_points",
]
