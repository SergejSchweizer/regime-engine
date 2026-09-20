"""Versioned contracts for the independent K=2..5 champion slots.

This module intentionally contains no model fitting or MLflow client calls.
It is the immutable boundary shared by K-specific selection, outer validation,
and the registry adapter.  In particular, no raw PLL/AIC/BIC value appears in
the cross-feature-set promotion contract.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite

from market_regime_engine.contracts.core import (
    K_CHAMPION_POLICY_ID,
    K_CHAMPION_POLICY_VERSION,
    K_CHAMPION_SLOT_IDS,
    KChampionSlot,
)

LEGAL_MODEL_FAMILIES = ("gaussian_hmm", "gmm_hmm", "student_t_hmm")
WITHIN_K_COMPARISON_DOMAIN = "same_feature_vector.v1"
K_SLOT_COMPARISON_DOMAIN = "k_slot_promotion.v1"
K_SLOT_PROMOTION_SCORE_VERSION = "k_slot_promotion.v1"
K_SLOT_SCORE_TOLERANCE = 1.0e-12
MIN_VALID_FOLD_RATE = 0.80
MIN_VALID_FOLD_COUNT = 3


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty trimmed string")
    return value


def _sha(value: object, field: str) -> str:
    value = _text(value, field)
    if (
        len(value) != 64
        or value != value.lower()
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _utc(value: object, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != UTC.utcoffset(value)
    ):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return value


def _unit(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite and in [0, 1]")
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return value


def _slot(value: object) -> KChampionSlot:
    try:
        slot = value if isinstance(value, KChampionSlot) else KChampionSlot(str(value))
    except ValueError as exc:
        raise ValueError("slot_id must be one of k2, k3, k4 or k5") from exc
    if slot.value not in K_CHAMPION_SLOT_IDS:
        raise ValueError("slot_id is outside the K-champion portfolio")
    return slot


def feature_order_hash(feature_order: Sequence[str]) -> str:
    """Hash an ordered feature tuple without operational metadata."""

    order = tuple(_text(feature, "feature_order item") for feature in feature_order)
    if not order or len(set(order)) != len(order):
        raise ValueError("feature_order must be non-empty and duplicate-free")
    payload = json.dumps(list(order), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class KChampionSelection:
    """One immutable eligible selection record for exactly one K slot."""

    slot_id: str
    state_count: int
    model_family: str
    candidate_identity: str
    feature_order: tuple[str, ...]
    feature_order_hash: str
    source_snapshot_id: str
    profile_id: str
    profile_config_version: int
    policy_id: str
    policy_version: str
    validation_cutoff: datetime
    deployment_cutoff: datetime
    comparison_domain_id: str
    promotion_score_version: str
    reference_teacher_id: str
    artifact_hash: str
    source_build_id: str = ""
    source_catalog_hash: str = ""

    def __post_init__(self) -> None:
        slot = _slot(self.slot_id)
        if isinstance(self.state_count, bool) or self.state_count != slot.state_count:
            raise ValueError("state_count must equal the K encoded by slot_id")
        if self.model_family not in LEGAL_MODEL_FAMILIES:
            raise ValueError("model_family must be Gaussian, GMM or Student-t HMM")
        _text(self.candidate_identity, "candidate_identity")
        computed_order_hash = feature_order_hash(self.feature_order)
        if self.feature_order_hash != computed_order_hash:
            raise ValueError("feature_order_hash does not match the ordered feature tuple")
        _sha(self.feature_order_hash, "feature_order_hash")
        _text(self.source_snapshot_id, "source_snapshot_id")
        _text(self.profile_id, "profile_id")
        if isinstance(self.profile_config_version, bool) or self.profile_config_version < 1:
            raise ValueError("profile_config_version must be positive")
        if self.policy_id != K_CHAMPION_POLICY_ID:
            raise ValueError("unsupported K-champion policy identifier")
        if self.policy_version != K_CHAMPION_POLICY_VERSION:
            raise ValueError("unsupported K-champion policy version")
        validation = _utc(self.validation_cutoff, "validation_cutoff")
        deployment = _utc(self.deployment_cutoff, "deployment_cutoff")
        if validation >= deployment:
            raise ValueError("deployment_cutoff must be after validation_cutoff")
        if self.comparison_domain_id != K_SLOT_COMPARISON_DOMAIN:
            raise ValueError("selection records must use the K-slot promotion domain")
        if self.promotion_score_version != K_SLOT_PROMOTION_SCORE_VERSION:
            raise ValueError("unsupported K-slot promotion score version")
        _text(self.reference_teacher_id, "reference_teacher_id")
        _sha(self.artifact_hash, "artifact_hash")
        if self.source_build_id:
            _text(self.source_build_id, "source_build_id")
        if self.source_catalog_hash:
            _sha(self.source_catalog_hash, "source_catalog_hash")

    @property
    def slot(self) -> KChampionSlot:
        return _slot(self.slot_id)

    @property
    def alias(self) -> str:
        return self.slot.alias

    @property
    def policy_hash(self) -> str:
        payload = {
            "comparison_domain_id": self.comparison_domain_id,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "promotion_score_version": self.promotion_score_version,
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def canonical_json(self) -> str:
        payload = asdict(self)
        payload["validation_cutoff"] = self.validation_cutoff.isoformat().replace("+00:00", "Z")
        payload["deployment_cutoff"] = self.deployment_cutoff.isoformat().replace("+00:00", "Z")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_canonical_json(cls, payload: str) -> KChampionSelection:
        """Load one exact selection record and reject schema drift.

        Operational fields such as run IDs, wall-clock timestamps and temporary
        paths are intentionally not part of this contract.  Rejecting unknown
        keys keeps them out of the identity hash instead of silently accepting
        non-canonical provenance.
        """

        try:
            raw = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("K-champion selection must be canonical JSON") from exc
        expected = {item.name for item in fields(cls)}
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("unknown or missing K-champion selection fields")
        feature_order = raw["feature_order"]
        if not isinstance(feature_order, list):
            raise ValueError("feature_order must be a JSON array")
        for name in ("state_count", "profile_config_version"):
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        cutoffs: dict[str, datetime] = {}
        for name in ("validation_cutoff", "deployment_cutoff"):
            value = raw[name]
            if not isinstance(value, str) or not value.endswith("Z"):
                raise ValueError(f"{name} must use canonical UTC Z notation")
            try:
                cutoffs[name] = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
            except ValueError as exc:
                raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
        values = dict(raw)
        values["feature_order"] = tuple(feature_order)
        values.update(cutoffs)
        return cls(**values)

    @property
    def selection_hash(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def idempotency_key(self) -> str:
        """Stable key for one candidate artifact, excluding run metadata."""

        payload = {
            "artifact_hash": self.artifact_hash,
            "candidate_identity": self.candidate_identity,
            "feature_order_hash": self.feature_order_hash,
            "policy_hash": self.policy_hash,
            "slot_id": self.slot_id,
            "source_snapshot_id": self.source_snapshot_id,
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class KChampionPromotionEvidence:
    """Dimension-independent evidence for comparing feature sets within one K."""

    slot_id: str
    source_snapshot_id: str
    profile_id: str
    profile_config_version: int
    policy_version: str
    validation_cutoff: datetime
    reference_teacher_id: str
    feature_order_hash: str
    planned_fold_count: int
    valid_fold_count: int
    latest_complete_fold_valid: bool
    mean_soft_regime_nmi: float | None
    worst_fold_soft_regime_nmi: float | None
    common_support: float | None
    stability: float | None
    rejection_reasons: tuple[str, ...] = ()
    comparison_domain_id: str = K_SLOT_COMPARISON_DOMAIN
    promotion_score_version: str = K_SLOT_PROMOTION_SCORE_VERSION

    def __post_init__(self) -> None:
        _slot(self.slot_id)
        _text(self.source_snapshot_id, "source_snapshot_id")
        _text(self.profile_id, "profile_id")
        if isinstance(self.profile_config_version, bool) or self.profile_config_version < 1:
            raise ValueError("profile_config_version must be positive")
        if self.policy_version != K_CHAMPION_POLICY_VERSION:
            raise ValueError("unsupported policy version")
        _utc(self.validation_cutoff, "validation_cutoff")
        _text(self.reference_teacher_id, "reference_teacher_id")
        _sha(self.feature_order_hash, "feature_order_hash")
        if (
            isinstance(self.planned_fold_count, bool)
            or isinstance(self.valid_fold_count, bool)
            or not isinstance(self.planned_fold_count, int)
            or not isinstance(self.valid_fold_count, int)
            or self.planned_fold_count < 1
            or not 0 <= self.valid_fold_count <= self.planned_fold_count
        ):
            raise ValueError("promotion fold counts are inconsistent")
        if not isinstance(self.latest_complete_fold_valid, bool):
            raise ValueError("latest_complete_fold_valid must be boolean")
        for value, name in (
            (self.mean_soft_regime_nmi, "mean_soft_regime_nmi"),
            (self.worst_fold_soft_regime_nmi, "worst_fold_soft_regime_nmi"),
            (self.common_support, "common_support"),
            (self.stability, "stability"),
        ):
            if value is not None:
                _unit(value, name)
        if not isinstance(self.rejection_reasons, tuple) or any(
            not isinstance(reason, str) or not reason or reason.strip() != reason
            for reason in self.rejection_reasons
        ):
            raise ValueError("rejection_reasons must contain trimmed non-empty strings")
        if len(set(self.rejection_reasons)) != len(self.rejection_reasons):
            raise ValueError("rejection_reasons must be unique")
        if self.comparison_domain_id != K_SLOT_COMPARISON_DOMAIN:
            raise ValueError("promotion evidence must use the K-slot domain")
        if self.promotion_score_version != K_SLOT_PROMOTION_SCORE_VERSION:
            raise ValueError("unsupported promotion score version")
        if self.eligible and any(
            value is None
            for value in (
                self.mean_soft_regime_nmi,
                self.worst_fold_soft_regime_nmi,
                self.common_support,
                self.stability,
            )
        ):
            raise ValueError("eligible promotion evidence requires every score component")

    @property
    def valid_fold_rate(self) -> float:
        return self.valid_fold_count / self.planned_fold_count

    @property
    def eligible(self) -> bool:
        return (
            not self.rejection_reasons
            and self.valid_fold_rate >= MIN_VALID_FOLD_RATE
            and self.valid_fold_count >= MIN_VALID_FOLD_COUNT
            and self.latest_complete_fold_valid
        )

    @property
    def score_tuple(self) -> tuple[float, float, float, float]:
        if not self.eligible:
            raise ValueError("ineligible promotion evidence has no score tuple")
        assert self.mean_soft_regime_nmi is not None
        assert self.worst_fold_soft_regime_nmi is not None
        assert self.common_support is not None
        assert self.stability is not None
        return (
            self.valid_fold_rate,
            self.mean_soft_regime_nmi,
            self.worst_fold_soft_regime_nmi,
            self.common_support,
        )


@dataclass(frozen=True, slots=True)
class KChampionPromotionCandidate:
    selection: KChampionSelection
    evidence: KChampionPromotionEvidence

    def __post_init__(self) -> None:
        if self.selection.slot_id != self.evidence.slot_id:
            raise ValueError("selection/evidence slot mismatch")
        if self.selection.source_snapshot_id != self.evidence.source_snapshot_id:
            raise ValueError("selection/evidence source snapshot mismatch")
        if self.selection.profile_id != self.evidence.profile_id:
            raise ValueError("selection/evidence profile mismatch")
        if self.selection.profile_config_version != self.evidence.profile_config_version:
            raise ValueError("selection/evidence profile version mismatch")
        if self.selection.policy_version != self.evidence.policy_version:
            raise ValueError("selection/evidence policy version mismatch")
        if self.selection.validation_cutoff != self.evidence.validation_cutoff:
            raise ValueError("selection/evidence validation cutoff mismatch")
        if self.selection.reference_teacher_id != self.evidence.reference_teacher_id:
            raise ValueError("selection/evidence teacher identity mismatch")
        if self.selection.feature_order_hash != self.evidence.feature_order_hash:
            raise ValueError("selection/evidence feature hash mismatch")


@dataclass(frozen=True, slots=True)
class KChampionPromotionDecision:
    """Deterministic within-K feature-set decision; it never chooses another K."""

    slot_id: str
    candidates: tuple[KChampionPromotionCandidate, ...]
    winner: KChampionPromotionCandidate | None
    no_champion_reason: str | None = None

    def __post_init__(self) -> None:
        _slot(self.slot_id)
        if not self.candidates:
            raise ValueError("promotion decision requires candidates")
        if any(candidate.selection.slot_id != self.slot_id for candidate in self.candidates):
            raise ValueError("promotion decision candidates must share one slot")
        hashes = tuple(candidate.selection.feature_order_hash for candidate in self.candidates)
        if len(set(hashes)) != len(hashes):
            raise ValueError("promotion candidates must have distinct feature sets")
        if self.winner is not None:
            if self.winner not in self.candidates or not self.winner.evidence.eligible:
                raise ValueError("winner must be an eligible supplied candidate")
            if self.no_champion_reason is not None:
                raise ValueError("winner cannot have a no-champion reason")
        elif (
            not self.no_champion_reason
            or self.no_champion_reason.strip() != self.no_champion_reason
        ):
            raise ValueError("missing winner requires a no-champion reason")


def rank_k_slot_candidates(
    candidates: Sequence[KChampionPromotionCandidate],
) -> KChampionPromotionDecision:
    """Rank feature-set candidates using only dimension-independent policy evidence."""

    items = tuple(candidates)
    if not items:
        raise ValueError("at least one K-slot promotion candidate is required")
    slot_id = items[0].selection.slot_id
    if any(item.selection.slot_id != slot_id for item in items):
        raise ValueError("cross-K promotion comparison is forbidden")
    first = items[0].evidence
    for item in items[1:]:
        evidence = item.evidence
        fields = (
            (evidence.source_snapshot_id, first.source_snapshot_id, "source snapshot"),
            (evidence.profile_id, first.profile_id, "profile"),
            (evidence.profile_config_version, first.profile_config_version, "profile version"),
            (evidence.policy_version, first.policy_version, "policy version"),
            (evidence.validation_cutoff, first.validation_cutoff, "validation window"),
            (evidence.reference_teacher_id, first.reference_teacher_id, "reference teacher"),
            (evidence.promotion_score_version, first.promotion_score_version, "score version"),
        )
        for left, right, label in fields:
            if left != right:
                raise ValueError(f"promotion candidates must share identical {label}")
    eligible = [item for item in items if item.evidence.eligible]
    if not eligible:
        return KChampionPromotionDecision(
            slot_id=slot_id,
            candidates=items,
            winner=None,
            no_champion_reason="no eligible K-slot promotion candidate",
        )
    remaining = eligible
    for index in range(4):
        maximum = max(item.evidence.score_tuple[index] for item in remaining)
        remaining = [
            item
            for item in remaining
            if item.evidence.score_tuple[index] >= maximum - K_SLOT_SCORE_TOLERANCE
        ]
        if len(remaining) == 1:
            break
    winner = min(
        remaining,
        key=lambda item: (
            item.selection.feature_order_hash,
            item.selection.candidate_identity,
        ),
    )
    return KChampionPromotionDecision(slot_id=slot_id, candidates=items, winner=winner)


__all__ = [
    "K_SLOT_COMPARISON_DOMAIN",
    "K_SLOT_PROMOTION_SCORE_VERSION",
    "K_SLOT_SCORE_TOLERANCE",
    "LEGAL_MODEL_FAMILIES",
    "MIN_VALID_FOLD_COUNT",
    "MIN_VALID_FOLD_RATE",
    "WITHIN_K_COMPARISON_DOMAIN",
    "KChampionPromotionCandidate",
    "KChampionPromotionDecision",
    "KChampionPromotionEvidence",
    "KChampionSelection",
    "feature_order_hash",
    "rank_k_slot_candidates",
]
