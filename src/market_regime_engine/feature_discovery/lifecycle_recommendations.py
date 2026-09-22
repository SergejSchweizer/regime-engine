"""Deterministic, approval-free feature lifecycle recommendations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path

from market_regime_engine.feature_discovery.metadata_store import (
    FeatureRegistryRow,
    FoldFeatureStat,
)

LIFECYCLE_ACTIVE = "ACTIVE"
LIFECYCLE_DEPRECATED_CANDIDATE = "DEPRECATED_CANDIDATE"
LIFECYCLE_DEPRECATED = "DEPRECATED"
LIFECYCLE_DROPPABLE = "DROPPABLE"
LIFECYCLE_STATES = (
    LIFECYCLE_ACTIVE,
    LIFECYCLE_DEPRECATED_CANDIDATE,
    LIFECYCLE_DEPRECATED,
    LIFECYCLE_DROPPABLE,
)


@dataclass(frozen=True, slots=True)
class FeatureLifecyclePolicy:
    """Versioned thresholds used by the recommendation-only report."""

    version: str = "feature_lifecycle_v1"
    minimum_eligible_folds: int = 20
    pca_credit_epsilon: float = 1.0e-12
    selection_window_folds: int = 20

    def __post_init__(self) -> None:
        if not self.version or self.version.strip() != self.version:
            raise ValueError("lifecycle policy version must be non-empty and trimmed")
        if self.minimum_eligible_folds < 1 or self.selection_window_folds < 1:
            raise ValueError("lifecycle fold thresholds must be positive")
        if not isfinite(self.pca_credit_epsilon) or self.pca_credit_epsilon < 0.0:
            raise ValueError("lifecycle PCA-credit epsilon must be finite and non-negative")

    def canonical_dict(self) -> dict[str, object]:
        return asdict(self)

    def profile_hash(self, feature_selection_profile_hash: str) -> str:
        """Return the hash that stores the lifecycle epsilon with the profile identity."""

        if len(feature_selection_profile_hash) != 64 or any(
            char not in "0123456789abcdef" for char in feature_selection_profile_hash
        ):
            raise ValueError("feature-selection profile hash must be a lowercase SHA-256")
        payload = {
            "feature_selection_profile_hash": feature_selection_profile_hash,
            "lifecycle_policy": self.canonical_dict(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureLifecycleEvidence:
    feature_name: str
    role: str
    eligible_folds: int
    recent_eligible_folds: int
    recent_selected_folds: int
    recent_representative_folds: int
    total_pca_credit: float
    recent_pca_credit: float
    current_status: str
    recommended_status: str
    automatic_transition: bool
    reason: str


@dataclass(frozen=True, slots=True)
class FeatureLifecycleReport:
    policy: FeatureLifecyclePolicy
    policy_profile_hash: str
    recommendations: tuple[FeatureLifecycleEvidence, ...]

    @property
    def report_hash(self) -> str:
        payload = {
            "policy_profile_hash": self.policy_profile_hash,
            "recommendations": [asdict(item) for item in self.recommendations],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "report_hash": self.report_hash,
            "policy": self.policy.canonical_dict(),
            "policy_profile_hash": self.policy_profile_hash,
            "recommendations": [asdict(item) for item in self.recommendations],
        }


def build_lifecycle_report(
    registry_rows: tuple[FeatureRegistryRow, ...],
    fold_rows: tuple[FoldFeatureStat, ...],
    *,
    feature_selection_profile_hash: str,
    policy: FeatureLifecyclePolicy | None = None,
) -> FeatureLifecycleReport:
    """Build recommendations solely from committed registry and fold evidence."""

    resolved_policy = policy or FeatureLifecyclePolicy()
    by_feature: dict[str, list[FoldFeatureStat]] = {}
    for row in fold_rows:
        by_feature.setdefault(row.feature_name, []).append(row)
    output: list[FeatureLifecycleEvidence] = []
    for registry in sorted(registry_rows, key=lambda item: item.feature_name):
        if registry.lifecycle_status not in LIFECYCLE_STATES:
            raise ValueError(f"unknown lifecycle status: {registry.lifecycle_status}")
        rows = sorted(by_feature.get(registry.feature_name, ()), key=lambda item: item.fold_id)
        eligible_rows = [row for row in rows if row.eligible]
        recent = list(reversed(eligible_rows[-resolved_policy.selection_window_folds :]))
        selected = sum(row.final_selection for row in recent)
        representative = sum(row.representative for row in recent)
        total_credit = sum(row.pca_credit or 0.0 for row in rows)
        recent_credit = sum(row.pca_credit or 0.0 for row in recent)
        generated = registry.role == "transformation"
        stale = (
            generated
            and len(eligible_rows) >= resolved_policy.minimum_eligible_folds
            and len(recent) >= resolved_policy.selection_window_folds
            and selected == 0
            and representative == 0
            and total_credit < resolved_policy.pca_credit_epsilon
        )
        reactivated = (
            selected > 0
            or representative > 0
            or recent_credit >= resolved_policy.pca_credit_epsilon
        )
        if not generated:
            recommended = (
                LIFECYCLE_ACTIVE
                if registry.lifecycle_status in (LIFECYCLE_DEPRECATED, LIFECYCLE_DROPPABLE)
                else registry.lifecycle_status
            )
            automatic = False
            reason = "core/raw features require explicit operator lifecycle decisions"
        elif reactivated:
            recommended = LIFECYCLE_ACTIVE
            automatic = registry.lifecycle_status != LIFECYCLE_ACTIVE
            reason = "recent selection, representation, or material PCA credit reactivates evidence"
        elif stale:
            recommended = LIFECYCLE_DEPRECATED_CANDIDATE
            automatic = registry.lifecycle_status != LIFECYCLE_DEPRECATED_CANDIDATE
            reason = (
                "20 eligible folds have no recent selection, representation, or material PCA credit"
            )
        else:
            recommended = LIFECYCLE_ACTIVE
            automatic = False
            reason = "lifecycle deprecation predicates are not yet satisfied"
        output.append(
            FeatureLifecycleEvidence(
                registry.feature_name,
                registry.role,
                len(eligible_rows),
                len(recent),
                selected,
                representative,
                total_credit,
                recent_credit,
                registry.lifecycle_status,
                recommended,
                automatic,
                reason,
            )
        )
    return FeatureLifecycleReport(
        resolved_policy,
        resolved_policy.profile_hash(feature_selection_profile_hash),
        tuple(output),
    )


def write_lifecycle_report(report: FeatureLifecycleReport, path: str | Path) -> Path:
    """Emit one deterministic local report; no database DDL/DML is performed."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.as_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    )
    return destination


__all__ = [
    "LIFECYCLE_ACTIVE",
    "LIFECYCLE_DEPRECATED",
    "LIFECYCLE_DEPRECATED_CANDIDATE",
    "LIFECYCLE_DROPPABLE",
    "LIFECYCLE_STATES",
    "FeatureLifecycleEvidence",
    "FeatureLifecyclePolicy",
    "FeatureLifecycleReport",
    "build_lifecycle_report",
    "write_lifecycle_report",
]
