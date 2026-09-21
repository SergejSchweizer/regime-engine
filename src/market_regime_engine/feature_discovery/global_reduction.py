"""TRAIN-only global stable absolute-Pearson redundancy pruning."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

from market_regime_engine.feature_discovery.family_reduction import (
    _absolute_pearson,
    _contiguous_thirds,
)
from market_regime_engine.feature_discovery.feature_roles import (
    FeatureRoleContract,
    FeatureSelectionProfile,
    FeatureStage,
)


@dataclass(frozen=True, slots=True)
class GlobalCorrelationEvidence:
    leader: str
    removed: str
    full_absolute_pearson: float
    subwindow_absolute_pearsons: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class GlobalCorrelationResult:
    representatives: tuple[str, ...]
    removed_features: tuple[str, ...]
    evidence: tuple[GlobalCorrelationEvidence, ...]
    profile_hash: str

    @property
    def result_hash(self) -> str:
        payload = {
            "profile_hash": self.profile_hash,
            "representatives": self.representatives,
            "removed_features": self.removed_features,
            "evidence": [
                {
                    "leader": item.leader,
                    "removed": item.removed,
                    "full_absolute_pearson": item.full_absolute_pearson,
                    "subwindow_absolute_pearsons": item.subwindow_absolute_pearsons,
                }
                for item in self.evidence
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()


def _stable_redundancy(
    left: Sequence[float | None],
    right: Sequence[float | None],
    profile: FeatureSelectionProfile,
) -> tuple[float, tuple[float, ...]] | None:
    full = _absolute_pearson(left, right)
    if full is None or full[1] < profile.correlation_min_pair_rows:
        return None
    subwindows: list[float] = []
    for start, end in _contiguous_thirds(len(left)):
        result = _absolute_pearson(left[start:end], right[start:end])
        if result is None or result[1] < profile.correlation_min_subwindow_rows:
            return None
        subwindows.append(result[0])
    if full[0] < profile.correlation_abs_threshold or any(
        value < profile.correlation_subwindow_abs_threshold for value in subwindows
    ):
        return None
    return full[0], tuple(subwindows)


def prune_global_correlated_features(
    feature_values: Mapping[str, Sequence[float | None]],
    contract: FeatureRoleContract,
    *,
    profile: FeatureSelectionProfile | None = None,
) -> GlobalCorrelationResult:
    """Keep deterministic redundancy leaders among core and family-PC inputs."""

    names = tuple(feature_values)
    if not names:
        raise ValueError("global correlation pruning requires feature vectors")
    contract.validate_stage_features(FeatureStage.CORRELATION, names)
    row_counts = {len(values) for values in feature_values.values()}
    if len(row_counts) != 1 or not row_counts or next(iter(row_counts)) < 1:
        raise ValueError("global correlation vectors must have one non-empty common row count")
    resolved_profile = contract.profile if profile is None else profile
    if resolved_profile.profile_hash != contract.profile.profile_hash:
        raise ValueError("global correlation profile must match the role contract profile")

    representatives: list[str] = []
    removed: list[str] = []
    evidence: list[GlobalCorrelationEvidence] = []
    for name in names:
        if name in removed:
            continue
        representatives.append(name)
        for candidate in names:
            if candidate == name or candidate in removed or candidate in representatives:
                continue
            redundant = _stable_redundancy(
                feature_values[name], feature_values[candidate], resolved_profile
            )
            if redundant is None:
                continue
            removed.append(candidate)
            evidence.append(
                GlobalCorrelationEvidence(
                    leader=name,
                    removed=candidate,
                    full_absolute_pearson=redundant[0],
                    subwindow_absolute_pearsons=redundant[1],
                )
            )

    return GlobalCorrelationResult(
        representatives=tuple(representatives),
        removed_features=tuple(removed),
        evidence=tuple(evidence),
        profile_hash=resolved_profile.profile_hash,
    )


__all__ = [
    "GlobalCorrelationEvidence",
    "GlobalCorrelationResult",
    "prune_global_correlated_features",
]
