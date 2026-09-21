"""TRAIN-only family near-duplicate pruning for the canonical selection path."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import fsum, isfinite, sqrt

from market_regime_engine.feature_discovery.feature_roles import (
    FeatureRoleContract,
    FeatureSelectionProfile,
    FeatureStage,
)


@dataclass(frozen=True, slots=True)
class FamilyPairEvidence:
    family: str
    leader: str
    duplicate: str
    full_absolute_pearson: float
    subwindow_absolute_pearsons: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FamilyNearDuplicateResult:
    retained_features: tuple[str, ...]
    removed_features: tuple[str, ...]
    evidence: tuple[FamilyPairEvidence, ...]
    profile_hash: str

    @property
    def result_hash(self) -> str:
        payload = {
            "profile_hash": self.profile_hash,
            "retained_features": self.retained_features,
            "removed_features": self.removed_features,
            "evidence": [
                {
                    "family": item.family,
                    "leader": item.leader,
                    "duplicate": item.duplicate,
                    "full_absolute_pearson": item.full_absolute_pearson,
                    "subwindow_absolute_pearsons": item.subwindow_absolute_pearsons,
                }
                for item in self.evidence
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(encoded).hexdigest()


def _absolute_pearson(
    left: Sequence[float | None], right: Sequence[float | None]
) -> tuple[float, int] | None:
    if len(left) != len(right):
        raise ValueError("family feature vectors must have equal row counts")
    pairs = tuple(
        (float(a), float(b))
        for a, b in zip(left, right, strict=True)
        if a is not None and b is not None
    )
    if len(pairs) < 2 or any(not isfinite(value) for pair in pairs for value in pair):
        return None
    left_values = tuple(pair[0] for pair in pairs)
    right_values = tuple(pair[1] for pair in pairs)
    left_mean = fsum(left_values) / len(left_values)
    right_mean = fsum(right_values) / len(right_values)
    left_centered = tuple(value - left_mean for value in left_values)
    right_centered = tuple(value - right_mean for value in right_values)
    left_ss = fsum(value * value for value in left_centered)
    right_ss = fsum(value * value for value in right_centered)
    if left_ss <= 0.0 or right_ss <= 0.0:
        return None
    correlation = fsum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered, strict=True)
    ) / sqrt(left_ss * right_ss)
    if not isfinite(correlation):
        return None
    return abs(min(1.0, max(-1.0, correlation))), len(pairs)


def _contiguous_thirds(row_count: int) -> tuple[tuple[int, int], ...]:
    if row_count < 1:
        raise ValueError("family vectors cannot be empty")
    quotient, remainder = divmod(row_count, 3)
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(3):
        size = quotient + (1 if index < remainder else 0)
        bounds.append((start, start + size))
        start += size
    return tuple(bounds)


def _stable_duplicate(
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
    if full[0] < profile.family_near_duplicate_abs_threshold or any(
        value < profile.family_near_duplicate_subwindow_abs_threshold for value in subwindows
    ):
        return None
    return full[0], tuple(subwindows)


def prune_family_near_duplicates(
    feature_values: Mapping[str, Sequence[float | None]],
    contract: FeatureRoleContract,
    *,
    profile: FeatureSelectionProfile | None = None,
) -> FamilyNearDuplicateResult:
    """Retain the earliest stable leader within each transformation family.

    The function accepts only already materialized TRAIN vectors.  It does not
    fill missing values, inspect TEST rows, score HMM usefulness, or compare
    features across families.  A pair lacking complete support is retained.
    """

    names = tuple(feature_values)
    if not names:
        raise ValueError("family near-duplicate pruning requires feature vectors")
    contract.validate_stage_features(
        # Family PCA is the semantic boundary immediately after this stage.
        # The input itself is the transformation side of that boundary.
        stage=FeatureStage.FAMILY_PCA,
        feature_names=names,
    )
    row_counts = {len(values) for values in feature_values.values()}
    if len(row_counts) != 1 or not row_counts or next(iter(row_counts)) < 1:
        raise ValueError("family vectors must have one non-empty common row count")
    resolved_profile = contract.profile if profile is None else profile
    if resolved_profile.profile_hash != contract.profile.profile_hash:
        raise ValueError("family pruning profile must match the role contract profile")

    by_family: dict[str, list[str]] = {}
    for name in names:
        family = contract.assignment(name).family
        if family is None:
            raise ValueError("family pruning inputs must have a transformation family")
        by_family.setdefault(family, []).append(name)

    retained: list[str] = []
    removed: list[str] = []
    evidence: list[FamilyPairEvidence] = []
    for name in names:
        if name in removed:
            continue
        retained.append(name)
        family = contract.assignment(name).family
        assert family is not None
        for candidate in by_family[family]:
            if candidate == name or candidate in removed or candidate in retained:
                continue
            duplicate = _stable_duplicate(
                feature_values[name], feature_values[candidate], resolved_profile
            )
            if duplicate is None:
                continue
            removed.append(candidate)
            evidence.append(
                FamilyPairEvidence(
                    family=family,
                    leader=name,
                    duplicate=candidate,
                    full_absolute_pearson=duplicate[0],
                    subwindow_absolute_pearsons=duplicate[1],
                )
            )

    return FamilyNearDuplicateResult(
        retained_features=tuple(retained),
        removed_features=tuple(removed),
        evidence=tuple(evidence),
        profile_hash=resolved_profile.profile_hash,
    )


__all__ = [
    "FamilyNearDuplicateResult",
    "FamilyPairEvidence",
    "prune_family_near_duplicates",
]
