"""TRAIN-only global stable absolute-Pearson redundancy pruning."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from statistics import median

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
    full_support_count: int
    subwindow_absolute_pearsons: tuple[float, ...]
    subwindow_support_counts: tuple[int, ...]


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
                    "full_support_count": item.full_support_count,
                    "subwindow_absolute_pearsons": item.subwindow_absolute_pearsons,
                    "subwindow_support_counts": item.subwindow_support_counts,
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
) -> tuple[float, int, tuple[float, ...], tuple[int, ...]] | None:
    full = _absolute_pearson(left, right)
    if full is None or full[1] < profile.correlation_min_pair_rows:
        return None
    subwindows: list[float] = []
    support_counts: list[int] = []
    for start, end in _contiguous_thirds(len(left)):
        result = _absolute_pearson(left[start:end], right[start:end])
        if result is None or result[1] < profile.correlation_min_subwindow_rows:
            return None
        subwindows.append(result[0])
        support_counts.append(result[1])
    if full[0] < profile.correlation_abs_threshold or any(
        value < profile.correlation_subwindow_abs_threshold for value in subwindows
    ):
        return None
    return full[0], full[1], tuple(subwindows), tuple(support_counts)


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

    canonical_order = {
        assignment.feature_name: index for index, assignment in enumerate(contract.assignments)
    }
    next_order = len(canonical_order)
    for name in sorted(name for name in names if name not in canonical_order):
        canonical_order[name] = next_order
        next_order += 1
    ordered_names = tuple(
        sorted(
            names,
            key=lambda name: (
                name.startswith("family_pc_"),
                canonical_order[name],
                name,
            ),
        )
    )
    remaining = set(ordered_names)
    representatives: list[str] = []
    removed: list[str] = []
    evidence: list[GlobalCorrelationEvidence] = []
    while remaining:
        candidates = tuple(sorted(remaining, key=lambda name: canonical_order[name]))
        neighborhoods: dict[
            str,
            list[tuple[str, tuple[float, int, tuple[float, ...], tuple[int, ...]]]],
        ] = {name: [] for name in candidates}
        for left_index, left_name in enumerate(candidates):
            for right_name in candidates[left_index + 1 :]:
                redundant = _stable_redundancy(
                    feature_values[left_name], feature_values[right_name], resolved_profile
                )
                if redundant is None:
                    continue
                neighborhoods[left_name].append((right_name, redundant))
                neighborhoods[right_name].append((left_name, redundant))

        def leader_key(
            name: str,
            neighborhoods: dict[
                str,
                list[tuple[str, tuple[float, int, tuple[float, ...], tuple[int, ...]]]],
            ] = neighborhoods,
        ) -> tuple[int, float, float, int, int]:
            neighborhood = neighborhoods[name]
            correlation_median = (
                median(item[1][0] for item in neighborhood) if neighborhood else 0.0
            )
            coverage = sum(value is not None for value in feature_values[name]) / len(
                feature_values[name]
            )
            return (
                -len(neighborhood),
                -correlation_median,
                -coverage,
                int(name.startswith("family_pc_")),
                canonical_order[name],
            )

        leader = min(candidates, key=leader_key)
        representatives.append(leader)
        direct_duplicates = sorted(neighborhoods[leader], key=lambda item: canonical_order[item[0]])
        for duplicate_name, redundant in direct_duplicates:
            removed.append(duplicate_name)
            evidence.append(
                GlobalCorrelationEvidence(
                    leader=leader,
                    removed=duplicate_name,
                    full_absolute_pearson=redundant[0],
                    full_support_count=redundant[1],
                    subwindow_absolute_pearsons=redundant[2],
                    subwindow_support_counts=redundant[3],
                )
            )
        remaining.difference_update((leader, *(name for name, _ in direct_duplicates)))

    representatives.sort(key=lambda name: canonical_order[name])
    removed.sort(key=lambda name: canonical_order[name])
    evidence.sort(key=lambda item: (canonical_order[item.leader], canonical_order[item.removed]))

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
