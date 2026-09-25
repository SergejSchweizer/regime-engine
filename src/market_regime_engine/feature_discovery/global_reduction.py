"""TRAIN-only global stable absolute-Pearson redundancy pruning."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from statistics import median
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np

from market_regime_engine.feature_discovery.family_reduction import (
    _absolute_pearson,
    _contiguous_thirds,
)
from market_regime_engine.feature_discovery.feature_roles import (
    FeatureRoleContract,
    FeatureSelectionProfile,
    FeatureStage,
)
from market_regime_engine.runtime.cpu import available_cpu_count
from market_regime_engine.runtime.parallel import (
    FoldParallelExecutor,
    ParallelExecutionPlan,
    ReadOnlyMatrix,
)

_Redundancy = tuple[float, int, tuple[float, ...], tuple[int, ...]]
_PairResult = tuple[int, int, _Redundancy]


@dataclass(frozen=True, slots=True)
class _CorrelationTileTask:
    matrix_path: str
    matrix_shape: tuple[int, int]
    matrix_dtype: str
    pairs: tuple[tuple[int, int], ...]
    profile: FeatureSelectionProfile


def _run_correlation_tile(task: _CorrelationTileTask) -> tuple[_PairResult, ...]:
    matrix = np.memmap(
        task.matrix_path,
        dtype=np.dtype(task.matrix_dtype),
        mode="r",
        shape=task.matrix_shape,
    )
    try:
        results: list[_PairResult] = []
        for left_index, right_index in task.pairs:
            redundant = _stable_redundancy(
                cast(Sequence[float | None], matrix[:, left_index]),
                cast(Sequence[float | None], matrix[:, right_index]),
                task.profile,
            )
            if redundant is not None:
                results.append((left_index, right_index, redundant))
        return tuple(results)
    finally:
        del matrix


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
) -> _Redundancy | None:
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
    max_workers: int | None = None,
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
    matrix = np.ascontiguousarray(
        np.asarray(
            tuple(
                tuple(np.nan if value is None else value for value in feature_values[name])
                for name in ordered_names
            ),
            dtype=np.float64,
        ).T
    )
    pair_indexes = tuple(
        (left_index, right_index)
        for left_index in range(len(ordered_names))
        for right_index in range(left_index + 1, len(ordered_names))
    )
    edges: dict[tuple[int, int], _Redundancy] = {}
    if pair_indexes:
        with (
            TemporaryDirectory(prefix="regime-global-correlation-") as directory,
            ReadOnlyMatrix.create(matrix, directory) as shared_matrix,
        ):
            tile_size = max(1, len(pair_indexes) // max(1, 4 * available_cpu_count()))
            tiles = tuple(
                pair_indexes[start : start + tile_size]
                for start in range(0, len(pair_indexes), tile_size)
            )
            plan = ParallelExecutionPlan.create(
                len(tiles),
                requested_workers=max_workers,
                shared_matrix_identity=shared_matrix.identity,
            )
            tasks = tuple(
                _CorrelationTileTask(
                    matrix_path=str(shared_matrix.path),
                    matrix_shape=(matrix.shape[0], matrix.shape[1]),
                    matrix_dtype=matrix.dtype.str,
                    pairs=tile,
                    profile=resolved_profile,
                )
                for tile in tiles
            )
            with FoldParallelExecutor[_CorrelationTileTask, tuple[_PairResult, ...]](
                plan, max_pending=max(1, plan.worker_count * 2)
            ) as executor:
                tile_results = executor.map_ordered(_run_correlation_tile, tasks)
            for tile in tile_results:
                for left_index, right_index, redundant in tile:
                    edges[(left_index, right_index)] = redundant

    # Build the graph once.  The previous implementation rebuilt every
    # candidate pair's neighborhood on every greedy iteration.  With a
    # largely uncorrelated universe that made the deterministic reduction
    # O(n^3) in Python and dominated the fold before any HMM frontier could
    # start.  Removing a leader and its direct duplicates only changes the
    # neighborhoods of those removed vertices, so maintain adjacency maps
    # incrementally instead.
    neighborhoods: dict[str, dict[str, _Redundancy]] = {
        name: {} for name in ordered_names
    }
    for (left_index, right_index), redundancy in edges.items():
        left_name = ordered_names[left_index]
        right_name = ordered_names[right_index]
        neighborhoods[left_name][right_name] = redundancy
        neighborhoods[right_name][left_name] = redundancy

    remaining = set(ordered_names)
    representatives: list[str] = []
    removed: list[str] = []
    evidence: list[GlobalCorrelationEvidence] = []
    while remaining:
        candidates = tuple(sorted(remaining, key=lambda name: canonical_order[name]))

        def leader_key(
            name: str,
        ) -> tuple[int, float, float, int, int]:
            neighborhood = neighborhoods[name]
            correlation_median = (
                median(item[0] for item in neighborhood.values()) if neighborhood else 0.0
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
        direct_duplicates = sorted(
            neighborhoods[leader].items(), key=lambda item: canonical_order[item[0]]
        )
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
        removed_names = (leader, *(name for name, _ in direct_duplicates))
        remaining.difference_update(removed_names)
        for removed_name in removed_names:
            for neighbor in tuple(neighborhoods[removed_name]):
                neighborhoods[neighbor].pop(removed_name, None)
            neighborhoods[removed_name].clear()

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
