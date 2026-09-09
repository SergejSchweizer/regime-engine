"""Deterministic temporary prototypes for the global v4 feature clusters."""

from __future__ import annotations

from collections.abc import Mapping
from math import fsum, isfinite

from market_regime_engine.feature_discovery.contracts import (
    ClusterSolution,
    DistanceMatrixResult,
    PrototypeSet,
)

PROTOTYPE_TIE_TOLERANCE = 1.0e-12


def _canonical_ordinals(
    distance: DistanceMatrixResult,
    clusters: ClusterSolution,
) -> Mapping[str, int]:
    feature_order = distance.feature_order
    if clusters.candidate_count != len(feature_order):
        raise ValueError("cluster and distance candidate counts differ")
    if clusters.feature_ordinals:
        if tuple(name for name, _ in clusters.feature_ordinals) != feature_order:
            raise ValueError("cluster feature ordinals must follow distance feature order")
        ordinals = dict(clusters.feature_ordinals)
    else:
        ordinals = {name: index + 1 for index, name in enumerate(feature_order)}
    if tuple(sorted(ordinals, key=ordinals.__getitem__)) != feature_order:
        raise ValueError("cluster feature ordinals must be canonical")
    return ordinals


def _mean_distance(
    feature: str,
    members: tuple[str, ...],
    positions: Mapping[str, int],
    distances: tuple[tuple[float, ...], ...],
) -> float:
    if len(members) == 1:
        return 0.0
    position = positions[feature]
    values = tuple(distances[position][positions[other]] for other in members if other != feature)
    mean = fsum(values) / len(values)
    if not isfinite(mean):
        raise ValueError("prototype mean distances must be finite")
    return mean


def select_temporary_prototypes(
    clusters: ClusterSolution,
    distance: DistanceMatrixResult,
) -> PrototypeSet:
    """Select one correlation medoid per cluster for initialization only.

    For a non-singleton cluster, every member receives its arithmetic mean
    distance to the *other* members.  The smallest mean wins; values within
    ``1e-12`` are one anchored tie tier and the smallest canonical ordinal
    wins.  ``PrototypeSet.mean_distances`` retains every candidate mean in
    cluster order, so downstream code can audit both the winner and ties.

    This function has no model, target, semantic-label, or economic input by
    design.  The returned prototypes are initialization-only and must not
    receive final feature-selection privilege.
    """

    if not isinstance(clusters, ClusterSolution):
        raise TypeError("temporary prototype selection requires a ClusterSolution")
    if not isinstance(distance, DistanceMatrixResult):
        raise TypeError("temporary prototype selection requires a DistanceMatrixResult")
    ordinals = _canonical_ordinals(distance, clusters)
    feature_order = distance.feature_order
    positions = {name: index for index, name in enumerate(feature_order)}
    matrix = distance.distances
    if len(matrix) != len(feature_order) or any(len(row) != len(feature_order) for row in matrix):
        raise ValueError("distance matrix must match the cluster feature universe")

    expected_cluster_ids = tuple(f"cluster_{index:03d}" for index in range(clusters.selected_count))
    if tuple(cluster_id for cluster_id, _ in clusters.memberships) != expected_cluster_ids:
        raise ValueError("cluster memberships must be in canonical cluster order")

    seen: set[str] = set()
    selected: list[str] = []
    evidence: list[tuple[str, float]] = []
    for cluster_id, raw_members in clusters.memberships:
        if not raw_members or any(member not in positions for member in raw_members):
            raise ValueError(f"{cluster_id} contains an unknown feature")
        members = tuple(sorted(raw_members, key=ordinals.__getitem__))
        if len(set(members)) != len(members) or seen.intersection(members):
            raise ValueError("cluster memberships must cover each feature exactly once")
        seen.update(members)
        candidate_means = tuple(
            (member, _mean_distance(member, members, positions, matrix)) for member in members
        )
        evidence.extend(candidate_means)
        minimum = min(mean for _member, mean in candidate_means)
        tied = tuple(
            member
            for member, mean in candidate_means
            if abs(mean - minimum) <= PROTOTYPE_TIE_TOLERANCE
        )
        selected.append(min(tied, key=ordinals.__getitem__))

    if seen != set(feature_order):
        raise ValueError("cluster memberships must cover the distance feature universe")
    return PrototypeSet(
        cluster_ids=expected_cluster_ids,
        prototypes=tuple(selected),
        mean_distances=tuple(evidence),
    )


select_prototypes = select_temporary_prototypes
build_temporary_prototypes = select_temporary_prototypes


__all__ = [
    "PROTOTYPE_TIE_TOLERANCE",
    "build_temporary_prototypes",
    "select_prototypes",
    "select_temporary_prototypes",
]
