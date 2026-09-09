"""Deterministic global average-linkage hierarchy and M* selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

import numpy as np
from sklearn.cluster import AgglomerativeClustering  # type: ignore[import-untyped]
from sklearn.metrics import silhouette_samples  # type: ignore[import-untyped]

from market_regime_engine.feature_discovery.contracts import (
    CLUSTER_COUNT_MAX,
    CLUSTER_COUNT_MIN,
    SILHOUETTE_SINGLETON_VALUE,
    SILHOUETTE_TIE_TOLERANCE,
    ClusterSolution,
    DistanceMatrixResult,
)


def _validate_ordinals(
    feature_order: tuple[str, ...],
    feature_ordinals: Mapping[str, int] | Sequence[tuple[str, int]] | None,
) -> tuple[tuple[str, int], ...]:
    if feature_ordinals is None:
        return tuple((name, index + 1) for index, name in enumerate(feature_order))
    if isinstance(feature_ordinals, Mapping):
        supplied = tuple((name, feature_ordinals[name]) for name in feature_order)
    else:
        supplied = tuple(feature_ordinals)
        if tuple(name for name, _ in supplied) != feature_order:
            raise ValueError("feature ordinals must follow canonical distance feature order")
    if tuple(name for name, _ in supplied) != feature_order:
        raise ValueError("feature ordinals must cover distance feature order")
    ordinals = tuple(ordinal for _, ordinal in supplied)
    if any(
        not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 1
        for ordinal in ordinals
    ):
        raise ValueError("feature ordinals must be positive integers")
    if ordinals != tuple(sorted(ordinals)) or len(set(ordinals)) != len(ordinals):
        raise ValueError("feature ordinals must be strictly increasing in canonical order")
    return supplied


def _full_hierarchy(
    distances: np.ndarray,
) -> tuple[tuple[int, int, float, int], ...]:
    """Fit exactly one pinned full hierarchy and retain its merge tree."""

    model = AgglomerativeClustering(
        n_clusters=1,
        metric="precomputed",
        linkage="average",
        compute_full_tree=True,
        compute_distances=True,
    )
    model.fit(distances)
    children = np.asarray(model.children_, dtype=np.int64)
    merge_distances = np.asarray(model.distances_, dtype=np.float64)
    if children.shape != (distances.shape[0] - 1, 2):
        raise ValueError("average-linkage hierarchy did not produce a full merge tree")
    if merge_distances.shape != (distances.shape[0] - 1,) or not np.all(
        np.isfinite(merge_distances)
    ):
        raise ValueError("average-linkage hierarchy distances are invalid")

    sizes = [1] * distances.shape[0]
    tree: list[tuple[int, int, float, int]] = []
    for merge_index, (left_raw, right_raw) in enumerate(children):
        left = int(left_raw)
        right = int(right_raw)
        if left < 0 or right < 0 or left >= len(sizes) or right >= len(sizes):
            raise ValueError("average-linkage hierarchy contains an invalid child index")
        size = sizes[left] + sizes[right]
        sizes.append(size)
        tree.append((left, right, float(merge_distances[merge_index]), size))
    if len(sizes) != 2 * distances.shape[0] - 1 or sizes[-1] != distances.shape[0]:
        raise ValueError("average-linkage hierarchy sizes are inconsistent")
    return tuple(tree)


def _forest_memberships(
    feature_order: tuple[str, ...],
    merge_tree: tuple[tuple[int, int, float, int], ...],
    cluster_count: int,
    ordinals: Mapping[str, int],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    leaf_count = len(feature_order)
    active: set[int] = set(range(leaf_count))
    for merge_index, (left, right, _distance, _size) in enumerate(
        merge_tree[: leaf_count - cluster_count]
    ):
        active.remove(left)
        active.remove(right)
        active.add(leaf_count + merge_index)

    def leaves(node: int) -> tuple[str, ...]:
        if node < leaf_count:
            return (feature_order[node],)
        left, right, _distance, _size = merge_tree[node - leaf_count]
        members = leaves(left) + leaves(right)
        return tuple(sorted(members, key=ordinals.__getitem__))

    roots = sorted(active, key=lambda node: min(ordinals[name] for name in leaves(node)))
    return tuple(
        (f"cluster_{cluster_index:03d}", leaves(root)) for cluster_index, root in enumerate(roots)
    )


def _labels(
    feature_order: tuple[str, ...],
    memberships: tuple[tuple[str, tuple[str, ...]], ...],
) -> np.ndarray:
    index = {name: position for position, name in enumerate(feature_order)}
    labels = np.full(len(feature_order), -1, dtype=np.int64)
    for cluster_index, (_cluster_id, members) in enumerate(memberships):
        for name in members:
            labels[index[name]] = cluster_index
    if np.any(labels < 0) or len(set(labels.tolist())) != len(memberships):
        raise ValueError("cluster memberships do not cover every feature exactly once")
    return labels


def _silhouette(
    distances: np.ndarray,
    labels: np.ndarray,
    memberships: tuple[tuple[str, tuple[str, ...]], ...],
) -> float:
    scores = np.asarray(
        silhouette_samples(distances, labels, metric="precomputed"), dtype=np.float64
    )
    for cluster_index, (_cluster_id, members) in enumerate(memberships):
        if len(members) == 1:
            scores[labels == cluster_index] = SILHOUETTE_SINGLETON_VALUE
    if not np.all(np.isfinite(scores)):
        raise ValueError("silhouette samples must be finite")
    mean = float(np.mean(scores))
    if not isfinite(mean) or not -1.0 <= mean <= 1.0:
        raise ValueError("mean silhouette must be finite in [-1,1]")
    return mean


def select_global_clusters(
    distance: DistanceMatrixResult,
    feature_ordinals: Mapping[str, int] | Sequence[tuple[str, int]] | None = None,
) -> ClusterSolution:
    """Build one hierarchy, derive every bounded cut, and select M*.

    The input matrix is treated as an immutable canonical artifact.  No
    per-M hierarchy is fitted; all memberships and silhouettes come from the
    one merge tree.
    """

    if not isinstance(distance, DistanceMatrixResult):
        raise TypeError("global clustering requires a DistanceMatrixResult")
    feature_order = distance.feature_order
    candidate_count = len(feature_order)
    if candidate_count < 3:
        raise ValueError("global clustering requires at least three features")
    ordinals_evidence = _validate_ordinals(feature_order, feature_ordinals)
    ordinals = dict(ordinals_evidence)
    matrix = np.asarray(distance.distances, dtype=np.float64)
    if matrix.shape != (candidate_count, candidate_count) or not np.all(np.isfinite(matrix)):
        raise ValueError("distance matrix must be finite and square")
    if not np.array_equal(matrix, matrix.T) or not np.all(np.diag(matrix) == 0.0):
        raise ValueError("distance matrix must be symmetric with an exact zero diagonal")

    merge_tree = _full_hierarchy(np.array(matrix, dtype=np.float64, copy=True))
    maximum_count = min(CLUSTER_COUNT_MAX, candidate_count - 1)
    candidate_memberships: list[tuple[int, tuple[tuple[str, tuple[str, ...]], ...]]] = []
    silhouette_curve: list[tuple[int, float]] = []
    for cluster_count in range(CLUSTER_COUNT_MIN, maximum_count + 1):
        memberships = _forest_memberships(feature_order, merge_tree, cluster_count, ordinals)
        labels = _labels(feature_order, memberships)
        silhouette_curve.append((cluster_count, _silhouette(matrix, labels, memberships)))
        candidate_memberships.append((cluster_count, memberships))

    best_silhouette = max(value for _count, value in silhouette_curve)
    tied_counts = tuple(
        count
        for count, value in silhouette_curve
        if best_silhouette - value <= SILHOUETTE_TIE_TOLERANCE
    )
    selected_count = min(tied_counts)
    selected_silhouette = dict(silhouette_curve)[selected_count]
    selected_memberships = dict(candidate_memberships)[selected_count]
    if selected_silhouette <= 0.0:
        raise ValueError("best global silhouette must be strictly positive")
    singleton_count = sum(len(members) == 1 for _cluster_id, members in selected_memberships)
    return ClusterSolution(
        candidate_count=candidate_count,
        selected_count=selected_count,
        silhouette_curve=tuple(silhouette_curve),
        memberships=selected_memberships,
        selected_silhouette=selected_silhouette,
        singleton_count=singleton_count,
        merge_tree=merge_tree,
        candidate_memberships=tuple(candidate_memberships),
        feature_ordinals=ordinals_evidence,
    )


build_global_hierarchy = select_global_clusters
cluster_features = select_global_clusters


__all__ = ["build_global_hierarchy", "cluster_features", "select_global_clusters"]
