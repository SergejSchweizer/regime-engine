from __future__ import annotations

from dataclasses import replace

import pytest

from market_regime_engine.feature_discovery.clustering import select_global_clusters
from market_regime_engine.feature_discovery.contracts import (
    ClusterSolution,
    DistanceMatrixResult,
)
from market_regime_engine.feature_discovery.prototypes import (
    select_temporary_prototypes,
)


def _distance(
    matrix: tuple[tuple[float, ...], ...],
    names: tuple[str, ...] | None = None,
) -> DistanceMatrixResult:
    order = names or tuple(f"f{index}" for index in range(len(matrix)))
    support = tuple(tuple(504 for _ in order) for _ in order)
    return DistanceMatrixResult(order, matrix, support)


def _four_feature_distance() -> DistanceMatrixResult:
    return _distance(
        (
            (0.0, 0.1, 0.9, 0.9),
            (0.1, 0.0, 0.9, 0.9),
            (0.9, 0.9, 0.0, 0.1),
            (0.9, 0.9, 0.1, 0.0),
        )
    )


def test_independent_medoid_arithmetic_and_all_candidate_evidence() -> None:
    distance = _four_feature_distance()
    clusters = select_global_clusters(distance)

    prototypes = select_temporary_prototypes(clusters, distance)

    assert prototypes.cluster_ids == ("cluster_000", "cluster_001")
    assert prototypes.prototypes == ("f0", "f2")
    assert prototypes.mean_distances == (
        ("f0", 0.1),
        ("f1", 0.1),
        ("f2", 0.1),
        ("f3", 0.1),
    )
    assert prototypes.temporary is True


def test_singleton_is_its_own_prototype_with_zero_mean_distance() -> None:
    distance = _distance(
        (
            (0.0, 0.1, 0.9),
            (0.1, 0.0, 0.9),
            (0.9, 0.9, 0.0),
        )
    )
    clusters = select_global_clusters(distance)

    prototypes = select_temporary_prototypes(clusters, distance)

    assert prototypes.prototypes == ("f0", "f2")
    assert ("f2", 0.0) in prototypes.mean_distances


def test_anchored_tie_uses_smallest_canonical_ordinal() -> None:
    distance = _distance(
        (
            (0.0, 0.1, 0.3, 0.9),
            (0.1, 0.0, 0.3000000000005, 0.9),
            (0.3, 0.3000000000005, 0.0, 0.9),
            (0.9, 0.9, 0.9, 0.0),
        )
    )
    clusters = ClusterSolution(
        candidate_count=4,
        selected_count=2,
        silhouette_curve=((2, 0.5), (3, 0.4)),
        memberships=(("cluster_000", ("f0", "f1", "f2")), ("cluster_001", ("f3",))),
        selected_silhouette=0.5,
        singleton_count=1,
        feature_ordinals=(("f0", 10), ("f1", 20), ("f2", 30), ("f3", 40)),
    )

    prototypes = select_temporary_prototypes(clusters, distance)

    assert prototypes.prototypes == ("f0", "f3")


def test_ordinals_are_identity_and_input_order_does_not_change_bytes() -> None:
    distance = _four_feature_distance()
    clusters = select_global_clusters(
        distance,
        (("f0", 10), ("f1", 20), ("f2", 30), ("f3", 40)),
    )
    first = select_temporary_prototypes(clusters, distance)
    second = select_temporary_prototypes(clusters, replace(distance))

    assert first == second


def test_invalid_types_mismatched_universe_and_malformed_memberships_fail_closed() -> None:
    distance = _four_feature_distance()
    clusters = select_global_clusters(distance)
    with pytest.raises(TypeError, match="ClusterSolution"):
        select_temporary_prototypes(object(), distance)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DistanceMatrixResult"):
        select_temporary_prototypes(clusters, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="candidate counts"):
        select_temporary_prototypes(
            select_global_clusters(
                _distance(
                    (
                        (0.0, 0.1, 0.9),
                        (0.1, 0.0, 0.9),
                        (0.9, 0.9, 0.0),
                    )
                )
            ),
            distance,
        )

    malformed = object.__new__(ClusterSolution)
    object.__setattr__(malformed, "candidate_count", 4)
    object.__setattr__(malformed, "selected_count", 2)
    object.__setattr__(malformed, "feature_ordinals", ())
    with pytest.raises(ValueError, match="exactly once"):
        object.__setattr__(
            malformed,
            "memberships",
            (("cluster_000", ("f0", "f1")), ("cluster_001", ("f1", "f2"))),
        )
        select_temporary_prototypes(malformed, distance)  # type: ignore[arg-type]


def test_private_guards_reject_malformed_contract_objects() -> None:
    distance = _four_feature_distance()
    clusters = select_global_clusters(distance)

    malformed = object.__new__(ClusterSolution)
    object.__setattr__(malformed, "candidate_count", 4)
    object.__setattr__(malformed, "selected_count", 2)
    object.__setattr__(malformed, "memberships", clusters.memberships)
    object.__setattr__(malformed, "feature_ordinals", (("f1", 1), ("f0", 2), ("f2", 3), ("f3", 4)))
    with pytest.raises(ValueError, match="follow distance"):
        select_temporary_prototypes(malformed, distance)  # type: ignore[arg-type]

    object.__setattr__(malformed, "feature_ordinals", (("f0", 2), ("f1", 1), ("f2", 3), ("f3", 4)))
    with pytest.raises(ValueError, match="must be canonical"):
        select_temporary_prototypes(malformed, distance)  # type: ignore[arg-type]

    object.__setattr__(malformed, "feature_ordinals", ())
    object.__setattr__(
        malformed,
        "memberships",
        (("cluster_000", ("f0", "unknown")), ("cluster_001", ("f2", "f3"))),
    )
    with pytest.raises(ValueError, match="unknown feature"):
        select_temporary_prototypes(malformed, distance)  # type: ignore[arg-type]

    object.__setattr__(
        malformed, "memberships", (("cluster_000", ("f0", "f1")), ("cluster_001", ("f2",)))
    )
    with pytest.raises(ValueError, match="distance feature universe"):
        select_temporary_prototypes(malformed, distance)  # type: ignore[arg-type]

    malformed_distance = object.__new__(DistanceMatrixResult)
    object.__setattr__(malformed_distance, "feature_order", distance.feature_order)
    object.__setattr__(malformed_distance, "distances", ((0.0,),))
    with pytest.raises(ValueError, match="match the cluster"):
        select_temporary_prototypes(clusters, malformed_distance)  # type: ignore[arg-type]

    object.__setattr__(
        malformed_distance,
        "distances",
        (
            (0.0, 0.1, 0.9, 0.9),
            (0.1, 0.0, 0.9, 0.9),
            (0.9, 0.9, 0.0, float("nan")),
            (0.9, 0.9, float("nan"), 0.0),
        ),
    )
    with pytest.raises(ValueError, match="mean distances"):
        select_temporary_prototypes(clusters, malformed_distance)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="canonical cluster order"):
        object.__setattr__(
            malformed,
            "memberships",
            (("cluster_001", ("f0", "f1")), ("cluster_000", ("f2", "f3"))),
        )
        select_temporary_prototypes(malformed, distance)  # type: ignore[arg-type]
