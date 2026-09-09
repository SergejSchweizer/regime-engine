from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from sklearn.metrics import silhouette_samples  # type: ignore[import-untyped]

import market_regime_engine.feature_discovery.clustering as clustering
from market_regime_engine.feature_discovery.clustering import select_global_clusters
from market_regime_engine.feature_discovery.contracts import DistanceMatrixResult


def _distance(
    matrix: tuple[tuple[float, ...], ...],
    names: tuple[str, ...] | None = None,
) -> DistanceMatrixResult:
    order = names or tuple(f"f{index}" for index in range(len(matrix)))
    size = len(order)
    support = tuple(tuple(504 for _ in range(size)) for _ in range(size))
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


def test_single_hierarchy_contains_every_cut_and_matches_silhouette_oracle() -> None:
    distance = _four_feature_distance()
    solution = select_global_clusters(distance)

    assert solution.candidate_count == 4
    assert tuple(count for count, _ in solution.silhouette_curve) == (2, 3)
    assert len(solution.merge_tree) == 3
    assert tuple(count for count, _ in solution.candidate_memberships) == (2, 3)
    assert solution.selected_count == 2
    assert solution.selected_silhouette > 0.0
    assert solution.memberships == (
        ("cluster_000", ("f0", "f1")),
        ("cluster_001", ("f2", "f3")),
    )
    labels = np.array([0, 0, 1, 1])
    expected = float(
        np.mean(silhouette_samples(np.asarray(distance.distances), labels, metric="precomputed"))
    )
    assert dict(solution.silhouette_curve)[2] == pytest.approx(expected)


def test_cluster_ids_follow_minimum_supplied_canonical_ordinal() -> None:
    solution = select_global_clusters(
        _four_feature_distance(),
        (("f0", 10), ("f1", 20), ("f2", 30), ("f3", 40)),
    )
    assert solution.feature_ordinals == (("f0", 10), ("f1", 20), ("f2", 30), ("f3", 40))
    assert solution.memberships[0][0] == "cluster_000"
    assert solution.memberships[0][1] == ("f0", "f1")
    mapping_solution = select_global_clusters(
        _four_feature_distance(),
        {"f0": 10, "f1": 20, "f2": 30, "f3": 40},
    )
    assert mapping_solution.solution_hash == solution.solution_hash


def test_equal_merge_ties_and_repeated_runs_are_byte_deterministic() -> None:
    distance = _four_feature_distance()
    first = select_global_clusters(distance)
    second = select_global_clusters(distance)
    assert first.solution_hash == second.solution_hash
    assert first.merge_tree == second.merge_tree
    assert first.candidate_memberships == second.candidate_memberships


def test_singleton_silhouette_is_exactly_zero() -> None:
    solution = select_global_clusters(
        _distance(
            (
                (0.0, 0.1, 0.9),
                (0.1, 0.0, 0.9),
                (0.9, 0.9, 0.0),
            )
        )
    )
    assert solution.singleton_count == 1
    assert solution.selected_silhouette > 0.0


def test_anchored_silhouette_tie_prefers_smaller_cluster_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(clustering, "_silhouette", lambda *_args: 0.5)
    solution = select_global_clusters(_four_feature_distance())
    assert solution.selected_count == 2
    assert solution.silhouette_curve == ((2, 0.5), (3, 0.5))


def test_invalid_inputs_and_nonpositive_best_silhouette_fail_closed() -> None:
    with pytest.raises(TypeError, match="DistanceMatrixResult"):
        select_global_clusters(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least three"):
        select_global_clusters(
            _distance(
                (
                    (0.0, 0.5),
                    (0.5, 0.0),
                )
            )
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        select_global_clusters(
            _four_feature_distance(), (("f0", 2), ("f1", 1), ("f2", 3), ("f3", 4))
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        select_global_clusters(
            _four_feature_distance(), (("f0", 1), ("f1", 2), ("f2", 2), ("f3", 3))
        )
    with pytest.raises(ValueError, match="follow canonical"):
        select_global_clusters(
            _four_feature_distance(), (("f1", 1), ("f0", 2), ("f2", 3), ("f3", 4))
        )
    with pytest.raises(ValueError, match="positive integers"):
        select_global_clusters(
            _four_feature_distance(), (("f0", 0), ("f1", 1), ("f2", 2), ("f3", 3))
        )
    zero_distance = _distance(
        (
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
        )
    )
    with pytest.raises(ValueError, match="strictly positive"):
        select_global_clusters(zero_distance)


def test_malformed_distance_matrices_fail_closed() -> None:
    malformed = object.__new__(DistanceMatrixResult)
    object.__setattr__(malformed, "feature_order", ("f0", "f1", "f2"))
    object.__setattr__(malformed, "distances", ((0.0, 0.5), (0.5, 0.0)))
    with pytest.raises(ValueError, match="finite and square"):
        select_global_clusters(malformed)
    object.__setattr__(
        malformed,
        "distances",
        (
            (0.0, 0.5, 0.2),
            (0.4, 0.0, 0.5),
            (0.2, 0.5, 0.0),
        ),
    )
    with pytest.raises(ValueError, match="symmetric"):
        select_global_clusters(malformed)


def test_private_structural_guards_cover_backend_and_membership_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHierarchy:
        def __init__(self, children: object, distances: object) -> None:
            self.children_ = children
            self.distances_ = distances

        def fit(self, _values: np.ndarray) -> FakeHierarchy:
            return self

    matrix = np.zeros((3, 3), dtype=np.float64)
    monkeypatch.setattr(
        clustering,
        "AgglomerativeClustering",
        lambda **_kwargs: FakeHierarchy(np.zeros((1, 2)), np.array([0.1])),
    )
    with pytest.raises(ValueError, match="full merge tree"):
        clustering._full_hierarchy(matrix)
    monkeypatch.setattr(
        clustering,
        "AgglomerativeClustering",
        lambda **_kwargs: FakeHierarchy(np.zeros((2, 2)), np.array([0.1])),
    )
    with pytest.raises(ValueError, match="distances are invalid"):
        clustering._full_hierarchy(matrix)
    monkeypatch.setattr(
        clustering,
        "AgglomerativeClustering",
        lambda **_kwargs: FakeHierarchy(
            np.array(((0, 1), (9, 2))),
            np.array((0.1, 0.2)),
        ),
    )
    with pytest.raises(ValueError, match="child index"):
        clustering._full_hierarchy(matrix)
    monkeypatch.setattr(
        clustering,
        "AgglomerativeClustering",
        lambda **_kwargs: FakeHierarchy(
            np.array(((0, 1), (0, 2))),
            np.array((0.1, 0.2)),
        ),
    )
    with pytest.raises(ValueError, match="sizes are inconsistent"):
        clustering._full_hierarchy(matrix)

    with pytest.raises(ValueError, match="cover every feature"):
        clustering._labels(
            ("f0", "f1", "f2"),
            (("cluster_000", ("f0",)),),
        )
    monkeypatch.setattr(
        clustering,
        "silhouette_samples",
        lambda *_args, **_kwargs: np.array([np.nan, 0.0, 0.0]),
    )
    with pytest.raises(ValueError, match="silhouette samples"):
        clustering._silhouette(
            matrix,
            np.array((0, 0, 1)),
            (("cluster_000", ("f0", "f1")), ("cluster_001", ("f2",))),
        )
    monkeypatch.setattr(
        clustering,
        "silhouette_samples",
        lambda *_args, **_kwargs: np.array([2.0, 2.0, 2.0]),
    )
    with pytest.raises(ValueError, match="mean silhouette"):
        clustering._silhouette(
            matrix,
            np.array((0, 0, 1)),
            (("cluster_000", ("f0", "f1")), ("cluster_001", ("f2",))),
        )


def test_distance_matrix_is_copied_before_hierarchy_fit() -> None:
    distance = _four_feature_distance()
    solution = select_global_clusters(distance)
    changed = replace(
        distance,
        distances=(
            (0.0, 0.8, 0.2, 0.2),
            (0.8, 0.0, 0.2, 0.2),
            (0.2, 0.2, 0.0, 0.8),
            (0.2, 0.2, 0.8, 0.0),
        ),
    )
    assert solution.solution_hash != select_global_clusters(changed).solution_hash
