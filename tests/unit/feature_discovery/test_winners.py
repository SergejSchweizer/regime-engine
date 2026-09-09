from __future__ import annotations

import pytest

from market_regime_engine.feature_discovery.contracts import (
    ClusterSolution,
    FeatureRegimeScore,
)
from market_regime_engine.feature_discovery.winners import select_cluster_winners


def solution(*, three_members: bool = False) -> ClusterSolution:
    memberships = (
        ("cluster_000", ("f0", "f1", "f2")) if three_members else ("cluster_000", ("f0", "f1")),
        ("cluster_001", ("f3",)) if three_members else ("cluster_001", ("f2", "f3")),
    )
    return ClusterSolution(
        candidate_count=4,
        selected_count=2,
        silhouette_curve=((2, 0.5), (3, 0.4)),
        memberships=memberships,
        selected_silhouette=0.5,
        singleton_count=1 if three_members else 0,
    )


def score(
    name: str,
    ordinal: int,
    ratio: float,
    *,
    eligible: bool = True,
) -> FeatureRegimeScore:
    return FeatureRegimeScore(
        feature_name=name,
        canonical_ordinal=ordinal,
        coverage=1.0,
        observation_count=126,
        bin_counts=(13, 13, 13, 13, 13, 13, 12, 12, 12, 12),
        joint_bin_state_masses=((0.05, 0.05),) * 10,
        bin_masses=(0.1,) * 10,
        state_masses=(0.5, 0.5),
        mutual_information=0.0,
        state_entropy=0.6931471805599453,
        state_information_ratio=ratio,
        state_weights=(0.5, 0.5),
        state_means=(1.0, -1.0),
        state_variances=(0.5, 0.5),
        between_variance=1.0,
        within_variance=0.5,
        eta_squared=2.0 / 3.0,
        eligible=eligible,
        exclusion_reason=None if eligible else "test exclusion",
    )


def test_prototype_has_no_privilege_and_global_winner_order_is_canonical() -> None:
    result = select_cluster_winners(
        solution(),
        (
            score("f0", 1, 0.1),
            score("f1", 2, 0.8),
            score("f2", 3, 0.1),
            score("f3", 4, 0.2),
        ),
    )

    assert result.ranked_features == ("f1", "f3")
    assert result.winner_set.winners[0].winner_feature == "f1"
    assert result.cluster_decisions[0].winner_feature == "f1"
    assert result.cluster_decisions[0].candidate_features == ("f0", "f1")
    assert result.global_tiers
    assert all(tier.candidate_features for tier in result.global_tiers)


def test_tolerance_tiers_are_globally_anchored_not_pairwise_transitive() -> None:
    result = select_cluster_winners(
        solution(three_members=True),
        (
            score("f0", 1, 0.0),
            score("f1", 2, 0.75e-12),
            score("f2", 3, 1.5e-12),
            score("f3", 4, 0.2),
        ),
    )

    # f0 must not join the f1/f2 tier through pairwise chaining.
    assert result.cluster_decisions[0].winner_feature == "f1"
    ratio_tiers = [
        tier
        for tier in result.cluster_decisions[0].tiers
        if tier.stage == "state_information_ratio"
    ]
    assert ratio_tiers[0].candidate_features == ("f1", "f2")
    assert ratio_tiers[1].candidate_features == ("f0",)


def test_each_selected_cluster_requires_an_eligible_score() -> None:
    scores = (
        score("f0", 1, 0.4, eligible=False),
        score("f1", 2, 0.3, eligible=False),
        score("f2", 3, 0.2),
        score("f3", 4, 0.1),
    )
    with pytest.raises(ValueError, match="cluster_000 has no eligible"):
        select_cluster_winners(solution(), scores)


def test_full_cluster_table_is_reduced_to_one_winner_per_cluster() -> None:
    result = select_cluster_winners(
        solution(),
        tuple(score(f"f{index}", index + 1, 0.8 - index * 0.1) for index in range(4)),
    )

    assert len(result.winner_set.winners) == 2
    assert len(set(result.ranked_features)) == 2
    assert tuple(item.cluster_id for item in result.cluster_decisions) == (
        "cluster_000",
        "cluster_001",
    )
    assert all(
        len(decision.eligible_candidate_features) >= 1 for decision in result.cluster_decisions
    )
