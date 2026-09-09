"""Cluster-local and global ranking of distribution-sensitive feature winners."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from market_regime_engine.feature_discovery.contracts import (
    ClusterSolution,
    ClusterWinner,
    FeatureRegimeScore,
    RankedWinnerSet,
)

_TOLERANCE = 1.0e-12
_RANKING_STAGES: tuple[tuple[str, bool], ...] = (
    ("state_information_ratio", True),
    ("eta_squared", True),
    ("coverage", True),
    ("observation_count", True),
    ("canonical_ordinal", False),
)


@dataclass(frozen=True, slots=True)
class WinnerTierEvidence:
    """One globally anchored tolerance tier considered at a ranking stage."""

    stage: str
    candidate_features: tuple[str, ...]
    anchor_value: float
    higher_is_better: bool

    def __post_init__(self) -> None:
        if not self.stage or not self.candidate_features:
            raise ValueError("winner tier evidence requires a stage and candidates")
        if len(set(self.candidate_features)) != len(self.candidate_features):
            raise ValueError("winner tier candidates must be unique")
        if not isfinite(self.anchor_value):
            raise ValueError("winner tier anchor must be finite")


@dataclass(frozen=True, slots=True)
class ClusterWinnerDecision:
    """Complete candidate, gate and anchored-tier evidence for one cluster."""

    cluster_id: str
    candidate_features: tuple[str, ...]
    eligible_candidate_features: tuple[str, ...]
    tiers: tuple[WinnerTierEvidence, ...]
    winner_feature: str

    def __post_init__(self) -> None:
        if not self.cluster_id.startswith("cluster_"):
            raise ValueError("cluster winner decision has an invalid cluster ID")
        if not self.candidate_features or len(set(self.candidate_features)) != len(
            self.candidate_features
        ):
            raise ValueError("cluster winner decision candidates must be unique")
        if not self.eligible_candidate_features:
            raise ValueError("cluster winner decision requires an eligible candidate")
        if any(
            feature not in self.candidate_features for feature in self.eligible_candidate_features
        ):
            raise ValueError("eligible cluster winner candidates must be cluster members")
        if self.winner_feature not in self.eligible_candidate_features:
            raise ValueError("cluster winner must be an eligible cluster candidate")
        if not self.tiers:
            raise ValueError("cluster winner decision requires tier evidence")


@dataclass(frozen=True, slots=True)
class RegimeWinnerSelection:
    """Ranked winner set plus every cluster-local and global decision."""

    winner_set: RankedWinnerSet
    cluster_decisions: tuple[ClusterWinnerDecision, ...]
    global_tiers: tuple[WinnerTierEvidence, ...]

    def __post_init__(self) -> None:
        if not self.cluster_decisions or not self.global_tiers:
            raise ValueError("regime winner selection requires complete decision evidence")
        decision_ids = tuple(item.cluster_id for item in self.cluster_decisions)
        winner_ids = tuple(item.cluster_id for item in self.winner_set.winners)
        if set(decision_ids) != set(winner_ids) or len(decision_ids) != len(set(decision_ids)):
            raise ValueError("cluster decision evidence must cover exactly the winner set")

    @property
    def winners(self) -> tuple[ClusterWinner, ...]:
        return self.winner_set.winners

    @property
    def ranked_features(self) -> tuple[str, ...]:
        return self.winner_set.ranked_features


def _value(score: FeatureRegimeScore, field_name: str) -> float:
    value = getattr(score, field_name)
    if value is None or not isfinite(value):
        raise ValueError(f"eligible feature score has no finite {field_name}")
    return float(value)


def _anchored_tiers(
    candidates: tuple[FeatureRegimeScore, ...],
) -> tuple[
    tuple[tuple[FeatureRegimeScore, ...], ...],
    tuple[WinnerTierEvidence, ...],
]:
    groups: tuple[tuple[FeatureRegimeScore, ...], ...] = (candidates,)
    evidence: list[WinnerTierEvidence] = []
    for field_name, higher_is_better in _RANKING_STAGES:
        next_groups: list[tuple[FeatureRegimeScore, ...]] = []
        for group in groups:
            remaining = list(group)
            while remaining:
                values = [_value(score, field_name) for score in remaining]
                anchor = max(values) if higher_is_better else min(values)
                tied = tuple(
                    score
                    for score in remaining
                    if (
                        _value(score, field_name) >= anchor - _TOLERANCE
                        if higher_is_better
                        else _value(score, field_name) <= anchor + _TOLERANCE
                    )
                )
                evidence.append(
                    WinnerTierEvidence(
                        stage=field_name,
                        candidate_features=tuple(score.feature_name for score in tied),
                        anchor_value=anchor,
                        higher_is_better=higher_is_better,
                    )
                )
                next_groups.append(tied)
                tied_names = {score.feature_name for score in tied}
                remaining = [score for score in remaining if score.feature_name not in tied_names]
        groups = tuple(next_groups)
    return groups, tuple(evidence)


def _rank_scores(
    candidates: tuple[FeatureRegimeScore, ...],
) -> tuple[tuple[FeatureRegimeScore, ...], tuple[WinnerTierEvidence, ...]]:
    groups, evidence = _anchored_tiers(candidates)
    ranked = tuple(
        score
        for group in groups
        for score in sorted(group, key=lambda item: (item.canonical_ordinal, item.feature_name))
    )
    return ranked, evidence


def _validate_inputs(
    solution: ClusterSolution,
    scores: tuple[FeatureRegimeScore, ...],
) -> dict[str, FeatureRegimeScore]:
    if not isinstance(solution, ClusterSolution):
        raise TypeError("winner selection requires a ClusterSolution")
    if not scores:
        raise ValueError("winner selection requires feature scores")
    by_name = {score.feature_name: score for score in scores}
    if len(by_name) != len(scores):
        raise ValueError("feature scores must have unique feature names")
    members = tuple(feature for _, features in solution.memberships for feature in features)
    if set(by_name) != set(members):
        raise ValueError("feature scores must cover exactly every cluster member")
    return by_name


def select_cluster_winners(
    solution: ClusterSolution,
    scores: tuple[FeatureRegimeScore, ...],
) -> RegimeWinnerSelection:
    """Select one eligible regime feature per cluster, then rank winners globally."""

    by_name = _validate_inputs(solution, scores)
    decisions: list[ClusterWinnerDecision] = []
    cluster_winners: list[ClusterWinner] = []
    for cluster_id, members in solution.memberships:
        candidates = tuple(by_name[feature] for feature in members)
        eligible = tuple(score for score in candidates if score.eligible)
        if not eligible:
            raise ValueError(f"{cluster_id} has no eligible regime feature score")
        ranked, tiers = _rank_scores(eligible)
        winner = ranked[0]
        decisions.append(
            ClusterWinnerDecision(
                cluster_id=cluster_id,
                candidate_features=members,
                eligible_candidate_features=tuple(score.feature_name for score in eligible),
                tiers=tiers,
                winner_feature=winner.feature_name,
            )
        )
        cluster_winners.append(
            ClusterWinner(
                cluster_id=cluster_id,
                member_features=members,
                winner_feature=winner.feature_name,
                winner_score=winner,
            )
        )

    winner_scores = tuple(item.winner_score for item in cluster_winners)
    ranked_scores, global_tiers = _rank_scores(winner_scores)
    winner_by_name = {item.winner_feature: item for item in cluster_winners}
    ranked_winners = tuple(winner_by_name[score.feature_name] for score in ranked_scores)
    return RegimeWinnerSelection(
        winner_set=RankedWinnerSet(
            winners=ranked_winners,
            ranked_features=tuple(score.feature_name for score in ranked_scores),
        ),
        cluster_decisions=tuple(decisions),
        global_tiers=global_tiers,
    )


select_regime_winners = select_cluster_winners
rank_cluster_winners = select_cluster_winners


__all__ = [
    "ClusterWinnerDecision",
    "RegimeWinnerSelection",
    "WinnerTierEvidence",
    "rank_cluster_winners",
    "select_cluster_winners",
    "select_regime_winners",
]
