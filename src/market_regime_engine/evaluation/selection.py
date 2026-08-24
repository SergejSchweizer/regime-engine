"""Deterministic statistical champion selection for the exact Xetra HMM grid."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
from math import isfinite

from market_regime_engine.training.candidate_grid import (
    CANDIDATE_VALID_FOLD_RATE_GATE,
    RANKING_ABS_TOLERANCE,
    CandidateAggregate,
    CandidateGridEvaluation,
)


@dataclass(frozen=True, slots=True)
class CandidateSelectionEvidence:
    candidate_id: str
    state_count: int
    accepted: bool
    rejection_reasons: tuple[str, ...]
    rank: int | None

    def __post_init__(self) -> None:
        if self.accepted == bool(self.rejection_reasons):
            raise ValueError("accepted candidates have no rejections; rejected candidates require them")
        if self.accepted != (self.rank is not None):
            raise ValueError("only accepted candidates receive a statistical rank")
        if self.rank is not None and self.rank < 1:
            raise ValueError("candidate rank must be one-based")


@dataclass(frozen=True, slots=True)
class StatisticalChampionSelection:
    champion_candidate_id: str
    champion_state_count: int
    ranked_candidate_ids: tuple[str, ...]
    evidence: tuple[CandidateSelectionEvidence, ...]
    ranking_abs_tolerance: float = RANKING_ABS_TOLERANCE

    def __post_init__(self) -> None:
        if self.ranking_abs_tolerance != 1e-12:
            raise ValueError("statistical ranking tolerance must be exactly 1e-12")
        if not self.ranked_candidate_ids or self.ranked_candidate_ids[0] != self.champion_candidate_id:
            raise ValueError("champion must be the first accepted ranked candidate")
        champion = next(
            (item for item in self.evidence if item.candidate_id == self.champion_candidate_id),
            None,
        )
        if champion is None or champion.state_count != self.champion_state_count:
            raise ValueError("champion identity/state count must match selection evidence")


def _rejections(candidate: CandidateAggregate) -> tuple[str, ...]:
    reasons: list[str] = []
    if candidate.valid_fold_count == 0:
        reasons.append("zero valid folds")
    if candidate.valid_fold_rate < CANDIDATE_VALID_FOLD_RATE_GATE:
        reasons.append("valid-fold rate below 0.80")
    score_values = (
        candidate.oos_predictive_loglik_mean,
        candidate.oos_predictive_loglik_std,
        candidate.oos_predictive_loglik_worst_fold,
        candidate.bic_mean,
        candidate.aic_mean,
    )
    if any(value is None or not isfinite(value) for value in score_values):
        reasons.append("missing or nonfinite ranking metric")
    return tuple(reasons)


def _numeric_compare(left: float, right: float, *, higher_is_better: bool) -> int:
    if abs(left - right) <= RANKING_ABS_TOLERANCE:
        return 0
    if higher_is_better:
        return -1 if left > right else 1
    return -1 if left < right else 1


def _required(value: float | None, name: str) -> float:
    if value is None or not isfinite(value):
        raise ValueError(f"accepted candidate has invalid {name}")
    return value


def _compare(left: CandidateAggregate, right: CandidateAggregate) -> int:
    stages = (
        ("oos_predictive_loglik_mean", True),
        ("oos_predictive_loglik_std", False),
        ("oos_predictive_loglik_worst_fold", True),
        ("bic_mean", False),
        ("aic_mean", False),
    )
    for field_name, higher_is_better in stages:
        comparison = _numeric_compare(
            _required(getattr(left, field_name), field_name),
            _required(getattr(right, field_name), field_name),
            higher_is_better=higher_is_better,
        )
        if comparison:
            return comparison
    if left.state_count != right.state_count:
        return -1 if left.state_count < right.state_count else 1
    if left.candidate_id < right.candidate_id:
        return -1
    if left.candidate_id > right.candidate_id:
        return 1
    return 0


def select_statistical_champion(grid: CandidateGridEvaluation) -> StatisticalChampionSelection:
    """Apply hard gates then the exact seven-stage EVALUATION ranking."""

    rejection_map = {candidate.candidate_id: _rejections(candidate) for candidate in grid.aggregates}
    accepted = tuple(
        candidate for candidate in grid.aggregates if not rejection_map[candidate.candidate_id]
    )
    if not accepted:
        raise ValueError("no candidate passes statistical hard gates")
    ranked = tuple(sorted(accepted, key=cmp_to_key(_compare)))
    rank_map = {candidate.candidate_id: index for index, candidate in enumerate(ranked, start=1)}
    evidence = tuple(
        CandidateSelectionEvidence(
            candidate_id=candidate.candidate_id,
            state_count=candidate.state_count,
            accepted=candidate.candidate_id in rank_map,
            rejection_reasons=rejection_map[candidate.candidate_id],
            rank=rank_map.get(candidate.candidate_id),
        )
        for candidate in grid.aggregates
    )
    champion = ranked[0]
    return StatisticalChampionSelection(
        champion_candidate_id=champion.candidate_id,
        champion_state_count=champion.state_count,
        ranked_candidate_ids=tuple(candidate.candidate_id for candidate in ranked),
        evidence=evidence,
    )
