"""Deterministic statistical champion selection for the exact Xetra HMM grid."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from statistics import fmean, pstdev

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
            raise ValueError(
                "accepted candidates have no rejections; rejected candidates require them"
            )
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
    common_valid_fold_ids: tuple[str, ...] = ()
    common_valid_fold_count: int = 0
    common_valid_fold_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.ranking_abs_tolerance != 1e-12:
            raise ValueError("statistical ranking tolerance must be exactly 1e-12")
        if self.common_valid_fold_count != len(self.common_valid_fold_ids):
            raise ValueError("common valid-fold count must match its IDs")
        if not 0.0 <= self.common_valid_fold_rate <= 1.0:
            raise ValueError("common valid-fold rate must be in [0, 1]")
        invalid_champion_order = (
            not self.ranked_candidate_ids
            or self.ranked_candidate_ids[0] != self.champion_candidate_id
        )
        if invalid_champion_order:
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


def _required(value: float | None, name: str) -> float:
    if value is None or not isfinite(value):
        raise ValueError(f"accepted candidate has invalid {name}")
    return value


def _anchored_partition(
    candidates: tuple[CandidateAggregate, ...], field_name: str, higher_is_better: bool
) -> tuple[tuple[CandidateAggregate, ...], ...]:
    """Partition a group into globally anchored numeric-tolerance tiers."""

    remaining = list(candidates)
    partitions: list[tuple[CandidateAggregate, ...]] = []
    while remaining:
        values = [_required(getattr(item, field_name), field_name) for item in remaining]
        anchor = max(values) if higher_is_better else min(values)
        tied = tuple(
            item
            for item in remaining
            if (
                _required(getattr(item, field_name), field_name) >= anchor - RANKING_ABS_TOLERANCE
                if higher_is_better
                else _required(getattr(item, field_name), field_name)
                <= anchor + RANKING_ABS_TOLERANCE
            )
        )
        partitions.append(tied)
        tied_ids = {item.candidate_id for item in tied}
        remaining = [item for item in remaining if item.candidate_id not in tied_ids]
    return tuple(partitions)


def _rank(accepted: tuple[CandidateAggregate, ...]) -> tuple[CandidateAggregate, ...]:
    stages = (
        ("oos_predictive_loglik_mean", True),
        ("oos_predictive_loglik_std", False),
        ("oos_predictive_loglik_worst_fold", True),
        ("bic_mean", False),
        ("aic_mean", False),
    )
    groups = (accepted,)
    for field_name, higher_is_better in stages:
        groups = tuple(
            subgroup
            for group in groups
            for subgroup in _anchored_partition(group, field_name, higher_is_better)
        )
    return tuple(
        item
        for group in groups
        for item in sorted(group, key=lambda item: (item.state_count, item.candidate_id))
    )


def _common_support_ranked_aggregates(
    grid: CandidateGridEvaluation,
    accepted: tuple[CandidateAggregate, ...],
) -> tuple[tuple[CandidateAggregate, ...], tuple[str, ...], float]:
    evaluations = {
        evaluation.candidate_id: evaluation
        for evaluation in grid.evaluations
        if evaluation.candidate_id in {candidate.candidate_id for candidate in accepted}
    }
    if any(
        len(evaluations[candidate.candidate_id].folds) != candidate.planned_fold_count
        for candidate in accepted
    ):
        return accepted, (), 1.0
    common_ids = set(fold.fold_id for fold in evaluations[accepted[0].candidate_id].valid_folds)
    for candidate in accepted[1:]:
        common_ids.intersection_update(
            fold.fold_id for fold in evaluations[candidate.candidate_id].valid_folds
        )
    ordered_ids = tuple(
        fold.fold_id
        for fold in evaluations[accepted[0].candidate_id].folds
        if fold.fold_id in common_ids
    )
    rate = len(ordered_ids) / accepted[0].planned_fold_count
    if rate < CANDIDATE_VALID_FOLD_RATE_GATE:
        raise ValueError("common valid-fold rate is below 0.80")
    ranked_inputs: list[CandidateAggregate] = []
    for candidate in accepted:
        folds = {
            fold.fold_id: fold
            for fold in evaluations[candidate.candidate_id].valid_folds
            if fold.fold_id in common_ids
        }
        oos = tuple(
            _required(folds[fold_id].oos_predictive_log_likelihood_per_observation, "OOS metric")
            for fold_id in ordered_ids
        )
        bics = tuple(_required(folds[fold_id].bic, "BIC") for fold_id in ordered_ids)
        aics = tuple(_required(folds[fold_id].aic, "AIC") for fold_id in ordered_ids)
        ranked_inputs.append(
            replace(
                candidate,
                oos_predictive_loglik_mean=fmean(oos),
                oos_predictive_loglik_std=pstdev(oos),
                oos_predictive_loglik_worst_fold=min(oos),
                oos_predictive_loglik_best_fold=max(oos),
                bic_mean=fmean(bics),
                aic_mean=fmean(aics),
            )
        )
    return tuple(ranked_inputs), ordered_ids, rate


def select_statistical_champion(grid: CandidateGridEvaluation) -> StatisticalChampionSelection:
    """Apply hard gates then the exact seven-stage EVALUATION ranking."""

    rejection_map = {
        candidate.candidate_id: _rejections(candidate) for candidate in grid.aggregates
    }
    accepted = tuple(
        candidate for candidate in grid.aggregates if not rejection_map[candidate.candidate_id]
    )
    if not accepted:
        raise ValueError("no candidate passes statistical hard gates")
    ranking_inputs, common_valid_fold_ids, common_valid_fold_rate = (
        _common_support_ranked_aggregates(grid, accepted)
    )
    ranked = _rank(ranking_inputs)
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
        common_valid_fold_ids=common_valid_fold_ids,
        common_valid_fold_count=len(common_valid_fold_ids),
        common_valid_fold_rate=common_valid_fold_rate,
    )
