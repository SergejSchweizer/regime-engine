"""Canonical medoid-univariate Xetra v3 evaluation orchestration."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import cast

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.walk_forward import (
    WalkForwardEvaluation,
)
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan
from market_regime_engine.evaluations.agreement import (
    UnivariateAgreement,
    compare_univariate_to_multivariate,
)
from market_regime_engine.evaluations.clocks import EvaluationClock
from market_regime_engine.evaluations.contracts import EvaluationId, EvaluationLineage, FeatureSpec
from market_regime_engine.evaluations.medoid_multivariate import MedoidMultivariateEvaluation
from market_regime_engine.evaluations.univariate_grid import (
    CandidateRunner,
    UnivariateFeatureGrid,
    evaluate_univariate_feature_grid,
    run_univariate_candidate,
)
from market_regime_engine.profiles.config import ModelProfile

_NMI_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class MedoidUnivariateEvaluation:
    feature_spec: FeatureSpec
    clock: EvaluationClock
    lineage: EvaluationLineage
    feature_grids: tuple[UnivariateFeatureGrid, ...]
    agreements: tuple[UnivariateAgreement, ...]
    eligible_feature_names: tuple[str, ...]
    champion_tie_feature_names: tuple[str, ...]
    medoid_univariate_evaluation_champion: str | None
    no_champion_reason: str | None

    def __post_init__(self) -> None:
        if self.feature_spec.evaluation_id is not EvaluationId.MEDOID_UNIVARIATE:
            raise ValueError("medoid-univariate result requires its evaluation ID")
        if (
            tuple(grid.feature_name for grid in self.feature_grids)
            != self.feature_spec.feature_order
        ):
            raise ValueError("medoid-univariate grids must preserve canonical medoid order")
        if tuple(agreement.feature_name for agreement in self.agreements) != tuple(
            grid.feature_name
            for grid in self.feature_grids
            if grid.diagnostic_feature_model_winner is not None
        ):
            raise ValueError("medoid-univariate agreements must cover each feature winner")
        if (self.medoid_univariate_evaluation_champion is None) == (
            self.no_champion_reason is None
        ):
            raise ValueError("medoid-univariate result must contain exactly one champion outcome")


def _winner_evaluation(grid: UnivariateFeatureGrid) -> WalkForwardEvaluation:
    winner = grid.diagnostic_feature_model_winner
    if winner is None:
        raise ValueError("cannot resolve an evaluation for a missing feature winner")
    for evaluation in grid.candidate_grid.evaluations:
        if evaluation.candidate_id == winner:
            return evaluation
    raise ValueError("feature winner is not present in its candidate grid")


def _select_champion(
    agreements: tuple[UnivariateAgreement, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], str | None, str | None]:
    eligible = tuple(
        agreement
        for agreement in agreements
        if agreement.unavailable_reason is None and agreement.shared_fold_rate >= 0.80
    )
    if not eligible:
        return (), (), None, "no feature winner has eligible agreement evidence"
    nmi_values = tuple(
        agreement.dominant_state_nmi
        for agreement in eligible
        if agreement.dominant_state_nmi is not None
    )
    maximum_nmi = max(nmi_values)
    nmi_ties = tuple(
        agreement
        for agreement in eligible
        if agreement.dominant_state_nmi is not None
        and abs(agreement.dominant_state_nmi - maximum_nmi) <= _NMI_TOLERANCE
    )
    maximum_timestamps = max(agreement.shared_timestamp_count for agreement in nmi_ties)
    champion_ties = tuple(
        agreement.feature_name
        for agreement in nmi_ties
        if agreement.shared_timestamp_count == maximum_timestamps
    )
    return (
        tuple(agreement.feature_name for agreement in eligible),
        champion_ties,
        min(champion_ties),
        None,
    )


def evaluate_medoid_univariate(
    source_rows: pd.DataFrame,
    *,
    plan: WalkForwardPlan,
    profile: ModelProfile,
    feature_spec: FeatureSpec,
    clock: EvaluationClock,
    lineage: EvaluationLineage,
    multivariate: MedoidMultivariateEvaluation,
    runner: CandidateRunner | None = None,
) -> MedoidUnivariateEvaluation:
    """Evaluate every Stage-1 medoid independently and select by agreement evidence only."""

    if profile.profile_id != "xetra" or profile.profile_config_version != 3:
        raise ValueError("medoid-univariate evaluation requires the canonical Xetra v3 profile")
    if feature_spec.evaluation_id is not EvaluationId.MEDOID_UNIVARIATE:
        raise ValueError("medoid-univariate evaluation requires its feature spec")
    if (
        clock.evaluation_id is not EvaluationId.MEDOID_UNIVARIATE
        or clock.feature_order != feature_spec.feature_order
    ):
        raise ValueError("medoid-univariate clock must match its feature spec")
    if (
        lineage.evaluation_id is not EvaluationId.MEDOID_UNIVARIATE
        or lineage.evaluation_plan_hash != plan.plan_hash
        or lineage.clock_hash != clock.clock_hash
    ):
        raise ValueError("medoid-univariate lineage must match its plan and clock")
    if multivariate.medoid_multivariate_statistical_champion is None:
        raise ValueError("medoid-univariate evaluation requires a multivariate champion")
    multivariate_winner = next(
        (
            evaluation
            for evaluation in multivariate.candidate_grid.evaluations
            if evaluation.candidate_id == multivariate.medoid_multivariate_statistical_champion
        ),
        None,
    )
    if multivariate_winner is None:
        raise ValueError("multivariate champion is not present in its candidate grid")
    active_runner = cast(CandidateRunner, run_univariate_candidate) if runner is None else runner

    def evaluate_feature(feature_name: str) -> UnivariateFeatureGrid:
        return evaluate_univariate_feature_grid(
            source_rows,
            plan=plan,
            profile=profile,
            feature_spec=feature_spec,
            feature_name=feature_name,
            clock=clock,
            lineage=lineage,
            runner=active_runner,
        )

    workers = min(len(feature_spec.feature_order), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        grids = tuple(executor.map(evaluate_feature, feature_spec.feature_order))
    agreements = tuple(
        compare_univariate_to_multivariate(
            grid.feature_name, _winner_evaluation(grid), multivariate_winner
        )
        for grid in grids
        if grid.diagnostic_feature_model_winner is not None
    )
    eligible, ties, champion, reason = _select_champion(agreements)
    return MedoidUnivariateEvaluation(
        feature_spec, clock, lineage, grids, agreements, eligible, ties, champion, reason
    )
