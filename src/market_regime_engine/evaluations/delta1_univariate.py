"""Canonical delta1-univariate Xetra v3 evaluation orchestration."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import cast

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan
from market_regime_engine.evaluations.agreement import (
    UnivariateAgreement,
    compare_univariate_to_multivariate,
)
from market_regime_engine.evaluations.clocks import EvaluationClock
from market_regime_engine.evaluations.contracts import (
    DELTA1_FEATURES,
    EvaluationId,
    EvaluationLineage,
    FeatureSpec,
)
from market_regime_engine.evaluations.medoid_multivariate import MedoidMultivariateEvaluation
from market_regime_engine.evaluations.medoid_univariate import _select_champion, _winner_evaluation
from market_regime_engine.evaluations.scheduling import randomized_order
from market_regime_engine.evaluations.univariate_grid import (
    CandidateRunner,
    UnivariateFeatureGrid,
    evaluate_univariate_feature_grid,
    run_univariate_candidate,
)
from market_regime_engine.profiles.config import ModelProfile


@dataclass(frozen=True, slots=True)
class Delta1UnivariateEvaluation:
    clock: EvaluationClock
    lineage: EvaluationLineage
    feature_grids: tuple[UnivariateFeatureGrid, ...]
    agreements: tuple[UnivariateAgreement, ...]
    eligible_feature_names: tuple[str, ...]
    champion_tie_feature_names: tuple[str, ...]
    delta1_univariate_evaluation_champion: str | None
    no_champion_reason: str | None

    def __post_init__(self) -> None:
        if tuple(grid.feature_name for grid in self.feature_grids) != DELTA1_FEATURES:
            raise ValueError("delta1-univariate grids must preserve canonical delta order")
        if tuple(agreement.feature_name for agreement in self.agreements) != tuple(
            grid.feature_name
            for grid in self.feature_grids
            if grid.diagnostic_feature_model_winner is not None
        ):
            raise ValueError("delta1-univariate agreements must cover each feature winner")
        if (self.delta1_univariate_evaluation_champion is None) == (
            self.no_champion_reason is None
        ):
            raise ValueError("delta1-univariate result must contain exactly one champion outcome")


def evaluate_delta1_univariate(
    source_rows: pd.DataFrame,
    *,
    plan: WalkForwardPlan,
    profile: ModelProfile,
    clock: EvaluationClock,
    lineage: EvaluationLineage,
    multivariate: MedoidMultivariateEvaluation,
    runner: CandidateRunner | None = None,
) -> Delta1UnivariateEvaluation:
    """Evaluate every canonical delta feature independently and rank by agreement only."""

    feature_spec = FeatureSpec(EvaluationId.DELTA1_UNIVARIATE, DELTA1_FEATURES)
    if profile.profile_id != "xetra" or profile.profile_config_version != 3:
        raise ValueError("delta1-univariate evaluation requires the canonical Xetra v3 profile")
    if (
        clock.evaluation_id is not EvaluationId.DELTA1_UNIVARIATE
        or clock.feature_order != DELTA1_FEATURES
    ):
        raise ValueError("delta1-univariate clock must match the canonical delta features")
    if (
        lineage.evaluation_id is not EvaluationId.DELTA1_UNIVARIATE
        or lineage.evaluation_plan_hash != plan.plan_hash
        or lineage.clock_hash != clock.clock_hash
    ):
        raise ValueError("delta1-univariate lineage must match its plan and clock")
    winner_id = multivariate.medoid_multivariate_statistical_champion
    if winner_id is None:
        raise ValueError("delta1-univariate evaluation requires a multivariate champion")
    multivariate_winner = next(
        (
            item
            for item in multivariate.candidate_grid.evaluations
            if item.candidate_id == winner_id
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

    workers = min(len(DELTA1_FEATURES), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        grids = tuple(
            executor.map(
                evaluate_feature,
                randomized_order(DELTA1_FEATURES, scope=EvaluationId.DELTA1_UNIVARIATE.value),
            )
        )
    by_feature_name = {grid.feature_name: grid for grid in grids}
    grids = tuple(by_feature_name[feature_name] for feature_name in DELTA1_FEATURES)
    agreements = tuple(
        compare_univariate_to_multivariate(
            grid.feature_name, _winner_evaluation(grid), multivariate_winner
        )
        for grid in grids
        if grid.diagnostic_feature_model_winner is not None
    )
    eligible, ties, champion, reason = _select_champion(agreements)
    return Delta1UnivariateEvaluation(
        clock, lineage, grids, agreements, eligible, ties, champion, reason
    )
