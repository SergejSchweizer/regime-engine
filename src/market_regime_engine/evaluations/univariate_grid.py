"""Reusable one-feature Xetra v3 candidate-grid evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.selection import (
    StatisticalChampionSelection,
    select_statistical_champion,
)
from market_regime_engine.evaluation.walk_forward import AdapterFactory, WalkForwardEvaluation
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan
from market_regime_engine.evaluations.clocks import EvaluationClock
from market_regime_engine.evaluations.contracts import (
    CandidateSpec,
    EvaluationId,
    EvaluationLineage,
    FeatureSpec,
    candidate_specs,
)
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.training.adapter_factory import adapter_factory
from market_regime_engine.training.candidate_grid import (
    CandidateGridEvaluation,
    aggregate_candidate,
)

_TIMESTAMP_COLUMN = "timestamp_m1"
CandidateRunner = Callable[
    [pd.DataFrame, WalkForwardPlan, ModelProfile, "_LineagedCandidate", AdapterFactory],
    WalkForwardEvaluation,
]


@dataclass(frozen=True, slots=True)
class _LineagedCandidate:
    spec: CandidateSpec
    source_build_id: str
    feature_selection_definition_hash: str
    feature_selection_execution_hash: str

    @property
    def candidate_id(self) -> str:
        return self.spec.candidate_id

    @property
    def model_family(self) -> str:
        return self.spec.model_family

    @property
    def state_count(self) -> int:
        return self.spec.state_count

    @property
    def mixture_count(self) -> int:
        return self.spec.mixture_count

    @property
    def feature_order(self) -> tuple[str, ...]:
        return self.spec.feature_order

    @property
    def feature_dimension(self) -> int:
        return self.spec.feature_dimension


@dataclass(frozen=True, slots=True)
class UnivariateFeatureGrid:
    feature_name: str
    feature_spec: FeatureSpec
    clock: EvaluationClock
    lineage: EvaluationLineage
    candidate_grid: CandidateGridEvaluation
    diagnostic_feature_model_winner: str | None
    no_winner_reason: str | None

    def __post_init__(self) -> None:
        if self.feature_name not in self.feature_spec.feature_order:
            raise ValueError("univariate feature must belong to its feature spec")
        if self.diagnostic_feature_model_winner is None == (self.no_winner_reason is None):
            raise ValueError("univariate grid must contain exactly one winner outcome")


def _clocked_source_rows(
    source_rows: pd.DataFrame, clock: EvaluationClock, feature_name: str
) -> pd.DataFrame:
    if _TIMESTAMP_COLUMN not in source_rows.columns or feature_name not in source_rows.columns:
        raise ValueError("univariate source rows must contain timestamp and selected feature")
    retained = {
        timestamp
        for fold in clock.fold_evidence
        for timestamp in (*fold.retained_train_timestamps, *fold.retained_test_timestamps)
    }
    result = source_rows.copy()
    result.loc[~result[_TIMESTAMP_COLUMN].isin(retained), feature_name] = None
    return result


def evaluate_univariate_feature_grid(
    source_rows: pd.DataFrame,
    *,
    plan: WalkForwardPlan,
    profile: ModelProfile,
    feature_spec: FeatureSpec,
    feature_name: str,
    clock: EvaluationClock,
    lineage: EvaluationLineage,
    runner: CandidateRunner,
) -> UnivariateFeatureGrid:
    """Evaluate the canonical 12 candidates for one feature on its shared evaluation clock."""

    if profile.profile_id != "xetra" or profile.profile_config_version != 3:
        raise ValueError("univariate grid requires the canonical Xetra v3 profile")
    if feature_spec.evaluation_id not in {
        EvaluationId.MEDOID_UNIVARIATE,
        EvaluationId.DELTA1_UNIVARIATE,
    }:
        raise ValueError("univariate grid rejects multivariate evaluation IDs")
    if feature_name not in feature_spec.feature_order:
        raise ValueError("feature_name must belong to the univariate feature spec")
    if (
        clock.evaluation_id is not feature_spec.evaluation_id
        or clock.feature_order != feature_spec.feature_order
    ):
        raise ValueError("univariate clock must match its evaluation feature spec")
    if lineage.evaluation_id is not feature_spec.evaluation_id:
        raise ValueError("univariate lineage must match its feature spec")
    if lineage.evaluation_plan_hash != plan.plan_hash or lineage.clock_hash != clock.clock_hash:
        raise ValueError("univariate lineage must match plan and clock")

    candidates = tuple(
        _LineagedCandidate(
            spec,
            lineage.source_build_id,
            lineage.feature_selection_definition_hash,
            lineage.feature_selection_execution_hash,
        )
        for spec in candidate_specs((feature_name,))
    )
    clocked_rows = _clocked_source_rows(source_rows, clock, feature_name)
    evaluations = tuple(
        runner(
            clocked_rows,
            plan,
            profile,
            candidate,
            cast(AdapterFactory, adapter_factory(profile, candidate)),
        )
        for candidate in candidates
    )
    grid = CandidateGridEvaluation(
        profile_id=profile.profile_id,
        profile_config_version=profile.profile_config_version,
        source_build_id=lineage.source_build_id,
        feature_order=(feature_name,),
        feature_selection_definition_hash=lineage.feature_selection_definition_hash,
        feature_selection_execution_hash=lineage.feature_selection_execution_hash,
        evaluation_plan_hash=plan.plan_hash,
        evaluations=evaluations,
        aggregates=tuple(aggregate_candidate(item) for item in evaluations),
    )
    winner: str | None
    reason: str | None
    try:
        selection: StatisticalChampionSelection = select_statistical_champion(grid)
    except ValueError as exc:
        winner, reason = None, str(exc)
    else:
        winner, reason = selection.champion_candidate_id, None
    return UnivariateFeatureGrid(feature_name, feature_spec, clock, lineage, grid, winner, reason)
