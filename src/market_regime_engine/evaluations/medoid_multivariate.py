"""Canonical production-eligible Xetra v3 medoid multivariate evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.selection import select_statistical_champion
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan
from market_regime_engine.evaluations.contracts import (
    EvaluationId,
    EvaluationLineage,
    FeatureSpec,
)
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.profiles.resolution import ResolvedSelectedFeatureProfile
from market_regime_engine.training.candidate_grid import (
    CandidateGridEvaluation,
    CandidateRunner,
    evaluate_candidate_grid,
)


@dataclass(frozen=True, slots=True)
class MedoidMultivariateEvaluation:
    feature_spec: FeatureSpec
    lineage: EvaluationLineage
    candidate_grid: CandidateGridEvaluation
    medoid_multivariate_statistical_champion: str | None
    no_champion_reason: str | None

    def __post_init__(self) -> None:
        if self.feature_spec.evaluation_id is not EvaluationId.MEDOID_MULTIVARIATE:
            raise ValueError("multivariate result requires the multivariate evaluation ID")
        if (self.medoid_multivariate_statistical_champion is None) == (
            self.no_champion_reason is None
        ):
            raise ValueError("multivariate result must contain exactly one champion outcome")


def evaluate_medoid_multivariate(
    source_rows: pd.DataFrame,
    *,
    plan: WalkForwardPlan,
    profile: ModelProfile,
    resolved_profile: ResolvedSelectedFeatureProfile,
    feature_spec: FeatureSpec,
    lineage: EvaluationLineage,
    runner: CandidateRunner | None = None,
) -> MedoidMultivariateEvaluation:
    """Evaluate the frozen Stage-2 tuple using the existing v3 grid and selection contracts."""

    if profile.profile_id != "xetra" or profile.profile_config_version != 3:
        raise ValueError("multivariate evaluation requires the canonical Xetra v3 profile")
    if not plan.folds or plan.evaluation_cutoff is None:
        raise ValueError("multivariate evaluation requires a complete walk-forward plan")
    if feature_spec.evaluation_id is not EvaluationId.MEDOID_MULTIVARIATE:
        raise ValueError("multivariate evaluation requires the multivariate feature spec")
    if feature_spec.feature_order != resolved_profile.final_features:
        raise ValueError("multivariate feature spec must equal frozen resolved Stage-2 features")
    if (
        lineage.evaluation_id is not EvaluationId.MEDOID_MULTIVARIATE
        or lineage.evaluation_plan_hash != plan.plan_hash
        or lineage.source_build_id != resolved_profile.source_build_id
        or lineage.feature_selection_definition_hash
        != resolved_profile.feature_selection_definition_hash
        or lineage.feature_selection_execution_hash
        != resolved_profile.feature_selection_execution_hash
    ):
        raise ValueError("multivariate lineage must match the frozen resolved profile and plan")
    if runner is None:
        grid = evaluate_candidate_grid(
            source_rows,
            plan=plan,
            profile=profile,
            resolved_profile=resolved_profile,
        )
    else:
        grid = evaluate_candidate_grid(
            source_rows,
            plan=plan,
            profile=profile,
            resolved_profile=resolved_profile,
            runner=runner,
        )
    champion: str | None
    reason: str | None
    try:
        champion = select_statistical_champion(grid).champion_candidate_id
    except ValueError as exc:
        champion, reason = None, str(exc)
    else:
        reason = None
    return MedoidMultivariateEvaluation(feature_spec, lineage, grid, champion, reason)
