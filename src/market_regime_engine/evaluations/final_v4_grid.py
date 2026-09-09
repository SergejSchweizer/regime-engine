"""Exact final Xetra v4 candidate grid on one frozen selected prefix."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.selection import (
    StatisticalChampionSelection,
    select_statistical_champion,
)
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan
from market_regime_engine.feature_discovery.contracts import (
    FINAL_CANDIDATE_IDS,
    MIN_PREFIX_LENGTH,
    content_hash,
)
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.profiles.resolution import (
    ResolvedCandidateProfile,
    ResolvedSelectedFeatureProfile,
)
from market_regime_engine.training.candidate_grid import (
    AdapterFactoryBuilder,
    CandidateGridEvaluation,
    CandidateRunner,
    _default_runner,
    evaluate_candidate_grid,
)

_TIMESTAMP_COLUMN = "timestamp_m1"
_GAUSSIAN_IDS = tuple(f"gaussian_hmm_k{state_count}_full" for state_count in (2, 3, 4, 5))
_GMM_IDS = tuple(f"gmm_hmm_k{state_count}_m2_full" for state_count in (2, 3, 4, 5))
_STUDENT_IDS = tuple(f"student_t_hmm_k{state_count}_full" for state_count in (2, 3, 4, 5))


@dataclass(frozen=True, slots=True)
class FinalV4GridEvaluation:
    """The exact 12-candidate evidence and one statistical selection outcome."""

    candidate_grid: CandidateGridEvaluation
    selection: StatisticalChampionSelection | None
    no_champion_reason: str | None

    def __post_init__(self) -> None:
        if self.candidate_grid.profile_config_version != 4:
            raise ValueError("final v4 grid result requires profile configuration version 4")
        if (
            tuple(item.candidate_id for item in self.candidate_grid.evaluations)
            != FINAL_CANDIDATE_IDS
        ):
            raise ValueError("final v4 grid must contain the exact ordered 12-candidate universe")
        if (
            tuple(item.candidate_id for item in self.candidate_grid.aggregates)
            != FINAL_CANDIDATE_IDS
        ):
            raise ValueError(
                "final v4 grid aggregates must contain the exact ordered 12 candidates"
            )
        if (self.selection is None) == (self.no_champion_reason is None):
            raise ValueError("final v4 grid requires exactly one champion or an explicit failure")
        if (
            self.selection is not None
            and self.selection.champion_candidate_id not in FINAL_CANDIDATE_IDS
        ):
            raise ValueError("final v4 champion is outside the exact candidate universe")

    @property
    def grid(self) -> CandidateGridEvaluation:
        """Short access alias for the persisted candidate-grid evidence."""
        return self.candidate_grid

    @property
    def champion(self) -> StatisticalChampionSelection | None:
        return self.selection


def _validate_inputs(
    source_rows: pd.DataFrame,
    feature_order: tuple[str, ...],
    original_feature_universe: tuple[str, ...],
    source_build_id: str,
    profile: ModelProfile,
    plan: WalkForwardPlan,
) -> None:
    if not isinstance(source_rows, pd.DataFrame):
        raise TypeError("final v4 grid requires a pandas DataFrame")
    if profile.profile_id != "xetra" or profile.profile_config_version != 4:
        raise ValueError("final v4 grid requires the canonical Xetra v4 profile")
    if len(feature_order) < MIN_PREFIX_LENGTH or len(set(feature_order)) != len(feature_order):
        raise ValueError("final v4 feature_order must contain at least two unique features")
    if not original_feature_universe or len(set(original_feature_universe)) != len(
        original_feature_universe
    ):
        raise ValueError("final v4 original feature universe must be non-empty and unique")
    if any(feature not in original_feature_universe for feature in feature_order):
        raise ValueError("final v4 feature_order must belong to the original feature universe")
    missing = tuple(
        feature for feature in (*(_TIMESTAMP_COLUMN,), *feature_order) if feature not in source_rows
    )
    if missing:
        raise ValueError(f"source rows are missing final v4 grid columns: {missing}")
    if not source_build_id or source_build_id.strip() != source_build_id:
        raise ValueError("final v4 source_build_id must be non-empty and trimmed")
    if not isinstance(plan, WalkForwardPlan) or not plan.folds:
        raise ValueError("final v4 grid requires a non-empty walk-forward plan")
    if tuple((item.state_count, item.mixture_count) for item in profile.gmm_hmms) != (
        (2, 2),
        (3, 2),
        (4, 2),
        (5, 2),
    ):
        raise ValueError("final v4 grid requires GMM-HMM K2-K5 with two mixtures")
    if profile.student_t_hmm is None or profile.student_t_hmm.candidate_states != (2, 3, 4, 5):
        raise ValueError("final v4 grid requires Student-t HMM K2-K5")


def _candidates(
    feature_order: tuple[str, ...],
    *,
    original_feature_universe: tuple[str, ...],
    source_build_id: str,
    feature_selection_definition_hash: str,
    feature_selection_execution_hash: str,
) -> tuple[ResolvedCandidateProfile, ...]:
    candidates: list[ResolvedCandidateProfile] = []
    for candidate_id in FINAL_CANDIDATE_IDS:
        if candidate_id in _GAUSSIAN_IDS:
            model_family = "gaussian_hmm"
            state_count = int(candidate_id.split("_k", 1)[1].split("_", 1)[0])
            mixture_count = 1
        elif candidate_id in _GMM_IDS:
            model_family = "gmm_hmm"
            state_count = int(candidate_id.split("_k", 1)[1].split("_", 1)[0])
            mixture_count = 2
        elif candidate_id in _STUDENT_IDS:
            model_family = "student_t_hmm"
            state_count = int(candidate_id.split("_k", 1)[1].split("_", 1)[0])
            mixture_count = 1
        else:
            raise ValueError(f"unsupported final v4 candidate ID: {candidate_id}")
        candidates.append(
            ResolvedCandidateProfile(
                candidate_id=candidate_id,
                state_count=state_count,
                covariance_type="full",
                feature_order=feature_order,
                feature_dimension=len(feature_order),
                source_build_id=source_build_id,
                feature_selection_definition_hash=feature_selection_definition_hash,
                feature_selection_execution_hash=feature_selection_execution_hash,
                original_feature_universe=original_feature_universe,
                preliminary_medoids=(),
                model_family=model_family,
                mixture_count=mixture_count,
                feature_contract_version=4,
            )
        )
    return tuple(candidates)


def evaluate_final_v4_grid(
    source_rows: pd.DataFrame,
    *,
    feature_order: tuple[str, ...],
    profile: ModelProfile,
    plan: WalkForwardPlan,
    source_build_id: str,
    original_feature_universe: tuple[str, ...] | None = None,
    feature_selection_definition_hash: str | None = None,
    feature_selection_execution_hash: str | None = None,
    adapter_factory_builder: AdapterFactoryBuilder | None = None,
    runner: CandidateRunner | None = None,
    max_workers: int | None = None,
) -> FinalV4GridEvaluation:
    """Run the exact final 12 candidates and apply statistical ranking once."""

    universe = (
        tuple(column for column in source_rows.columns if column != _TIMESTAMP_COLUMN)
        if original_feature_universe is None
        else original_feature_universe
    )
    _validate_inputs(source_rows, feature_order, universe, source_build_id, profile, plan)
    definition_hash = feature_selection_definition_hash or content_hash(
        ("xetra_global_regime_v4", "final_grid_definition")
    )
    execution_hash = feature_selection_execution_hash or content_hash(
        ("xetra_global_regime_v4", "final_grid_execution", feature_order, plan.plan_hash)
    )
    candidates = _candidates(
        feature_order,
        original_feature_universe=universe,
        source_build_id=source_build_id,
        feature_selection_definition_hash=definition_hash,
        feature_selection_execution_hash=execution_hash,
    )
    resolved = ResolvedSelectedFeatureProfile(
        profile_id="xetra",
        profile_config_version=4,
        registered_model=profile.registered_model,
        source_build_id=source_build_id,
        original_feature_universe=universe,
        preliminary_medoids=(),
        final_features=feature_order,
        feature_selection_definition_hash=definition_hash,
        feature_selection_execution_hash=execution_hash,
        candidates=candidates,
    )
    grid = evaluate_candidate_grid(
        source_rows,
        plan=plan,
        profile=profile,
        resolved_profile=resolved,
        adapter_factory_builder=adapter_factory_builder,
        runner=_default_runner if runner is None else runner,
        max_workers=max_workers,
    )
    reason: str | None
    try:
        selection = select_statistical_champion(grid)
    except ValueError as exc:
        selection, reason = None, str(exc)
    else:
        reason = None
    return FinalV4GridEvaluation(grid, selection, reason)


run_final_v4_grid = evaluate_final_v4_grid
evaluate_final_candidate_grid = evaluate_final_v4_grid


__all__ = [
    "FinalV4GridEvaluation",
    "evaluate_final_candidate_grid",
    "evaluate_final_v4_grid",
    "run_final_v4_grid",
]
