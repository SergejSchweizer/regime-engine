"""Causal Gaussian teacher selection on the pinned inner Xetra v4 clock."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from itertools import pairwise
from typing import TYPE_CHECKING, cast

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.model_clock import (
    build_model_clock_preflight,
    require_model_clock_eligible,
)
from market_regime_engine.evaluation.selection import (
    StatisticalChampionSelection,
    rank_same_feature_candidates,
)
from market_regime_engine.evaluation.walk_forward import (
    AdapterFactory,
    WalkForwardEvaluation,
    run_walk_forward_candidate,
)
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardFold, WalkForwardPlan
from market_regime_engine.evaluations.scheduling import randomized_order
from market_regime_engine.feature_discovery.contracts import (
    INNER_ALLOW_PARTIAL_FINAL_TEST,
    INNER_STEP_SOURCE_OBSERVATIONS,
    INNER_TEST_SOURCE_OBSERVATIONS,
    INNER_TRAIN_SOURCE_OBSERVATIONS,
    MIN_MODEL_CLOCK_VALID_FOLD_RATE,
    MIN_MODEL_TEST_OBSERVATIONS,
    MIN_MODEL_TRAIN_OBSERVATIONS,
    V4_PROVISIONAL_STATE_COUNTS,
    DiscoveryStatus,
    ModelClockPreflight,
)
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile
from market_regime_engine.training.adapter_factory import adapter_factory
from market_regime_engine.training.candidate_grid import (
    CandidateAggregate,
    aggregate_candidate,
)

if TYPE_CHECKING:
    from market_regime_engine.evaluation_runs.hmm_units import HMMSeedCheckpoint

_TIMESTAMP_COLUMN = "timestamp_m1"
_GAUSSIAN_CANDIDATE_IDS = tuple(
    f"gaussian_hmm_k{state_count}_full" for state_count in V4_PROVISIONAL_STATE_COUNTS
)

ProvisionalCandidateRunner = Callable[
    [pd.DataFrame, WalkForwardPlan, ModelProfile, ResolvedCandidateProfile, AdapterFactory],
    WalkForwardEvaluation,
]


def _evaluate_candidates(
    source_rows: pd.DataFrame,
    plan: WalkForwardPlan,
    profile: ModelProfile,
    candidates: tuple[ResolvedCandidateProfile, ...],
    runner: ProvisionalCandidateRunner,
    max_workers: int | None,
    seed_checkpoint_factory: Callable[[str, str, int], HMMSeedCheckpoint] | None = None,
) -> dict[str, WalkForwardEvaluation]:
    """Evaluate teacher candidates concurrently while preserving canonical output order."""

    worker_limit = (os.cpu_count() or 1) if max_workers is None else max_workers
    if worker_limit < 1:
        raise ValueError("max_workers must be at least 1")
    worker_limit = min(worker_limit, len(candidates))

    def evaluate(candidate: ResolvedCandidateProfile) -> WalkForwardEvaluation:
        if seed_checkpoint_factory is not None and runner is run_provisional_gaussian_candidate:
            return run_provisional_gaussian_candidate(
                source_rows,
                plan,
                profile,
                candidate,
                cast(AdapterFactory, adapter_factory(profile, candidate)),
                seed_checkpoint_factory=lambda fold_id: seed_checkpoint_factory(
                    candidate.candidate_id,
                    fold_id,
                    candidate.state_count,
                ),
            )
        return runner(
            source_rows,
            plan,
            profile,
            candidate,
            cast(AdapterFactory, adapter_factory(profile, candidate)),
        )

    if worker_limit == 1:
        return {candidate.candidate_id: evaluate(candidate) for candidate in candidates}
    with ThreadPoolExecutor(max_workers=worker_limit) as executor:
        futures = {
            candidate.candidate_id: executor.submit(evaluate, candidate) for candidate in candidates
        }
        return {candidate_id: future.result() for candidate_id, future in futures.items()}


def _utc(value: object, field_name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != UTC.utcoffset(value)
    ):
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    return value


@dataclass(frozen=True, slots=True)
class _InnerFold:
    """Fold shape shared with the existing runner without changing the outer pin."""

    fold_index: int
    fold_id: str
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_source_observations: int
    test_source_observations: int

    def __post_init__(self) -> None:
        if self.fold_index < 1 or self.fold_id != f"fold_{self.fold_index:03d}":
            raise ValueError("inner fold identity must be deterministic and one-based")
        for field_name in ("train_start", "train_end", "test_start", "test_end"):
            _utc(getattr(self, field_name), field_name)
        if not self.train_start <= self.train_end < self.test_start <= self.test_end:
            raise ValueError("inner TRAIN and TEST bounds must be ordered and non-overlapping")
        if self.train_source_observations < INNER_TRAIN_SOURCE_OBSERVATIONS:
            raise ValueError("inner TRAIN source observations are below the pinned minimum")
        if self.test_source_observations != INNER_TEST_SOURCE_OBSERVATIONS:
            raise ValueError("inner TEST source observations must be exactly 63")


def build_inner_walk_forward_plan(timestamps: Sequence[datetime]) -> WalkForwardPlan:
    """Build complete expanding `756/63/63` folds with no partial final TEST."""

    ordered = tuple(_utc(value, "source timestamp") for value in timestamps)
    if any(current <= previous for previous, current in pairwise(ordered)):
        raise ValueError("source timestamps must be strictly increasing and unique")

    folds: list[_InnerFold] = []
    train_end_index = INNER_TRAIN_SOURCE_OBSERVATIONS - 1
    test_start_index = train_end_index + 1
    fold_index = 1
    while test_start_index + INNER_TEST_SOURCE_OBSERVATIONS <= len(ordered):
        test_end_index = test_start_index + INNER_TEST_SOURCE_OBSERVATIONS - 1
        folds.append(
            _InnerFold(
                fold_index=fold_index,
                fold_id=f"fold_{fold_index:03d}",
                train_start=ordered[0],
                train_end=ordered[train_end_index],
                test_start=ordered[test_start_index],
                test_end=ordered[test_end_index],
                train_source_observations=train_end_index + 1,
                test_source_observations=INNER_TEST_SOURCE_OBSERVATIONS,
            )
        )
        fold_index += 1
        train_end_index += INNER_STEP_SOURCE_OBSERVATIONS
        test_start_index += INNER_STEP_SOURCE_OBSERVATIONS

    if not folds:
        raise ValueError("inner plan requires at least one complete 756/63/63 fold")
    payload = {
        "allow_partial_final_test": INNER_ALLOW_PARTIAL_FINAL_TEST,
        "minimum_train_source_observations": INNER_TRAIN_SOURCE_OBSERVATIONS,
        "step_source_observations": INNER_STEP_SOURCE_OBSERVATIONS,
        "test_source_observations": INNER_TEST_SOURCE_OBSERVATIONS,
        "folds": [
            {
                "fold_id": fold.fold_id,
                "fold_index": fold.fold_index,
                "test_end": fold.test_end.isoformat(),
                "test_start": fold.test_start.isoformat(),
                "train_end": fold.train_end.isoformat(),
                "train_start": fold.train_start.isoformat(),
            }
            for fold in folds
        ],
    }
    plan_hash = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return WalkForwardPlan(
        # The shared outer fold class intentionally pins TRAIN >=1260.  The
        # runner and clock consume this structural shape, so the inner plan
        # uses its own validated 756-row fold while retaining the plan type.
        folds=cast(tuple[WalkForwardFold, ...], tuple(folds)),
        evaluation_cutoff=folds[-1].test_end,
        plan_hash=plan_hash,
    )


def _validate_profile(profile: ModelProfile) -> None:
    if profile.profile_id != "xetra" or profile.profile_config_version != 4:
        raise ValueError("provisional teacher requires the canonical Xetra v4 profile")
    discovery = profile.feature_discovery
    if discovery is None:
        raise ValueError("provisional teacher requires the v4 feature-discovery profile")
    exact_inner = (
        discovery.inner_train_source_observations,
        discovery.inner_test_source_observations,
        discovery.inner_step_source_observations,
        discovery.inner_partial_final_test,
    )
    expected_inner = (
        INNER_TRAIN_SOURCE_OBSERVATIONS,
        INNER_TEST_SOURCE_OBSERVATIONS,
        INNER_STEP_SOURCE_OBSERVATIONS,
        INNER_ALLOW_PARTIAL_FINAL_TEST,
    )
    if exact_inner != expected_inner:
        raise ValueError("v4 provisional teacher requires the pinned inner 756/63/63 plan")
    if profile.gaussian_hmm.candidate_states != V4_PROVISIONAL_STATE_COUNTS:
        raise ValueError("provisional teacher requires Gaussian K2/K3/K4/K5 candidates")


def _validate_prototype_features(prototype_features: tuple[str, ...]) -> None:
    if (
        not isinstance(prototype_features, tuple)
        or not prototype_features
        or len(set(prototype_features)) != len(prototype_features)
        or any(not isinstance(name, str) or not name.strip() for name in prototype_features)
    ):
        raise ValueError("prototype_features must be a non-empty duplicate-free tuple")


def _validate_prototypes(source_rows: pd.DataFrame, prototype_features: tuple[str, ...]) -> None:
    _validate_prototype_features(prototype_features)
    required = (_TIMESTAMP_COLUMN, *prototype_features)
    missing = tuple(name for name in required if name not in source_rows.columns)
    if missing:
        raise ValueError(f"source rows are missing provisional teacher columns: {missing}")


def _candidates(
    prototype_features: tuple[str, ...],
    *,
    source_build_id: str,
    feature_selection_definition_hash: str,
    feature_selection_execution_hash: str,
) -> tuple[ResolvedCandidateProfile, ...]:
    return tuple(
        ResolvedCandidateProfile(
            candidate_id=f"gaussian_hmm_k{state_count}_full",
            state_count=state_count,
            covariance_type="full",
            feature_order=prototype_features,
            feature_dimension=len(prototype_features),
            source_build_id=source_build_id,
            feature_selection_definition_hash=feature_selection_definition_hash,
            feature_selection_execution_hash=feature_selection_execution_hash,
            original_feature_universe=prototype_features,
            feature_contract_version=4,
        )
        for state_count in V4_PROVISIONAL_STATE_COUNTS
    )


def run_provisional_gaussian_candidate(
    source_rows: pd.DataFrame,
    plan: WalkForwardPlan,
    profile: ModelProfile,
    candidate: ResolvedCandidateProfile,
    candidate_adapter_factory: AdapterFactory,
    *,
    seed_checkpoint_factory: Callable[[str], HMMSeedCheckpoint] | None = None,
) -> WalkForwardEvaluation:
    """Run one teacher candidate through the existing walk-forward runner."""

    return run_walk_forward_candidate(
        source_rows,
        plan=plan,
        profile=profile,
        candidate=candidate,
        adapter_factory=candidate_adapter_factory,
        seed_checkpoint_factory=seed_checkpoint_factory,
    )


@dataclass(frozen=True, slots=True)
class ProvisionalTeacherEvaluation:
    """Complete inner evidence and selection chain for the temporary teacher."""

    source_build_id: str
    prototype_features: tuple[str, ...]
    inner_plan: WalkForwardPlan
    model_clock: ModelClockPreflight
    candidate_evaluations: tuple[WalkForwardEvaluation, ...]
    candidate_aggregates: tuple[CandidateAggregate, ...]
    selection: StatisticalChampionSelection | None
    no_selection_reason: str | None

    def __post_init__(self) -> None:
        if not self.source_build_id or self.source_build_id.strip() != self.source_build_id:
            raise ValueError("source_build_id must be a non-empty trimmed string")
        _validate_prototype_features(self.prototype_features)
        expected_fold_ids = tuple(fold.fold_id for fold in self.inner_plan.folds)
        if not expected_fold_ids:
            raise ValueError("provisional teacher requires complete inner folds")
        if self.model_clock.plan_hash != self.inner_plan.plan_hash:
            raise ValueError("teacher model clock must match inner plan")
        if self.model_clock.feature_order != self.prototype_features:
            raise ValueError("teacher model clock must match prototype features")
        if self.model_clock.status is not DiscoveryStatus.VALID:
            raise ValueError("provisional teacher cannot persist an ineligible model clock")
        if (
            tuple(item.candidate_id for item in self.candidate_evaluations)
            != _GAUSSIAN_CANDIDATE_IDS
        ):
            raise ValueError("teacher evaluations must contain Gaussian K2/K3/K4/K5 in order")
        if (
            tuple(item.candidate_id for item in self.candidate_aggregates)
            != _GAUSSIAN_CANDIDATE_IDS
        ):
            raise ValueError("teacher aggregates must contain Gaussian K2/K3/K4/K5 in order")
        first_evaluation = self.candidate_evaluations[0]
        shared_contract = (
            first_evaluation.source_build_id,
            first_evaluation.feature_order,
            first_evaluation.feature_selection_definition_hash,
            first_evaluation.feature_selection_execution_hash,
        )
        if shared_contract[0] != self.source_build_id:
            raise ValueError("teacher source lineage differs from result identity")
        for evaluation, aggregate in zip(
            self.candidate_evaluations, self.candidate_aggregates, strict=True
        ):
            if evaluation.state_count != aggregate.state_count:
                raise ValueError("teacher evaluation and aggregate state counts differ")
            if evaluation.source_build_id != self.source_build_id:
                raise ValueError("teacher evaluation source lineage differs")
            if evaluation.feature_order != self.prototype_features:
                raise ValueError("teacher evaluation feature order differs")
            if (
                evaluation.source_build_id,
                evaluation.feature_order,
                evaluation.feature_selection_definition_hash,
                evaluation.feature_selection_execution_hash,
            ) != shared_contract:
                raise ValueError("teacher candidates must preserve shared selection lineage")
            if tuple(fold.fold_id for fold in evaluation.folds) != expected_fold_ids:
                raise ValueError("teacher candidates must preserve identical inner folds")
            if evaluation.evaluation_plan_hash != self.inner_plan.plan_hash:
                raise ValueError("teacher evaluation plan hash differs")
        if (self.selection is None) == (self.no_selection_reason is None):
            raise ValueError("teacher result must contain exactly one selection outcome")
        if (
            self.selection is not None
            and self.selection.champion_candidate_id not in _GAUSSIAN_CANDIDATE_IDS
        ):
            raise ValueError("teacher champion must be Gaussian K2/K3/K4/K5")
        if self.selection is not None and (
            not self.selection.ranked_candidate_ids
            or len(set(self.selection.ranked_candidate_ids))
            != len(self.selection.ranked_candidate_ids)
            or any(
                candidate_id not in _GAUSSIAN_CANDIDATE_IDS
                for candidate_id in self.selection.ranked_candidate_ids
            )
        ):
            raise ValueError("teacher selection must rank only Gaussian K2/K3/K4/K5")

    @property
    def provisional_candidate_id(self) -> str | None:
        return None if self.selection is None else self.selection.champion_candidate_id

    @property
    def provisional_state_count(self) -> int | None:
        return None if self.selection is None else self.selection.champion_state_count


def select_provisional_teacher(
    source_rows: pd.DataFrame,
    *,
    profile: ModelProfile,
    prototype_features: tuple[str, ...],
    source_build_id: str,
    feature_selection_definition_hash: str,
    feature_selection_execution_hash: str,
    runner: ProvisionalCandidateRunner = run_provisional_gaussian_candidate,
    max_workers: int | None = None,
    seed_checkpoint_factory: Callable[[str, str, int], HMMSeedCheckpoint] | None = None,
) -> ProvisionalTeacherEvaluation:
    """Select provisional Gaussian K causally using only the supplied TRAIN rows."""

    if not isinstance(source_rows, pd.DataFrame):
        raise TypeError("source_rows must be a pandas DataFrame")
    _validate_profile(profile)
    _validate_prototypes(source_rows, prototype_features)
    plan = build_inner_walk_forward_plan(tuple(source_rows[_TIMESTAMP_COLUMN]))

    # This is deliberately before candidate construction, adapters, and runner calls.
    preflight = build_model_clock_preflight(
        source_rows,
        prototype_features,
        plan,
        minimum_model_train_observations=MIN_MODEL_TRAIN_OBSERVATIONS,
        minimum_model_test_observations=MIN_MODEL_TEST_OBSERVATIONS,
        minimum_valid_fold_rate=MIN_MODEL_CLOCK_VALID_FOLD_RATE,
    )
    require_model_clock_eligible(preflight, "prototype")

    candidates = _candidates(
        prototype_features,
        source_build_id=source_build_id,
        feature_selection_definition_hash=feature_selection_definition_hash,
        feature_selection_execution_hash=feature_selection_execution_hash,
    )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    scheduled = tuple(
        by_id[candidate_id]
        for candidate_id in randomized_order(_GAUSSIAN_CANDIDATE_IDS, scope="provisional-teacher")
    )
    if seed_checkpoint_factory is None:
        evaluations_by_id = _evaluate_candidates(
            source_rows,
            plan,
            profile,
            scheduled,
            runner,
            max_workers,
        )
    else:
        evaluations_by_id = _evaluate_candidates(
            source_rows,
            plan,
            profile,
            scheduled,
            runner,
            max_workers,
            seed_checkpoint_factory,
        )
    evaluations = tuple(evaluations_by_id[candidate_id] for candidate_id in _GAUSSIAN_CANDIDATE_IDS)
    aggregates = tuple(aggregate_candidate(evaluation) for evaluation in evaluations)
    selection: StatisticalChampionSelection | None
    reason: str | None
    try:
        selection = rank_same_feature_candidates(evaluations, aggregates)
    except ValueError as exc:
        selection, reason = None, str(exc)
    else:
        reason = None
    return ProvisionalTeacherEvaluation(
        source_build_id=source_build_id,
        prototype_features=prototype_features,
        inner_plan=plan,
        model_clock=preflight,
        candidate_evaluations=evaluations,
        candidate_aggregates=aggregates,
        selection=selection,
        no_selection_reason=reason,
    )


evaluate_provisional_teacher = select_provisional_teacher


__all__ = [
    "ProvisionalCandidateRunner",
    "ProvisionalTeacherEvaluation",
    "build_inner_walk_forward_plan",
    "evaluate_provisional_teacher",
    "run_provisional_gaussian_candidate",
    "select_provisional_teacher",
]
