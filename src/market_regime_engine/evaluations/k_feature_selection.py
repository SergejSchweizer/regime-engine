"""Process-parallel orchestration boundary for K-specific TRAIN-only selection.

The actual discovery stages are supplied as one process-safe policy callback.
The boundary owns the important invariants: exactly one K per task, immutable
source identity, no TEST frame in the callback signature, and canonical K
ordering.  This keeps the four policies independent and prevents the legacy
single-v4 selector from silently choosing another teacher K.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import cast

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.calendar_clock import (
    MIN_CALENDAR_MODEL_TEST_OBSERVATIONS,
    CalendarMonthFold,
    plan_calendar_month,
)
from market_regime_engine.evaluation.errors import RecoverableEvaluationInvalidity
from market_regime_engine.evaluation.model_clock import (
    build_calendar_model_clock_preflight,
    build_model_clock_preflight,
    require_model_clock_eligible,
)
from market_regime_engine.evaluation.walk_forward import AdapterFactory, WalkForwardEvaluation
from market_regime_engine.evaluations.global_regime_v4 import _as_feature_snapshot
from market_regime_engine.evaluations.k_champion_contract import (
    KChampionSelection,
    feature_order_hash,
)
from market_regime_engine.evaluations.process_parallel import cpu_process_pool, is_pickleable
from market_regime_engine.evaluations.provisional_teacher import (
    build_inner_calendar_month_plan,
    run_provisional_gaussian_candidate,
)
from market_regime_engine.feature_discovery.clustering import select_global_clusters
from market_regime_engine.feature_discovery.contracts import (
    INNER_TRAIN_SOURCE_OBSERVATIONS,
    MIN_MODEL_CLOCK_VALID_FOLD_RATE,
    MIN_MODEL_TEST_OBSERVATIONS,
    MIN_MODEL_TRAIN_OBSERVATIONS,
    ProvisionalTeacherReference,
    content_hash,
)
from market_regime_engine.feature_discovery.distance import global_absolute_spearman_distance
from market_regime_engine.feature_discovery.prefix_search import search_ranked_prefixes
from market_regime_engine.feature_discovery.prototypes import select_temporary_prototypes
from market_regime_engine.feature_discovery.quality import filter_outer_train_quality
from market_regime_engine.feature_discovery.scoring import score_all_raw_features
from market_regime_engine.feature_discovery.winners import select_cluster_winners
from market_regime_engine.features.ports import FeatureCatalogSnapshot
from market_regime_engine.preprocessing.pca_features import validate_pca_feature_universe
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile
from market_regime_engine.runtime.cpu import cpu_worker_count, nested_worker_limits
from market_regime_engine.training.adapter_factory import adapter_factory

# Retain the established module seam for test doubles while routing the
# canonical selector through the calendar-month implementation.
build_inner_walk_forward_plan = build_inner_calendar_month_plan

LEGAL_K = (2, 3, 4, 5)
K_FEATURE_SELECTION_VERSION = "k_specific_feature_selection.v1"


@dataclass(frozen=True, slots=True)
class KFeatureSelectionPayload:
    """Immutable TRAIN-only selection evidence returned by one K policy."""

    selection: KChampionSelection
    discovery_hash: str
    teacher_identity: str
    prefix_evidence_hash: str

    def __post_init__(self) -> None:
        for value, field in (
            (self.discovery_hash, "discovery_hash"),
            (self.prefix_evidence_hash, "prefix_evidence_hash"),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{field} must be a lowercase SHA-256")
        if not isinstance(self.teacher_identity, str) or not self.teacher_identity.strip():
            raise ValueError("teacher_identity must be non-empty")
        if self.teacher_identity != self.selection.reference_teacher_id:
            raise ValueError("teacher_identity must match the immutable selection teacher")


KTrainSelector = Callable[..., KFeatureSelectionPayload | None]


def _utc(value: object, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != UTC.utcoffset(value)
    ):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return value


def _fixed_k_teacher_reference(
    evaluation: WalkForwardEvaluation,
    *,
    source_build_id: str,
    prototype_features: tuple[str, ...],
) -> ProvisionalTeacherReference:
    """Build the frozen causal reference from one requested Gaussian K."""

    timestamps: list[datetime] = []
    probabilities: list[tuple[float, ...]] = []
    for fold in evaluation.valid_folds:
        if len(fold.oos_timestamps) != len(fold.oos_filtered_probabilities):
            raise ValueError("fixed-K teacher OOS timestamps/probabilities do not align")
        for timestamp, row in zip(
            fold.oos_timestamps, fold.oos_filtered_probabilities, strict=True
        ):
            timestamp = _utc(timestamp, "teacher OOS timestamp")
            if timestamps and timestamp <= timestamps[-1]:
                raise ValueError("fixed-K teacher OOS timestamps are not strictly increasing")
            values = tuple(float(value) for value in row)
            if len(values) != evaluation.state_count or any(value < 0.0 for value in values):
                raise ValueError("fixed-K teacher probabilities have the wrong dimension")
            if abs(sum(values) - 1.0) > 1.0e-10:
                raise ValueError("fixed-K teacher probabilities are not normalized")
            timestamps.append(timestamp)
            probabilities.append(values)
    if not timestamps:
        reasons = tuple(
            fold.failure_reason
            for fold in evaluation.folds
            if not fold.valid and fold.failure_reason
        )
        detail = "; ".join(reasons[:3]) if reasons else "no valid fold diagnostics"
        raise RecoverableEvaluationInvalidity(
            f"fixed-K teacher has no valid inner-fold support: {detail}"
        )
    return ProvisionalTeacherReference(
        candidate_id=evaluation.candidate_id,
        state_count=evaluation.state_count,
        timestamps=tuple(timestamps),
        filtered_probabilities=tuple(probabilities),
        dominant_states=tuple(
            max(range(evaluation.state_count), key=lambda index: (row[index], -index))
            for row in probabilities
        ),
        valid_inner_fold_ids=tuple(fold.fold_id for fold in evaluation.valid_folds),
        source_build_id=source_build_id,
        inner_plan_hash=evaluation.evaluation_plan_hash,
        prototype_features=prototype_features,
    )


def select_k_specific_feature_configuration(
    train_rows: pd.DataFrame,
    *,
    catalog: FeatureCatalogSnapshot,
    profile: ModelProfile,
    state_count: int,
    source_snapshot_id: str,
    source_build_id: str,
    validation_cutoff: datetime,
    deployment_cutoff: datetime,
    feature_selection_definition_hash: str | None = None,
    feature_selection_execution_hash: str | None = None,
    max_workers: int | None = None,
) -> KFeatureSelectionPayload:
    """Run the complete real TRAIN-only discovery chain for one fixed K.

    Every K owns its quality, distance, clustering, teacher, feature scoring
    and prefix evidence.  The fixed Gaussian teacher and prefix candidate
    cannot silently select another state count; all CPU-heavy stages retain
    their existing process-backed worker budget.
    """

    if state_count not in LEGAL_K:
        raise ValueError("K-specific selection accepts only K=2,3,4,5")
    source_snapshot_id = source_snapshot_id.strip()
    source_build_id = source_build_id.strip()
    if not source_snapshot_id or not source_build_id:
        raise ValueError("source and snapshot identities must be non-empty")
    validation_cutoff = _utc(validation_cutoff, "validation_cutoff")
    deployment_cutoff = _utc(deployment_cutoff, "deployment_cutoff")
    if validation_cutoff >= deployment_cutoff:
        raise ValueError("deployment_cutoff must be after validation_cutoff")
    mandatory_raw_order = validate_pca_feature_universe(
        catalog,
        component_count=profile.pca.component_count,
    )
    snapshot = _as_feature_snapshot(train_rows, catalog)
    if snapshot.lineage.source_build_id != source_build_id:
        raise ValueError("source_build_id differs from catalog lineage")
    timestamps = tuple(row.timestamp for row in snapshot.rows)
    if not timestamps or timestamps[-1] != validation_cutoff:
        raise RecoverableEvaluationInvalidity("TRAIN rows must end exactly at validation_cutoff")
    definition_hash = feature_selection_definition_hash or content_hash(
        (K_FEATURE_SELECTION_VERSION, profile.profile_hash, state_count, "definition")
    )
    execution_hash = feature_selection_execution_hash or content_hash(
        (K_FEATURE_SELECTION_VERSION, catalog.catalog_hash, state_count, validation_cutoff)
    )
    quality = filter_outer_train_quality(
        catalog,
        snapshot,
        timestamps[0],
        validation_cutoff,
        max_workers=max_workers,
    )
    distance = global_absolute_spearman_distance(snapshot, quality, max_workers=max_workers)
    clusters = select_global_clusters(distance, max_workers=max_workers)
    prototypes = select_temporary_prototypes(clusters, distance)
    inner_plan = build_inner_walk_forward_plan(timestamps)
    inner_folds = getattr(inner_plan, "folds", ())
    if inner_folds and isinstance(inner_folds[0], CalendarMonthFold):
        calendar_plan = plan_calendar_month(
            timestamps,
            minimum_train_source_observations=INNER_TRAIN_SOURCE_OBSERVATIONS,
        )
        preflight = build_calendar_model_clock_preflight(
            train_rows,
            prototypes.prototypes,
            calendar_plan,
            minimum_model_train_observations=MIN_MODEL_TRAIN_OBSERVATIONS,
            minimum_model_test_observations=MIN_CALENDAR_MODEL_TEST_OBSERVATIONS,
            minimum_valid_fold_rate=MIN_MODEL_CLOCK_VALID_FOLD_RATE,
        )
    else:
        preflight = build_model_clock_preflight(
            train_rows,
            prototypes.prototypes,
            inner_plan,
            minimum_model_train_observations=MIN_MODEL_TRAIN_OBSERVATIONS,
            minimum_model_test_observations=MIN_MODEL_TEST_OBSERVATIONS,
            minimum_valid_fold_rate=MIN_MODEL_CLOCK_VALID_FOLD_RATE,
        )
    require_model_clock_eligible(preflight, "prototype")
    teacher_candidate = ResolvedCandidateProfile(
        candidate_id=f"gaussian_hmm_k{state_count}_full",
        state_count=state_count,
        covariance_type="full",
        feature_order=prototypes.prototypes,
        feature_dimension=len(prototypes.prototypes),
        source_build_id=source_build_id,
        feature_selection_definition_hash=definition_hash,
        feature_selection_execution_hash=execution_hash,
        original_feature_universe=catalog.feature_names,
        feature_contract_version=4,
    )
    teacher_evaluation = run_provisional_gaussian_candidate(
        train_rows,
        inner_plan,
        profile,
        teacher_candidate,
        cast(AdapterFactory, adapter_factory(profile, teacher_candidate)),
        max_workers=max_workers,
        pca_raw_feature_order=mandatory_raw_order,
        pca_variance_threshold=profile.pca.variance_threshold,
    )
    teacher = _fixed_k_teacher_reference(
        teacher_evaluation,
        source_build_id=source_build_id,
        prototype_features=prototypes.prototypes,
    )
    scores = score_all_raw_features(snapshot, quality, teacher, max_workers=max_workers)
    winners = select_cluster_winners(clusters, scores)
    try:
        prefixes = search_ranked_prefixes(
            train_rows,
            ranked_features=winners.ranked_features,
            teacher=teacher,
            profile=profile,
            inner_plan=inner_plan,
            source_build_id=source_build_id,
            original_feature_universe=catalog.feature_names,
            feature_selection_definition_hash=definition_hash,
            feature_selection_execution_hash=execution_hash,
            max_workers=max_workers,
            pca_raw_feature_order=mandatory_raw_order,
            pca_variance_threshold=profile.pca.variance_threshold,
            state_counts=(state_count,),
        )
    except ValueError as error:
        raise RecoverableEvaluationInvalidity(str(error)) from error
    selected_prefix = prefixes.evaluations[prefixes.selected_prefix_length - 2]
    if not selected_prefix.valid:
        raise RecoverableEvaluationInvalidity("fixed-K prefix selection returned an invalid prefix")
    discovery_hash = content_hash(
        (
            catalog.catalog_hash,
            quality.result_hash,
            distance.matrix_hash,
            clusters.solution_hash,
            content_hash(prototypes),
            teacher.reference_hash,
            tuple(content_hash(score) for score in scores),
            content_hash(winners),
            content_hash(prefixes),
        )
    )
    teacher_identity = f"{teacher.candidate_id}:{teacher.reference_hash}"
    selection = KChampionSelection(
        slot_id=f"k{state_count}",
        state_count=state_count,
        model_family="gaussian_hmm",
        candidate_identity=selected_prefix.candidate_id,
        feature_order=selected_prefix.feature_order,
        feature_order_hash=feature_order_hash(selected_prefix.feature_order),
        source_snapshot_id=source_snapshot_id,
        profile_id=profile.profile_id,
        profile_config_version=profile.profile_config_version,
        policy_id="k_champion_portfolio",
        policy_version="k_champion_portfolio.v1",
        validation_cutoff=validation_cutoff,
        deployment_cutoff=deployment_cutoff,
        comparison_domain_id="k_slot_promotion.v1",
        promotion_score_version="k_slot_promotion.v1",
        reference_teacher_id=teacher_identity,
        artifact_hash=content_hash(
            (
                discovery_hash,
                prefixes.selected_prefix_length,
                prefixes.selected_candidate_id,
                selected_prefix.feature_order,
            )
        ),
        source_build_id=source_build_id,
        source_catalog_hash=catalog.catalog_hash,
    )
    return KFeatureSelectionPayload(
        selection=selection,
        discovery_hash=discovery_hash,
        teacher_identity=teacher_identity,
        prefix_evidence_hash=content_hash(prefixes),
    )


@dataclass(frozen=True, slots=True)
class KFeatureSelectionResult:
    state_count: int
    source_snapshot_id: str
    validation_cutoff: datetime
    selection: KChampionSelection | None
    discovery_hash: str | None
    teacher_identity: str | None
    prefix_evidence_hash: str | None
    eligible: bool
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if self.state_count not in LEGAL_K:
            raise ValueError("K feature selection accepts only K=2,3,4,5")
        if not self.source_snapshot_id.strip():
            raise ValueError("source_snapshot_id cannot be empty")
        _utc(self.validation_cutoff, "validation_cutoff")
        if self.eligible != (self.selection is not None):
            raise ValueError("selection eligibility must match selection presence")
        if self.selection is not None:
            if self.selection.state_count != self.state_count:
                raise ValueError("selector returned a different K")
            if self.selection.source_snapshot_id != self.source_snapshot_id:
                raise ValueError("selection source snapshot differs from task identity")
            if self.selection.validation_cutoff != self.validation_cutoff:
                raise ValueError("selection validation cutoff differs from task identity")
            if (
                not self.discovery_hash
                or not self.teacher_identity
                or not self.prefix_evidence_hash
            ):
                raise ValueError("eligible K selection requires discovery/teacher/prefix hashes")
            if self.rejection_reason is not None:
                raise ValueError("eligible K selection cannot have a rejection reason")
        elif not self.rejection_reason or self.rejection_reason.strip() != self.rejection_reason:
            raise ValueError("ineligible K selection requires a trimmed rejection reason")

    @property
    def selection_hash(self) -> str | None:
        return None if self.selection is None else self.selection.selection_hash


def _run_task(
    task: tuple[pd.DataFrame, int, str, datetime, KTrainSelector],
) -> KFeatureSelectionResult:
    train_rows, state_count, source_snapshot_id, validation_cutoff, selector = task
    try:
        payload = selector(
            train_rows,
            state_count=state_count,
            source_snapshot_id=source_snapshot_id,
            validation_cutoff=validation_cutoff,
        )
        if payload is None:
            return KFeatureSelectionResult(
                state_count,
                source_snapshot_id,
                validation_cutoff,
                None,
                None,
                None,
                None,
                False,
                "K-specific selector returned no eligible result",
            )
        if not isinstance(payload, KFeatureSelectionPayload):
            raise TypeError("K-specific selector must return KFeatureSelectionPayload or None")
        selection = payload.selection
        return KFeatureSelectionResult(
            state_count,
            source_snapshot_id,
            validation_cutoff,
            selection,
            payload.discovery_hash,
            payload.teacher_identity,
            payload.prefix_evidence_hash,
            True,
        )
    except RecoverableEvaluationInvalidity as exc:
        return KFeatureSelectionResult(
            state_count,
            source_snapshot_id,
            validation_cutoff,
            None,
            None,
            None,
            None,
            False,
            f"{type(exc).__name__}: {exc}",
        )


def run_k_feature_selection(
    train_rows: pd.DataFrame,
    *,
    source_snapshot_id: str,
    validation_cutoff: datetime,
    selector: KTrainSelector,
    requested_state_counts: tuple[int, ...] = LEGAL_K,
    max_workers: int | None = None,
) -> tuple[KFeatureSelectionResult, ...]:
    """Run one independent TRAIN-only selector per requested K."""

    if not isinstance(train_rows, pd.DataFrame):
        raise TypeError("K feature selection requires a pandas DataFrame")
    if not source_snapshot_id.strip():
        raise ValueError("source_snapshot_id cannot be empty")
    validation_cutoff = _utc(validation_cutoff, "validation_cutoff")
    requested = tuple(requested_state_counts)
    if requested != tuple(sorted(set(requested))) or any(k not in LEGAL_K for k in requested):
        raise ValueError("requested_state_counts must be a unique ordered subset of K=2,3,4,5")
    if not requested:
        raise ValueError("at least one K-specific selector task is required")
    if not is_pickleable(selector) and cpu_worker_count(max_workers, task_count=len(requested)) > 1:
        raise TypeError("parallel K selection requires a pickleable TRAIN-only selector")
    tasks = tuple(
        (train_rows.copy(deep=True), state_count, source_snapshot_id, validation_cutoff, selector)
        for state_count in requested
    )
    workers = cpu_worker_count(max_workers, task_count=len(tasks))
    if workers == 1:
        results = tuple(_run_task(task) for task in tasks)
    else:
        with cpu_process_pool(workers) as executor:
            results = tuple(executor.map(_run_task, tasks))
    return tuple(sorted(results, key=lambda item: item.state_count))


def run_real_k_feature_selection(
    train_rows: pd.DataFrame,
    *,
    catalog: FeatureCatalogSnapshot,
    profile: ModelProfile,
    source_snapshot_id: str,
    source_build_id: str,
    validation_cutoff: datetime,
    deployment_cutoff: datetime,
    requested_state_counts: tuple[int, ...] = LEGAL_K,
    feature_selection_definition_hash: str | None = None,
    feature_selection_execution_hash: str | None = None,
    max_workers: int | None = None,
) -> tuple[KFeatureSelectionResult, ...]:
    """Run the real fixed-K selector with a bounded nested CPU budget."""

    requested = tuple(requested_state_counts)
    if requested != tuple(sorted(set(requested))) or any(k not in LEGAL_K for k in requested):
        raise ValueError("requested_state_counts must be a unique ordered subset of K=2,3,4,5")
    if not requested:
        raise ValueError("at least one K-specific selector task is required")
    outer_workers = cpu_worker_count(max_workers, task_count=len(requested))
    total_budget = cpu_worker_count(max_workers)
    if outer_workers > 1:
        child_limits = nested_worker_limits(total_budget, outer_workers)
        child_workers = min(child_limits)
    else:
        child_workers = total_budget
    selector = partial(
        select_k_specific_feature_configuration,
        catalog=catalog,
        profile=profile,
        source_snapshot_id=source_snapshot_id,
        source_build_id=source_build_id,
        deployment_cutoff=deployment_cutoff,
        feature_selection_definition_hash=feature_selection_definition_hash,
        feature_selection_execution_hash=feature_selection_execution_hash,
        max_workers=child_workers,
    )
    return run_k_feature_selection(
        train_rows,
        source_snapshot_id=source_snapshot_id,
        validation_cutoff=validation_cutoff,
        selector=selector,
        requested_state_counts=requested,
        max_workers=max_workers,
    )


__all__ = [
    "K_FEATURE_SELECTION_VERSION",
    "KFeatureSelectionPayload",
    "KFeatureSelectionResult",
    "run_k_feature_selection",
    "run_real_k_feature_selection",
    "select_k_specific_feature_configuration",
]
