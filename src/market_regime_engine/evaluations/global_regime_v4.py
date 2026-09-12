"""Leak-free adaptive outer policy for the global Xetra regime discovery v4."""

from __future__ import annotations

import multiprocessing
import pickle
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from statistics import fmean, pstdev
from typing import cast

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.walk_forward import (
    AdapterFactory,
    WalkForwardEvaluation,
)
from market_regime_engine.evaluation.walk_forward_splits import (
    WalkForwardFold,
    WalkForwardPlan,
    plan_walk_forward,
)
from market_regime_engine.evaluation_runs.contracts import (
    DatasetSnapshotIdentity,
    EvaluationRunIdentity,
    WorkUnitIdentity,
)
from market_regime_engine.evaluation_runs.hmm_units import HMMSeedCheckpoint
from market_regime_engine.evaluation_runs.snapshot import ArrowDatasetSnapshotStore
from market_regime_engine.evaluation_runs.stages import StageCheckpoint
from market_regime_engine.evaluation_runs.store import SQLiteEvaluationRunStore
from market_regime_engine.evaluations.agreement_v4 import compute_soft_regime_nmi
from market_regime_engine.evaluations.final_v4_grid import (
    FinalV4GridEvaluation,
    evaluate_final_v4_grid,
)
from market_regime_engine.evaluations.final_v4_grid import (
    _candidates as final_candidates,
)
from market_regime_engine.evaluations.process_parallel import cpu_process_pool
from market_regime_engine.evaluations.provisional_teacher import (
    ProvisionalCandidateRunner,
    ProvisionalTeacherEvaluation,
    run_provisional_gaussian_candidate,
    select_provisional_teacher,
)
from market_regime_engine.evaluations.teacher_reference import (
    FrozenTeacherRefit,
    build_provisional_teacher_reference,
    refit_frozen_teacher,
)
from market_regime_engine.feature_discovery.clustering import select_global_clusters
from market_regime_engine.feature_discovery.contracts import (
    AdaptiveEvaluationResult,
    ClusterSolution,
    DistanceMatrixResult,
    FeatureRegimeScore,
    FinalSelectedConfiguration,
    OuterFoldResult,
    PrefixSearchResult,
    PrototypeSet,
    ProvisionalTeacherReference,
    QualityFilterResult,
    content_hash,
)
from market_regime_engine.feature_discovery.distance import global_absolute_spearman_distance
from market_regime_engine.feature_discovery.prefix_search import (
    PrefixCandidateRunner,
    run_prefix_gaussian_candidate,
    search_ranked_prefixes,
)
from market_regime_engine.feature_discovery.prototypes import select_temporary_prototypes
from market_regime_engine.feature_discovery.quality import filter_outer_train_quality
from market_regime_engine.feature_discovery.scoring import score_all_raw_features
from market_regime_engine.feature_discovery.winners import (
    RegimeWinnerSelection,
    select_cluster_winners,
)
from market_regime_engine.features.ports import (
    FeatureCatalogSnapshot,
    FeatureRequest,
    FeatureRow,
    FeatureSnapshot,
    SchemaWideFeatureSource,
)
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile
from market_regime_engine.runtime.cpu import cpu_worker_count
from market_regime_engine.training.adapter_factory import adapter_factory
from market_regime_engine.training.candidate_grid import CandidateRunner as GridCandidateRunner

_TIMESTAMP_COLUMN = "timestamp_m1"
_MIN_OUTER_TEST_SUPPORT = 42


@dataclass(frozen=True, slots=True)
class _OuterProcessContext:
    """Inherited immutable context for one durable outer-fold worker."""

    source_rows: pd.DataFrame
    catalog: FeatureCatalogSnapshot
    profile: ModelProfile
    build_id: str
    outer_runner: PrefixCandidateRunner
    teacher_refitter: Callable[..., FrozenTeacherRefit]
    nested_max_workers: int
    run_store_root: str | None
    run_identity: EvaluationRunIdentity | None


_OUTER_PROCESS_CONTEXT: _OuterProcessContext | None = None


def _initialize_outer_process_context(context: _OuterProcessContext | None = None) -> None:
    global _OUTER_PROCESS_CONTEXT
    if context is not None:
        _OUTER_PROCESS_CONTEXT = context
    if _OUTER_PROCESS_CONTEXT is None:
        raise RuntimeError("outer process context was not initialized")


def _evaluate_outer_fold_process(fold: WalkForwardFold) -> OuterFoldResult:
    """Evaluate one durable outer fold in its own interpreter.

    The default production path uses process workers at the outer-fold level.
    That keeps Python feature-discovery/scoring work outside the GIL while
    reducing nested numerical pools to one lane per process. The eight
    independent folds in a full run then occupy the machine globally instead
    of competing inside one interpreter's thread pool.
    """

    context = _OUTER_PROCESS_CONTEXT
    if context is None:
        raise RuntimeError("outer process context was not initialized")
    if context.run_store_root is None or context.run_identity is None:
        return _evaluate_outer_fold(
            context.source_rows,
            fold,
            catalog=context.catalog,
            profile=context.profile,
            build_id=context.build_id,
            outer_runner=context.outer_runner,
            teacher_refitter=context.teacher_refitter,
            max_workers=context.nested_max_workers,
        )
    store = SQLiteEvaluationRunStore(context.run_store_root)
    unit = WorkUnitIdentity(
        evaluation_run_key=context.run_identity.key,
        unit_type="outer_fold",
        coordinates=(("fold_id", fold.fold_id),),
        unit_parameters=(
            ("test_source_observations", str(fold.test_source_observations)),
            ("train_source_observations", str(fold.train_source_observations)),
        ),
    )
    cached_payload = store.load_completed_work_unit(context.run_identity, unit)
    if cached_payload is not None:
        cached = pickle.loads(cached_payload)
        if not isinstance(cached, OuterFoldResult) or cached.fold_index != fold.fold_index:
            raise ValueError("cached outer-fold payload is incompatible")
        return cached
    if not store.claim_work_unit(context.run_identity, unit):
        raise RuntimeError(f"outer fold work unit is currently claimed: {unit.key}")
    fold_result = _evaluate_outer_fold(
        context.source_rows,
        fold,
        catalog=context.catalog,
        profile=context.profile,
        build_id=context.build_id,
        outer_runner=context.outer_runner,
        teacher_refitter=context.teacher_refitter,
        max_workers=context.nested_max_workers,
        stage_checkpoint=StageCheckpoint(context.run_identity, store, fold.fold_id),
    )
    store.complete_work_unit(
        context.run_identity,
        unit,
        pickle.dumps(fold_result, protocol=pickle.HIGHEST_PROTOCOL),
    )
    return fold_result


@dataclass(frozen=True, slots=True)
class V4ConfigurationSelection:
    """All TRAIN-only evidence and the frozen final candidate for one outer fold."""

    source_build_id: str
    catalog_hash: str
    quality: QualityFilterResult
    distance: DistanceMatrixResult
    clusters: ClusterSolution
    prototypes: PrototypeSet
    teacher_evaluation: ProvisionalTeacherEvaluation
    teacher_reference: ProvisionalTeacherReference
    feature_scores: tuple[FeatureRegimeScore, ...]
    winner_selection: RegimeWinnerSelection
    prefix_search: PrefixSearchResult
    final_grid: FinalV4GridEvaluation
    final_candidate: ResolvedCandidateProfile
    feature_discovery_hash: str
    final_grid_plan: WalkForwardPlan | None = None

    def __post_init__(self) -> None:
        if self.source_build_id != self.quality.source_build_id:
            raise ValueError("v4 selection source build differs from quality evidence")
        if self.catalog_hash != self.quality.catalog_hash:
            raise ValueError("v4 selection catalog hash differs from quality evidence")
        if self.final_grid.selection is None:
            raise ValueError("v4 selection requires a final statistical champion")
        if self.final_candidate.candidate_id != self.final_grid.selection.champion_candidate_id:
            raise ValueError("v4 final candidate differs from final-grid champion")
        if (
            self.final_candidate.feature_order
            != self.prefix_search.evaluations[
                self.prefix_search.selected_prefix_length - 2
            ].feature_order
        ):
            raise ValueError("v4 final candidate does not preserve the selected prefix")
        if len(self.feature_discovery_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.feature_discovery_hash
        ):
            raise ValueError("v4 feature discovery hash must be a lowercase SHA-256")


def _utc(value: object, field_name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != UTC.utcoffset(value)
    ):
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    return value


def _as_feature_snapshot(
    train_rows: pd.DataFrame,
    catalog: FeatureCatalogSnapshot,
) -> FeatureSnapshot:
    required = (_TIMESTAMP_COLUMN, *catalog.feature_names)
    missing = tuple(name for name in required if name not in train_rows.columns)
    if missing:
        raise ValueError(f"TRAIN rows are missing catalog columns: {missing}")
    rows: list[FeatureRow] = []
    for row_index in range(len(train_rows)):
        timestamp = _utc(train_rows.iloc[row_index][_TIMESTAMP_COLUMN], "TRAIN timestamp")
        values: list[float | None] = []
        for feature in catalog.feature_names:
            value = train_rows.iloc[row_index][feature]
            if value is None or value is pd.NA:
                values.append(None)
                continue
            try:
                missing_value = bool(pd.isna(value))
            except TypeError, ValueError:
                missing_value = False
            values.append(None if missing_value else float(value))
        rows.append(FeatureRow(timestamp, tuple(values)))
    if not rows:
        raise ValueError("TRAIN rows cannot be empty")
    return FeatureSnapshot(
        lineage=catalog.lineage,
        feature_names=catalog.feature_names,
        rows=tuple(rows),
    )


def _validate_train_inputs(
    train_rows: pd.DataFrame,
    catalog: FeatureCatalogSnapshot,
    profile: ModelProfile,
    source_build_id: str,
) -> FeatureSnapshot:
    if not isinstance(train_rows, pd.DataFrame):
        raise TypeError("v4 configuration selection requires a pandas DataFrame")
    if not isinstance(catalog, FeatureCatalogSnapshot):
        raise TypeError("v4 configuration selection requires a feature catalog")
    if profile.profile_id != "xetra" or profile.profile_config_version != 4:
        raise ValueError("v4 configuration selection requires the canonical Xetra v4 profile")
    if source_build_id != catalog.lineage.source_build_id:
        raise ValueError("v4 selection source build differs from catalog lineage")
    if len(catalog.feature_names) < 3:
        raise ValueError("v4 configuration selection requires at least three catalog features")
    return _as_feature_snapshot(train_rows, catalog)


def _selection_hashes(
    profile: ModelProfile,
    catalog: FeatureCatalogSnapshot,
    train_end: datetime,
    definition_hash: str | None,
    execution_hash: str | None,
) -> tuple[str, str]:
    definition = definition_hash or content_hash(
        ("xetra_global_regime_v4", "selection_definition", profile.profile_hash)
    )
    execution = execution_hash or content_hash(
        (
            "xetra_global_regime_v4",
            "selection_execution",
            catalog.catalog_hash,
            train_end,
        )
    )
    return definition, execution


def _configuration_hash(
    catalog: FeatureCatalogSnapshot,
    quality: QualityFilterResult,
    distance: DistanceMatrixResult,
    clusters: ClusterSolution,
    prototypes: PrototypeSet,
    teacher: ProvisionalTeacherReference,
    scores: tuple[FeatureRegimeScore, ...],
    winners: RegimeWinnerSelection,
    prefixes: PrefixSearchResult,
    final_grid: FinalV4GridEvaluation,
) -> str:
    return content_hash(
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
            content_hash(final_grid.candidate_grid),
            content_hash(final_grid.selection),
        )
    )


def select_v4_configuration(
    train_rows: pd.DataFrame,
    *,
    catalog: FeatureCatalogSnapshot,
    profile: ModelProfile,
    source_build_id: str | None = None,
    feature_selection_definition_hash: str | None = None,
    feature_selection_execution_hash: str | None = None,
    teacher_runner: ProvisionalCandidateRunner | None = None,
    prefix_runner: PrefixCandidateRunner | None = None,
    grid_runner: GridCandidateRunner | None = None,
    max_workers: int | None = None,
    stage_checkpoint: StageCheckpoint | None = None,
) -> V4ConfigurationSelection:
    """Run the complete adaptive chain using only one immutable Outer-TRAIN frame."""

    build_id = catalog.lineage.source_build_id if source_build_id is None else source_build_id
    snapshot = _validate_train_inputs(train_rows, catalog, profile, build_id)
    train_start = snapshot.rows[0].timestamp
    train_end = snapshot.rows[-1].timestamp
    definition_hash, execution_hash = _selection_hashes(
        profile,
        catalog,
        train_end,
        feature_selection_definition_hash,
        feature_selection_execution_hash,
    )

    def checkpoint(
        stage: str,
        compute: Callable[[], object],
        *,
        parents: tuple[object, ...] = (),
    ) -> object:
        if stage_checkpoint is None:
            return compute()
        return stage_checkpoint.run(
            stage,
            compute,
            parameters=(
                ("catalog_hash", catalog.catalog_hash),
                ("source_build_id", build_id),
                ("train_end", train_end.isoformat()),
            ),
            parent_payloads=tuple(
                pickle.dumps(parent, protocol=pickle.HIGHEST_PROTOCOL) for parent in parents
            ),
        )

    seed_checkpoint_factory = None
    if stage_checkpoint is not None:

        def make_seed_checkpoint(
            candidate_id: str,
            fold_id: str,
            state_count: int,
        ) -> HMMSeedCheckpoint:
            return HMMSeedCheckpoint(
                run_identity=stage_checkpoint.identity,
                store=stage_checkpoint.store,
                candidate_id=candidate_id,
                fold_id=fold_id,
                state_count=state_count,
            )

        seed_checkpoint_factory = make_seed_checkpoint

    quality = cast(
        QualityFilterResult,
        checkpoint(
            "quality",
            lambda: filter_outer_train_quality(catalog, snapshot, train_start, train_end),
        ),
    )
    distance = cast(
        DistanceMatrixResult,
        checkpoint(
            "distance",
            lambda: global_absolute_spearman_distance(snapshot, quality),
            parents=(quality,),
        ),
    )
    clusters = cast(
        ClusterSolution,
        checkpoint("clusters", lambda: select_global_clusters(distance), parents=(distance,)),
    )
    prototypes = cast(
        PrototypeSet,
        checkpoint(
            "prototypes",
            lambda: select_temporary_prototypes(clusters, distance),
            parents=(clusters, distance),
        ),
    )
    teacher_evaluation = cast(
        ProvisionalTeacherEvaluation,
        checkpoint(
            "teacher",
            lambda: select_provisional_teacher(
                train_rows,
                profile=profile,
                prototype_features=prototypes.prototypes,
                source_build_id=build_id,
                feature_selection_definition_hash=definition_hash,
                feature_selection_execution_hash=execution_hash,
                runner=teacher_runner
                if teacher_runner is not None
                else run_provisional_gaussian_candidate,
                max_workers=max_workers,
                seed_checkpoint_factory=seed_checkpoint_factory,
            ),
            parents=(prototypes,),
        ),
    )
    teacher_reference = cast(
        ProvisionalTeacherReference,
        checkpoint(
            "teacher_reference",
            lambda: build_provisional_teacher_reference(teacher_evaluation),
            parents=(teacher_evaluation,),
        ),
    )
    feature_scores = cast(
        tuple[FeatureRegimeScore, ...],
        checkpoint(
            "feature_scores",
            lambda: score_all_raw_features(snapshot, quality, teacher_reference),
            parents=(quality, teacher_reference),
        ),
    )
    winner_selection = cast(
        RegimeWinnerSelection,
        checkpoint(
            "winners",
            lambda: select_cluster_winners(clusters, feature_scores),
            parents=(clusters, feature_scores),
        ),
    )
    prefix_search = cast(
        PrefixSearchResult,
        checkpoint(
            "prefix_search",
            lambda: search_ranked_prefixes(
                train_rows,
                ranked_features=winner_selection.ranked_features,
                teacher=teacher_reference,
                profile=profile,
                inner_plan=teacher_evaluation.inner_plan,
                source_build_id=build_id,
                original_feature_universe=catalog.feature_names,
                feature_selection_definition_hash=definition_hash,
                feature_selection_execution_hash=execution_hash,
                runner=prefix_runner
                if prefix_runner is not None
                else run_prefix_gaussian_candidate,
                max_workers=max_workers,
                seed_checkpoint_factory=seed_checkpoint_factory,
            ),
            parents=(winner_selection, teacher_reference),
        ),
    )
    selected_prefix = prefix_search.evaluations[prefix_search.selected_prefix_length - 2]
    final_grid = cast(
        FinalV4GridEvaluation,
        checkpoint(
            "final_grid",
            lambda: evaluate_final_v4_grid(
                train_rows,
                feature_order=selected_prefix.feature_order,
                profile=profile,
                plan=teacher_evaluation.inner_plan,
                source_build_id=build_id,
                original_feature_universe=catalog.feature_names,
                feature_selection_definition_hash=definition_hash,
                feature_selection_execution_hash=execution_hash,
                runner=grid_runner,
                max_workers=max_workers,
                seed_checkpoint_factory=seed_checkpoint_factory,
            ),
            parents=(prefix_search,),
        ),
    )
    if final_grid.selection is None:
        raise ValueError(
            f"final v4 grid has no statistical champion: {final_grid.no_champion_reason}"
        )
    candidates = final_candidates(
        selected_prefix.feature_order,
        original_feature_universe=catalog.feature_names,
        source_build_id=build_id,
        feature_selection_definition_hash=definition_hash,
        feature_selection_execution_hash=execution_hash,
    )
    final_candidate = next(
        candidate
        for candidate in candidates
        if candidate.candidate_id == final_grid.selection.champion_candidate_id
    )
    discovery_hash = _configuration_hash(
        catalog,
        quality,
        distance,
        clusters,
        prototypes,
        teacher_reference,
        feature_scores,
        winner_selection,
        prefix_search,
        final_grid,
    )
    return V4ConfigurationSelection(
        source_build_id=build_id,
        catalog_hash=catalog.catalog_hash,
        quality=quality,
        distance=distance,
        clusters=clusters,
        prototypes=prototypes,
        teacher_evaluation=teacher_evaluation,
        teacher_reference=teacher_reference,
        feature_scores=feature_scores,
        winner_selection=winner_selection,
        prefix_search=prefix_search,
        final_grid=final_grid,
        final_candidate=final_candidate,
        feature_discovery_hash=discovery_hash,
        final_grid_plan=teacher_evaluation.inner_plan,
    )


def _outer_fold_plan(fold: WalkForwardFold) -> WalkForwardPlan:
    # The existing runner validates fold positions relative to the supplied
    # plan.  A one-fold execution therefore uses a local fold identity while
    # retaining the outer fold's exact timestamps and source-row bounds.
    local_fold = WalkForwardFold(
        fold_index=1,
        fold_id="fold_001",
        train_start=fold.train_start,
        train_end=fold.train_end,
        test_start=fold.test_start,
        test_end=fold.test_end,
        train_source_observations=fold.train_source_observations,
        test_source_observations=fold.test_source_observations,
    )
    return WalkForwardPlan(
        folds=(local_fold,),
        evaluation_cutoff=fold.test_end,
        plan_hash=content_hash(
            (
                "xetra_outer_fold_v4",
                fold.fold_id,
                fold.train_start,
                fold.train_end,
                fold.test_start,
                fold.test_end,
                fold.train_source_observations,
                fold.test_source_observations,
            )
        ),
    )


def _fallback_configuration(
    catalog: FeatureCatalogSnapshot,
    source_build_id: str,
    fold: WalkForwardFold,
    reason: str,
) -> FinalSelectedConfiguration:
    features = catalog.feature_names[:2]
    if len(features) != 2:
        raise ValueError("cannot create a failure configuration without two catalog features")
    return FinalSelectedConfiguration(
        feature_order=features,
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        model_family="gaussian_hmm",
        selected_prefix_length=2,
        feature_discovery_hash=content_hash(("failed_outer_fold", fold.fold_id, reason)),
        source_build_id=source_build_id,
    )


def _invalid_outer_fold(
    fold: WalkForwardFold,
    configuration: FinalSelectedConfiguration,
    reason: str,
) -> OuterFoldResult:
    return OuterFoldResult(
        fold_index=fold.fold_index,
        train_start=fold.train_start,
        train_end=fold.train_end,
        test_start=fold.test_start,
        test_end=fold.test_end,
        final_configuration=configuration,
        oos_predictive_loglik_per_observation=0.0,
        oos_timestamps=(),
        oos_filtered_probabilities=(),
        outer_teacher_final_soft_nmi=None,
        outer_shared_timestamp_count=0,
        valid=False,
        failure_reason=reason,
    )


def _evaluate_outer_fold(
    source_rows: pd.DataFrame,
    fold: WalkForwardFold,
    *,
    catalog: FeatureCatalogSnapshot,
    profile: ModelProfile,
    build_id: str,
    outer_runner: PrefixCandidateRunner,
    teacher_refitter: Callable[..., FrozenTeacherRefit],
    max_workers: int | None,
    selection_sink: Callable[[int, V4ConfigurationSelection], None] | None = None,
    stage_checkpoint: StageCheckpoint | None = None,
) -> OuterFoldResult:
    """Evaluate one outer fold; callers may persist this atomic result."""

    train_rows = source_rows.iloc[: fold.train_source_observations].copy()
    test_rows = source_rows.iloc[
        fold.train_source_observations : fold.train_source_observations
        + fold.test_source_observations
    ].copy()
    try:
        selection = select_v4_configuration(
            train_rows,
            catalog=catalog,
            profile=profile,
            source_build_id=build_id,
            max_workers=max_workers,
            stage_checkpoint=stage_checkpoint,
        )
    except (ValueError, TypeError) as exc:
        return _invalid_outer_fold(
            fold,
            _fallback_configuration(catalog, build_id, fold, str(exc)),
            f"TRAIN-only v4 selection failed: {type(exc).__name__}: {exc}",
        )
    if selection_sink is not None:
        selection_sink(fold.fold_index, selection)

    configuration = FinalSelectedConfiguration(
        feature_order=selection.final_candidate.feature_order,
        candidate_id=selection.final_candidate.candidate_id,
        state_count=selection.final_candidate.state_count,
        model_family=selection.final_candidate.model_family,
        selected_prefix_length=len(selection.final_candidate.feature_order),
        feature_discovery_hash=selection.feature_discovery_hash,
        source_build_id=build_id,
        catalog_hash=selection.catalog_hash,
        selection_definition_hash=selection.final_candidate.feature_selection_definition_hash,
        selection_execution_hash=selection.final_candidate.feature_selection_execution_hash,
    )
    outer_source = source_rows.iloc[
        : fold.train_source_observations + fold.test_source_observations
    ].copy()
    try:
        model_evaluation = outer_runner(
            outer_source,
            _outer_fold_plan(fold),
            profile,
            selection.final_candidate,
            cast(AdapterFactory, adapter_factory(profile, selection.final_candidate)),
        )
        teacher_refit = teacher_refitter(
            train_rows,
            test_rows,
            reference=selection.teacher_reference,
            profile=profile,
        )
        return _valid_outer_fold(fold, configuration, model_evaluation, teacher_refit)
    except (ValueError, TypeError) as exc:
        return _invalid_outer_fold(
            fold,
            configuration,
            f"outer refit/TEST continuation failed: {type(exc).__name__}: {exc}",
        )


def _valid_outer_fold(
    fold: WalkForwardFold,
    configuration: FinalSelectedConfiguration,
    model_evaluation: WalkForwardEvaluation,
    teacher_refit: FrozenTeacherRefit,
) -> OuterFoldResult:
    valid_model_folds = model_evaluation.valid_folds
    if len(valid_model_folds) != 1:
        return _invalid_outer_fold(
            fold,
            configuration,
            "final outer candidate did not produce exactly one valid fold",
        )
    model_fold = valid_model_folds[0]
    if model_fold.oos_predictive_log_likelihood_per_observation is None:
        return _invalid_outer_fold(fold, configuration, "final outer candidate has no OOS score")
    agreement = compute_soft_regime_nmi(
        model_fold.oos_timestamps,
        model_fold.oos_filtered_probabilities,
        teacher_refit.test_timestamps,
        teacher_refit.test_filtered_probabilities,
    )
    if agreement.shared_timestamp_count < _MIN_OUTER_TEST_SUPPORT:
        return _invalid_outer_fold(
            fold,
            configuration,
            "outer teacher shared support "
            f"{agreement.shared_timestamp_count} below {_MIN_OUTER_TEST_SUPPORT}",
        )
    return OuterFoldResult(
        fold_index=fold.fold_index,
        train_start=fold.train_start,
        train_end=fold.train_end,
        test_start=fold.test_start,
        test_end=fold.test_end,
        final_configuration=configuration,
        oos_predictive_loglik_per_observation=float(
            model_fold.oos_predictive_log_likelihood_per_observation
        ),
        oos_timestamps=model_fold.oos_timestamps,
        oos_filtered_probabilities=model_fold.oos_filtered_probabilities,
        teacher_reference_hash=content_hash(teacher_refit.model_artifact),
        outer_teacher_final_soft_nmi=agreement.soft_regime_nmi,
        outer_shared_timestamp_count=agreement.shared_timestamp_count,
    )


def evaluate_global_regime_v4(
    source_rows: pd.DataFrame,
    *,
    catalog: FeatureCatalogSnapshot,
    profile: ModelProfile,
    source_build_id: str | None = None,
    outer_runner: PrefixCandidateRunner = run_prefix_gaussian_candidate,
    teacher_refitter: Callable[..., FrozenTeacherRefit] = refit_frozen_teacher,
    max_workers: int | None = None,
    run_store: SQLiteEvaluationRunStore | None = None,
    run_identity: EvaluationRunIdentity | None = None,
    selection_sink: Callable[[int, V4ConfigurationSelection], None] | None = None,
) -> AdaptiveEvaluationResult:
    """Run every outer fold with TRAIN-only adaptive selection and frozen TEST use."""

    if (run_store is None) != (run_identity is None):
        raise ValueError("run_store and run_identity must be supplied together")
    if not isinstance(source_rows, pd.DataFrame):
        raise TypeError("global v4 evaluation requires a pandas DataFrame")
    if run_store is not None and run_identity is not None:
        state = run_store.open_run(run_identity)
        run_store.enable_write_ahead_logging()
        if state.status == "COMPLETE" and selection_sink is None:
            payload = run_store.load_completed_run(run_identity)
            if payload is None:
                raise ValueError("completed evaluation has no durable result payload")
            cached = pickle.loads(payload)
            if not isinstance(cached, AdaptiveEvaluationResult):
                raise ValueError("durable evaluation result has an invalid type")
            if cached.result_hash != state.root_identity_hash:
                raise ValueError("durable evaluation result hash does not match its ledger")
            return cached
    build_id = catalog.lineage.source_build_id if source_build_id is None else source_build_id
    if build_id != catalog.lineage.source_build_id:
        raise ValueError("global v4 source build differs from catalog lineage")
    if _TIMESTAMP_COLUMN not in source_rows.columns:
        raise ValueError(f"source rows must contain {_TIMESTAMP_COLUMN}")
    timestamps = tuple(_utc(value, "source timestamp") for value in source_rows[_TIMESTAMP_COLUMN])
    if any(current <= previous for previous, current in pairwise(timestamps)):
        raise ValueError("source timestamps must be strictly increasing and unique")
    outer_plan = plan_walk_forward(timestamps, profile.walk_forward)
    requested_workers = cpu_worker_count(max_workers, task_count=len(outer_plan.folds))
    outer_worker_limit = requested_workers

    def evaluate_fold(fold: WalkForwardFold) -> OuterFoldResult:
        if run_store is None or run_identity is None:
            return _evaluate_outer_fold(
                source_rows,
                fold,
                catalog=catalog,
                profile=profile,
                build_id=build_id,
                outer_runner=outer_runner,
                teacher_refitter=teacher_refitter,
                max_workers=max_workers,
                selection_sink=selection_sink,
            )

        unit = WorkUnitIdentity(
            evaluation_run_key=run_identity.key,
            unit_type="outer_fold",
            coordinates=(("fold_id", fold.fold_id),),
            unit_parameters=(
                ("test_source_observations", str(fold.test_source_observations)),
                ("train_source_observations", str(fold.train_source_observations)),
            ),
        )
        cached_payload = run_store.load_completed_work_unit(run_identity, unit)
        if cached_payload is not None:
            cached = pickle.loads(cached_payload)
            if not isinstance(cached, OuterFoldResult) or cached.fold_index != fold.fold_index:
                raise ValueError("cached outer-fold payload is incompatible")
            if not cached.valid:
                run_store.terminalize_pending_stage_units_for_invalid_outer_fold(
                    run_identity,
                    fold.fold_id,
                    cached.failure_reason or "cached outer fold is domain-invalid",
                )
            if selection_sink is not None and cached.valid:
                train_rows = source_rows.iloc[: fold.train_source_observations].copy()
                selection = select_v4_configuration(
                    train_rows,
                    catalog=catalog,
                    profile=profile,
                    source_build_id=build_id,
                    max_workers=max_workers,
                    stage_checkpoint=StageCheckpoint(run_identity, run_store, fold.fold_id),
                )
                selection_sink(fold.fold_index, selection)
            return cached
        if not run_store.claim_work_unit(run_identity, unit):
            raise RuntimeError(f"outer fold work unit is currently claimed: {unit.key}")
        fold_result = _evaluate_outer_fold(
            source_rows,
            fold,
            catalog=catalog,
            profile=profile,
            build_id=build_id,
            outer_runner=outer_runner,
            teacher_refitter=teacher_refitter,
            max_workers=max_workers,
            selection_sink=selection_sink,
            stage_checkpoint=StageCheckpoint(run_identity, run_store, fold.fold_id),
        )
        run_store.complete_work_unit(
            run_identity,
            unit,
            pickle.dumps(fold_result, protocol=pickle.HIGHEST_PROTOCOL),
        )
        return fold_result

    use_process_outer = (
        run_store is not None
        and run_identity is not None
        and _evaluate_outer_fold.__module__ == __name__
        and outer_runner is run_prefix_gaussian_candidate
        and teacher_refitter is refit_frozen_teacher
        and outer_worker_limit > 1
    )
    use_process_outer_without_ledger = (
        run_store is None
        and run_identity is None
        and outer_runner is run_prefix_gaussian_candidate
        and teacher_refitter is refit_frozen_teacher
        and outer_worker_limit > 1
    )
    if use_process_outer:
        assert run_store is not None
        assert run_identity is not None
        context = _OuterProcessContext(
            source_rows=source_rows,
            catalog=catalog,
            profile=profile,
            build_id=build_id,
            outer_runner=outer_runner,
            teacher_refitter=teacher_refitter,
            nested_max_workers=1,
            run_store_root=str(run_store.root),
            run_identity=run_identity,
        )
        available_methods = multiprocessing.get_all_start_methods()
        if "fork" in available_methods and threading.current_thread() is threading.main_thread():
            _initialize_outer_process_context(context)
            with cpu_process_pool(
                max_workers=outer_worker_limit,
                initializer=_initialize_outer_process_context,
            ) as process_executor:
                futures = [
                    process_executor.submit(_evaluate_outer_fold_process, fold)
                    for fold in outer_plan.folds
                ]
                outer_results = []
                for fold, future in zip(outer_plan.folds, futures, strict=True):
                    fold_result = future.result()
                    if selection_sink is not None and fold_result.valid:
                        train_rows = source_rows.iloc[: fold.train_source_observations].copy()
                        selection = select_v4_configuration(
                            train_rows,
                            catalog=catalog,
                            profile=profile,
                            source_build_id=build_id,
                            max_workers=1,
                            stage_checkpoint=StageCheckpoint(run_identity, run_store, fold.fold_id),
                        )
                        selection_sink(fold.fold_index, selection)
                    outer_results.append(fold_result)
        else:
            with cpu_process_pool(
                max_workers=outer_worker_limit,
                initializer=_initialize_outer_process_context,
                initargs=(context,),
            ) as process_executor:
                futures = [
                    process_executor.submit(_evaluate_outer_fold_process, fold)
                    for fold in outer_plan.folds
                ]
                outer_results = []
                for fold, future in zip(outer_plan.folds, futures, strict=True):
                    fold_result = future.result()
                    if selection_sink is not None and fold_result.valid:
                        train_rows = source_rows.iloc[: fold.train_source_observations].copy()
                        selection = select_v4_configuration(
                            train_rows,
                            catalog=catalog,
                            profile=profile,
                            source_build_id=build_id,
                            max_workers=1,
                            stage_checkpoint=StageCheckpoint(run_identity, run_store, fold.fold_id),
                        )
                        selection_sink(fold.fold_index, selection)
                    outer_results.append(fold_result)
    elif use_process_outer_without_ledger:
        context = _OuterProcessContext(
            source_rows=source_rows,
            catalog=catalog,
            profile=profile,
            build_id=build_id,
            outer_runner=outer_runner,
            teacher_refitter=teacher_refitter,
            nested_max_workers=1,
            run_store_root=None,
            run_identity=None,
        )
        _initialize_outer_process_context(context)
        with cpu_process_pool(
            outer_worker_limit,
            initializer=_initialize_outer_process_context,
            initargs=(context,),
        ) as process_executor:
            futures = [
                process_executor.submit(_evaluate_outer_fold_process, fold)
                for fold in outer_plan.folds
            ]
            outer_results = []
            for fold, future in zip(outer_plan.folds, futures, strict=True):
                fold_result = future.result()
                if selection_sink is not None and fold_result.valid:
                    train_rows = source_rows.iloc[: fold.train_source_observations].copy()
                    selection = select_v4_configuration(
                        train_rows,
                        catalog=catalog,
                        profile=profile,
                        source_build_id=build_id,
                        max_workers=1,
                    )
                    selection_sink(fold.fold_index, selection)
                outer_results.append(fold_result)
    elif outer_worker_limit == 1:
        outer_results = [evaluate_fold(fold) for fold in outer_plan.folds]
    else:
        with ThreadPoolExecutor(max_workers=outer_worker_limit) as executor:
            futures = [executor.submit(evaluate_fold, fold) for fold in outer_plan.folds]
            # Result order is part of the evaluation contract, independent of
            # completion order and scheduler timing.
            outer_results = [future.result() for future in futures]

    valid_folds = tuple(fold for fold in outer_results if fold.valid)
    nmi_values = tuple(
        fold.outer_teacher_final_soft_nmi
        for fold in valid_folds
        if fold.outer_teacher_final_soft_nmi is not None
    )
    valid_count = len(valid_folds)
    valid_rate = valid_count / len(outer_results)
    latest_valid = bool(outer_results[-1].valid)
    evaluation_result = AdaptiveEvaluationResult(
        source_build_id=build_id,
        catalog_hash=catalog.catalog_hash,
        validation_evaluation_cutoff=cast(datetime, outer_plan.evaluation_cutoff),
        outer_folds=tuple(outer_results),
        valid_fold_count=valid_count,
        valid_fold_rate=valid_rate,
        soft_nmi_mean=fmean(nmi_values) if nmi_values else None,
        soft_nmi_population_std=pstdev(nmi_values) if nmi_values else None,
        soft_nmi_worst=min(nmi_values) if nmi_values else None,
        latest_complete_fold_valid=latest_valid,
        production_eligible=(valid_rate >= 0.80 and valid_count >= 3 and latest_valid),
        policy_hash=content_hash(("xetra_global_regime_v4", profile.profile_hash)),
        failure_reason=(
            None
            if valid_count == len(outer_results)
            else "one or more outer folds failed without reusing a prior configuration"
        ),
    )
    if run_store is not None and run_identity is not None:
        run_store.complete_run(
            run_identity,
            evaluation_result.result_hash,
            pickle.dumps(evaluation_result, protocol=pickle.HIGHEST_PROTOCOL),
        )
    return evaluation_result


def evaluate_global_regime_v4_from_source(
    source: SchemaWideFeatureSource,
    *,
    profile: ModelProfile,
    start: datetime | None = None,
    end: datetime | None = None,
    source_build_id: str | None = None,
    outer_runner: PrefixCandidateRunner = run_prefix_gaussian_candidate,
    teacher_refitter: Callable[..., FrozenTeacherRefit] = refit_frozen_teacher,
    max_workers: int | None = None,
    snapshot_store: ArrowDatasetSnapshotStore | None = None,
    run_store: SQLiteEvaluationRunStore | None = None,
    repository_commit_sha: str | None = None,
    uv_lock_sha256: str | None = None,
    python_version: str | None = None,
    evaluation_contract_version: int = 1,
    selection_sink: Callable[[int, V4ConfigurationSelection], None] | None = None,
) -> AdaptiveEvaluationResult:
    """Capture the complete dynamic source universe and run v4 on that snapshot.

    The empty feature request is intentional: the source, not the caller or a
    profile YAML list, determines the raw evaluation universe.  The database
    transaction is closed by the source before any clustering or HMM fitting
    starts, so subsequent schema changes cannot affect this evaluation.
    """

    if profile.profile_id != "xetra" or profile.profile_config_version != 4:
        raise ValueError("dynamic source evaluation requires the canonical Xetra v4 profile")
    if (snapshot_store is None) != (run_store is None):
        raise ValueError("snapshot_store and run_store must be supplied together")
    catalog, snapshot = source.read_schema_wide_with_catalog(
        FeatureRequest.all_features(start, end)
    )
    if snapshot.feature_names != catalog.feature_names:
        raise ValueError("dynamic source snapshot columns do not match its catalog")
    if snapshot.materialized_feature_data_sha256 is None:
        raise ValueError("dynamic source snapshot is missing its materialization digest")
    if catalog.materialized_feature_data_sha256 != snapshot.materialized_feature_data_sha256:
        raise ValueError("dynamic source catalog and snapshot materialization digests differ")
    if not snapshot.rows:
        raise ValueError("dynamic source snapshot contains no rows")
    run_identity: EvaluationRunIdentity | None = None
    if snapshot_store is not None and run_store is not None:
        if repository_commit_sha is None or uv_lock_sha256 is None or python_version is None:
            raise ValueError(
                "durable source evaluation requires repository, lockfile and Python identities"
            )
        dataset_identity = DatasetSnapshotIdentity.from_catalog(catalog)
        snapshot_store.finalize(dataset_identity, snapshot, catalog=catalog)
        snapshot = snapshot_store.load(dataset_identity)
        timestamps = tuple(row.timestamp for row in snapshot.rows)
        plan = plan_walk_forward(timestamps, profile.walk_forward)
        run_identity = EvaluationRunIdentity(
            evaluation_id="global_regime_v4",
            profile_id=profile.profile_id,
            profile_config_version=profile.profile_config_version,
            profile_hash=profile.profile_hash,
            evaluation_contract_version=evaluation_contract_version,
            evaluation_plan_hash=plan.plan_hash,
            dataset_snapshot_key=dataset_identity.key,
            evaluation_cutoff=cast(datetime, plan.evaluation_cutoff),
            repository_commit_sha=repository_commit_sha,
            uv_lock_sha256=uv_lock_sha256,
            python_version=python_version,
        )
        state = run_store.open_run(run_identity)
        if state.status == "COMPLETE":
            cached_result_payload = run_store.load_completed_run(run_identity)
            if cached_result_payload is None:
                raise ValueError("completed evaluation has no durable result payload")
            cached_result = pickle.loads(cached_result_payload)
            if (
                not isinstance(cached_result, AdaptiveEvaluationResult)
                or cached_result.result_hash != state.root_identity_hash
            ):
                raise ValueError("durable evaluation result is incompatible or corrupted")
            return cached_result
    rows = pd.DataFrame(
        [row.values for row in snapshot.rows],
        columns=snapshot.feature_names,
    )
    rows.insert(0, _TIMESTAMP_COLUMN, [row.timestamp for row in snapshot.rows])
    return evaluate_global_regime_v4(
        rows,
        catalog=catalog,
        profile=profile,
        source_build_id=source_build_id,
        outer_runner=outer_runner,
        teacher_refitter=teacher_refitter,
        max_workers=max_workers,
        run_store=run_store,
        run_identity=run_identity,
        selection_sink=selection_sink,
    )


run_global_regime_v4 = evaluate_global_regime_v4
evaluate_adaptive_global_regime = evaluate_global_regime_v4


__all__ = [
    "V4ConfigurationSelection",
    "evaluate_adaptive_global_regime",
    "evaluate_global_regime_v4",
    "evaluate_global_regime_v4_from_source",
    "run_global_regime_v4",
    "select_v4_configuration",
]
