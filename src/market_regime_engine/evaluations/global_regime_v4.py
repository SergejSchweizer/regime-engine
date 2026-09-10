"""Leak-free adaptive outer policy for the global Xetra regime discovery v4."""

from __future__ import annotations

from collections.abc import Callable
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
from market_regime_engine.evaluations.agreement_v4 import compute_soft_regime_nmi
from market_regime_engine.evaluations.final_v4_grid import (
    FinalV4GridEvaluation,
    evaluate_final_v4_grid,
)
from market_regime_engine.evaluations.final_v4_grid import (
    _candidates as final_candidates,
)
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
from market_regime_engine.training.adapter_factory import adapter_factory
from market_regime_engine.training.candidate_grid import CandidateRunner as GridCandidateRunner

_TIMESTAMP_COLUMN = "timestamp_m1"
_MIN_OUTER_TEST_SUPPORT = 42


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
    quality = filter_outer_train_quality(catalog, snapshot, train_start, train_end)
    distance = global_absolute_spearman_distance(snapshot, quality)
    clusters = select_global_clusters(distance)
    prototypes = select_temporary_prototypes(clusters, distance)
    teacher_evaluation = select_provisional_teacher(
        train_rows,
        profile=profile,
        prototype_features=prototypes.prototypes,
        source_build_id=build_id,
        feature_selection_definition_hash=definition_hash,
        feature_selection_execution_hash=execution_hash,
        runner=teacher_runner if teacher_runner is not None else run_provisional_gaussian_candidate,
        max_workers=max_workers,
    )
    teacher_reference = build_provisional_teacher_reference(teacher_evaluation)
    feature_scores = score_all_raw_features(snapshot, quality, teacher_reference)
    winner_selection = select_cluster_winners(clusters, feature_scores)
    prefix_search = search_ranked_prefixes(
        train_rows,
        ranked_features=winner_selection.ranked_features,
        teacher=teacher_reference,
        profile=profile,
        inner_plan=teacher_evaluation.inner_plan,
        source_build_id=build_id,
        original_feature_universe=catalog.feature_names,
        feature_selection_definition_hash=definition_hash,
        feature_selection_execution_hash=execution_hash,
        runner=prefix_runner if prefix_runner is not None else run_prefix_gaussian_candidate,
        max_workers=max_workers,
    )
    selected_prefix = prefix_search.evaluations[prefix_search.selected_prefix_length - 2]
    final_grid = evaluate_final_v4_grid(
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
) -> AdaptiveEvaluationResult:
    """Run every outer fold with TRAIN-only adaptive selection and frozen TEST use."""

    if not isinstance(source_rows, pd.DataFrame):
        raise TypeError("global v4 evaluation requires a pandas DataFrame")
    build_id = catalog.lineage.source_build_id if source_build_id is None else source_build_id
    if build_id != catalog.lineage.source_build_id:
        raise ValueError("global v4 source build differs from catalog lineage")
    if _TIMESTAMP_COLUMN not in source_rows.columns:
        raise ValueError(f"source rows must contain {_TIMESTAMP_COLUMN}")
    timestamps = tuple(_utc(value, "source timestamp") for value in source_rows[_TIMESTAMP_COLUMN])
    if any(current <= previous for previous, current in pairwise(timestamps)):
        raise ValueError("source timestamps must be strictly increasing and unique")
    outer_plan = plan_walk_forward(timestamps, profile.walk_forward)
    outer_results: list[OuterFoldResult] = []
    for fold in outer_plan.folds:
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
            )
        except (ValueError, TypeError) as exc:
            outer_results.append(
                _invalid_outer_fold(
                    fold,
                    _fallback_configuration(catalog, build_id, fold, str(exc)),
                    f"TRAIN-only v4 selection failed: {type(exc).__name__}: {exc}",
                )
            )
            continue

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
            outer_results.append(
                _valid_outer_fold(fold, configuration, model_evaluation, teacher_refit)
            )
        except (ValueError, TypeError) as exc:
            outer_results.append(
                _invalid_outer_fold(
                    fold,
                    configuration,
                    f"outer refit/TEST continuation failed: {type(exc).__name__}: {exc}",
                )
            )

    valid_folds = tuple(fold for fold in outer_results if fold.valid)
    nmi_values = tuple(
        fold.outer_teacher_final_soft_nmi
        for fold in valid_folds
        if fold.outer_teacher_final_soft_nmi is not None
    )
    valid_count = len(valid_folds)
    valid_rate = valid_count / len(outer_results)
    latest_valid = bool(outer_results[-1].valid)
    return AdaptiveEvaluationResult(
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
) -> AdaptiveEvaluationResult:
    """Capture the complete dynamic source universe and run v4 on that snapshot.

    The empty feature request is intentional: the source, not the caller or a
    profile YAML list, determines the raw evaluation universe.  The database
    transaction is closed by the source before any clustering or HMM fitting
    starts, so subsequent schema changes cannot affect this evaluation.
    """

    if profile.profile_id != "xetra" or profile.profile_config_version != 4:
        raise ValueError("dynamic source evaluation requires the canonical Xetra v4 profile")
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
