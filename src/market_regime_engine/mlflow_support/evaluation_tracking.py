"""MLflow tracking for the canonical Xetra v4 evaluation."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import cast

from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan, plan_walk_forward
from market_regime_engine.evaluation_statistics.contracts import (
    GLOBAL_V4_EVALUATION_ID,
    GlobalV4Evidence,
    RunStatistics,
    RunType,
    Status,
)
from market_regime_engine.evaluation_statistics.writer import StatisticsWriter
from market_regime_engine.evaluations.global_regime_v4 import V4ConfigurationSelection
from market_regime_engine.evaluations.plots import render_global_v4_diagnostics
from market_regime_engine.feature_discovery.contracts import (
    AdaptiveEvaluationResult,
    FinalSelectedConfiguration,
    OuterFoldResult,
)
from market_regime_engine.features.ports import FeatureCatalogSnapshot, FeatureSnapshot
from market_regime_engine.mlflow_support.metric_catalog import METRIC_CATALOG_VERSION
from market_regime_engine.mlflow_support.metric_export import MetricExportLedger
from market_regime_engine.mlflow_support.model_metrics import outer_selection_metric_points
from market_regime_engine.mlflow_support.ports import TrackingPort
from market_regime_engine.mlflow_support.tracking import (
    FileMlflowTrackingPort,
    _project_candidate_logged_model,
)
from market_regime_engine.profiles.config import ModelProfile
from market_regime_engine.runtime.cpu import cpu_worker_count

PayloadEmitter = Callable[[str, Path], None]


class GlobalV4TrackingResult:
    """Tracked v4 parent and outer-fold children with immutable local mirrors."""

    def __init__(
        self,
        parent_run_id: str,
        outer_fold_run_ids: tuple[tuple[str, str], ...],
        statistics_root: str,
        global_evidence_hash: str,
        plot_manifest_path: str,
        logged_model_ids: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.parent_run_id = parent_run_id
        self.outer_fold_run_ids = outer_fold_run_ids
        self.statistics_root = statistics_root
        self.global_evidence_hash = global_evidence_hash
        self.plot_manifest_path = plot_manifest_path
        self.logged_model_ids = logged_model_ids


def _running_statistics(
    run_name: str,
    run_type: RunType,
    evidence: dict[str, object],
) -> RunStatistics:
    return RunStatistics(
        GLOBAL_V4_EVALUATION_ID,
        "pending",
        run_type,
        run_name,
        Status.RUNNING,
        datetime.now(UTC),
        evidence=evidence,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def track_statistics_run(
    port: TrackingPort,
    writer: StatisticsWriter,
    *,
    run_name: str,
    statistics: RunStatistics,
    parent_run_id: str | None = None,
    payload_emitter: PayloadEmitter | None = None,
) -> tuple[str, str]:
    """Create one MLflow run and finalize its immutable local statistics dossier."""

    if statistics.status is not Status.RUNNING:
        raise ValueError("statistics tracking requires an initial RUNNING dossier")
    run_id = port.start_run(run_name=run_name, parent_run_id=parent_run_id)
    started = replace(statistics, mlflow_run_id=run_id, parent_run_id=parent_run_id)
    directory: Path | None = None
    finalized_locally = False
    try:
        directory = writer.start(started)
        if payload_emitter is not None:
            payload_emitter(run_id, directory)
        finalized = replace(started, status=Status.FINISHED, ended_at=datetime.now(UTC))
        digest = writer.finalize(finalized)
        finalized_locally = True
        path = directory / "statistics.json"
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("finalized statistics hash mismatch")
        port.log_params(run_id, {"statistics_sha256": digest})
        port.log_artifact(run_id, str(path), "statistics")
        port.end_run(run_id)
        return run_id, digest
    except BaseException as exc:
        if directory is not None and not finalized_locally:
            failed_evidence = dict(started.evidence)
            failed_evidence["failure"] = {
                "code": type(exc).__name__,
                "reason": "evaluation tracking payload/finalization failed",
            }
            with suppress(BaseException):
                writer.finalize(
                    replace(
                        started,
                        status=Status.FAILED,
                        ended_at=datetime.now(UTC),
                        evidence=failed_evidence,
                    )
                )
        port.fail_run(run_id)
        raise


def _configuration_record(configuration: FinalSelectedConfiguration) -> dict[str, object]:
    return {
        "feature_order": list(configuration.feature_order),
        "candidate_id": configuration.candidate_id,
        "state_count": configuration.state_count,
        "model_family": configuration.model_family,
        "selected_prefix_length": configuration.selected_prefix_length,
        "feature_discovery_hash": configuration.feature_discovery_hash,
        "source_build_id": configuration.source_build_id,
        "catalog_hash": configuration.catalog_hash,
        "selection_definition_hash": configuration.selection_definition_hash,
        "selection_execution_hash": configuration.selection_execution_hash,
        "state_identity_scope": configuration.state_identity_scope,
    }


def _aggregate_record(aggregate: object) -> dict[str, object]:
    return {
        name: getattr(aggregate, name)
        for name in (
            "candidate_id",
            "state_count",
            "planned_fold_count",
            "valid_fold_count",
            "invalid_fold_count",
            "valid_fold_rate",
            "passes_valid_fold_rate_gate",
            "oos_predictive_loglik_mean",
            "oos_predictive_loglik_std",
            "oos_predictive_loglik_worst_fold",
            "oos_predictive_loglik_best_fold",
            "bic_mean",
            "aic_mean",
        )
    }


def _outer_record(fold: OuterFoldResult) -> dict[str, object]:
    return {
        "fold_id": f"outer_fold_{fold.fold_index:03d}",
        "fold_index": fold.fold_index,
        "train_start": fold.train_start.isoformat(),
        "train_end": fold.train_end.isoformat(),
        "test_start": fold.test_start.isoformat(),
        "test_end": fold.test_end.isoformat(),
        "final_configuration": _configuration_record(fold.final_configuration),
        "oos_predictive_loglik_per_observation": fold.oos_predictive_loglik_per_observation,
        "state_identity": fold.state_identity,
        "teacher_reference_hash": fold.teacher_reference_hash,
        "outer_teacher_final_soft_nmi": fold.outer_teacher_final_soft_nmi,
        "outer_shared_timestamp_count": fold.outer_shared_timestamp_count,
        "valid": fold.valid,
        "failure_reason": fold.failure_reason,
    }


def _selected_fold_evidence(
    fold: OuterFoldResult,
    selection: V4ConfigurationSelection,
) -> dict[str, object]:
    final_grid = selection.final_grid
    final_selection = final_grid.selection
    if final_selection is None:
        raise ValueError("tracked global v4 selection requires a final-grid champion")
    quality = selection.quality
    teacher = selection.teacher_evaluation
    reference = selection.teacher_reference
    return {
        "identity": {
            "evaluation_id": GLOBAL_V4_EVALUATION_ID,
            "outer_fold_id": f"outer_fold_{fold.fold_index:03d}",
            "feature_discovery_hash": selection.feature_discovery_hash,
        },
        "lineage": {
            "source_build_id": selection.source_build_id,
            "catalog_hash": selection.catalog_hash,
            "quality_result_hash": quality.result_hash,
            "distance_matrix_hash": selection.distance.matrix_hash,
            "cluster_solution_hash": selection.clusters.solution_hash,
            "teacher_reference_hash": reference.reference_hash,
            "inner_plan_hash": reference.inner_plan_hash,
            "final_grid_plan_hash": final_grid.candidate_grid.evaluation_plan_hash,
        },
        "input": {
            "catalog_feature_order": list(selection.distance.feature_order),
            "selected_feature_order": list(selection.final_candidate.feature_order),
            "outer_train_bounds": {
                "start": fold.train_start.isoformat(),
                "end": fold.train_end.isoformat(),
            },
        },
        "quality": {
            "train_source_observation_count": quality.train_source_observation_count,
            "eligible_features": list(quality.eligible_features),
            "features": [
                {
                    "feature_name": item.feature_name,
                    "canonical_ordinal": item.canonical_ordinal,
                    "finite_observation_count": item.finite_observation_count,
                    "coverage": item.coverage,
                    "population_variance": item.population_variance,
                    "eligible": item.eligible,
                    "rejection_reason": item.rejection_reason,
                }
                for item in quality.features
            ],
        },
        "distance": {
            "feature_order": list(selection.distance.feature_order),
            "matrix_hash": selection.distance.matrix_hash,
            "minimum_pairwise_observations": selection.distance.minimum_pairwise_observations,
            "distances": [list(row) for row in selection.distance.distances],
            "pairwise_support": [list(row) for row in selection.distance.pairwise_support],
            "spearman_correlations": [
                list(row) for row in selection.distance.spearman_correlations
            ],
        },
        "clustering": {
            "solution_hash": selection.clusters.solution_hash,
            "candidate_count": selection.clusters.candidate_count,
            "selected_m": selection.clusters.selected_count,
            "silhouette_curve": [list(item) for item in selection.clusters.silhouette_curve],
            "selected_silhouette": selection.clusters.selected_silhouette,
            "singleton_count": selection.clusters.singleton_count,
            "memberships": [
                {"cluster_id": cluster_id, "features": list(features)}
                for cluster_id, features in selection.clusters.memberships
            ],
        },
        "prototypes": {
            "temporary": selection.prototypes.temporary,
            "cluster_ids": list(selection.prototypes.cluster_ids),
            "features": list(selection.prototypes.prototypes),
            "mean_distances": [list(item) for item in selection.prototypes.mean_distances],
        },
        "teacher": {
            "candidate_id": reference.candidate_id,
            "state_count": reference.state_count,
            "reference_hash": reference.reference_hash,
            "prototype_features": list(reference.prototype_features),
            "valid_inner_fold_ids": list(reference.valid_inner_fold_ids),
            "inner_plan_hash": reference.inner_plan_hash,
            "selection_candidate_id": teacher.provisional_candidate_id,
            "selection_state_count": teacher.provisional_state_count,
            "selection_no_reason": teacher.no_selection_reason,
        },
        "feature_scores": {
            "scores": [
                {
                    "feature_name": item.feature_name,
                    "canonical_ordinal": item.canonical_ordinal,
                    "coverage": item.coverage,
                    "observation_count": item.observation_count,
                    "state_information_ratio": item.state_information_ratio,
                    "eta_squared": item.eta_squared,
                    "eligible": item.eligible,
                    "exclusion_reason": item.exclusion_reason,
                }
                for item in selection.feature_scores
            ],
            "ranked_winners": list(selection.winner_selection.ranked_features),
            "cluster_winners": [
                {
                    "cluster_id": item.cluster_id,
                    "winner_feature": item.winner_feature,
                    "member_features": list(item.member_features),
                }
                for item in selection.winner_selection.winners
            ],
        },
        "prefix_search": {
            "ranked_features": list(selection.prefix_search.ranked_features),
            "selected_l": selection.prefix_search.selected_prefix_length,
            "selected_candidate_id": selection.prefix_search.selected_candidate_id,
            "evaluations": [
                {
                    "prefix_length": item.prefix_length,
                    "feature_order": list(item.feature_order),
                    "candidate_id": item.candidate_id,
                    "shared_timestamp_count": item.shared_timestamp_count,
                    "shared_teacher_coverage": item.shared_teacher_coverage,
                    "soft_regime_nmi": item.soft_regime_nmi,
                    "valid": item.valid,
                    "invalid_reason": item.invalid_reason,
                }
                for item in selection.prefix_search.evaluations
            ],
        },
        "final_grid": {
            "feature_order": list(final_grid.candidate_grid.feature_order),
            "evaluation_plan_hash": final_grid.candidate_grid.evaluation_plan_hash,
            "candidate_aggregates": [
                _aggregate_record(item) for item in final_grid.candidate_grid.aggregates
            ],
            "champion_candidate_id": final_selection.champion_candidate_id,
            "champion_state_count": final_selection.champion_state_count,
            "ranked_candidate_ids": list(final_selection.ranked_candidate_ids),
        },
        "outer_folds": [_outer_record(fold)],
        "agreement": {
            "outer_teacher_final_soft_nmi": fold.outer_teacher_final_soft_nmi,
            "outer_shared_timestamp_count": fold.outer_shared_timestamp_count,
            "teacher_reference_hash": fold.teacher_reference_hash,
        },
        "validity": {"valid": fold.valid, "failure_reason": fold.failure_reason},
        "stability": {"adjacent_cluster_membership_jaccard": []},
    }


def _failed_fold_evidence(fold: OuterFoldResult) -> dict[str, object]:
    return {
        "identity": {
            "evaluation_id": GLOBAL_V4_EVALUATION_ID,
            "outer_fold_id": f"outer_fold_{fold.fold_index:03d}",
        },
        "outer_folds": [_outer_record(fold)],
        "validity": {"valid": False, "failure_reason": fold.failure_reason},
        "failure": {
            "code": "OuterFoldFailure",
            "reason": fold.failure_reason or "outer fold did not complete",
        },
    }


def build_global_v4_evidence(
    result: AdaptiveEvaluationResult,
    *,
    catalog: FeatureCatalogSnapshot,
    snapshot: FeatureSnapshot,
    profile: ModelProfile,
    selections: Mapping[int, V4ConfigurationSelection],
    repository_commit_sha: str,
) -> GlobalV4Evidence:
    """Build the complete model-binary-free evidence bundle for one v4 run."""

    if result.source_build_id != catalog.lineage.source_build_id:
        raise ValueError("global v4 evidence source build differs from catalog lineage")
    if result.catalog_hash != catalog.catalog_hash:
        raise ValueError("global v4 evidence catalog differs from result")
    if snapshot.lineage != catalog.lineage:
        raise ValueError("global v4 evidence snapshot lineage differs from catalog")
    if snapshot.feature_names != catalog.feature_names:
        raise ValueError("global v4 evidence snapshot columns differ from catalog")
    if not repository_commit_sha or repository_commit_sha.strip() != repository_commit_sha:
        raise ValueError("repository commit identity must be non-empty and trimmed")
    outer_plan = plan_walk_forward(
        tuple(row.timestamp for row in snapshot.rows), profile.walk_forward
    )
    fold_evidence = tuple(
        (
            _selected_fold_evidence(fold, selections[fold.fold_index])
            if fold.fold_index in selections
            else _failed_fold_evidence(fold)
        )
        for fold in result.outer_folds
    )
    selected = tuple(
        item for item in fold_evidence if "quality" in item and "feature_scores" in item
    )
    failures = tuple(
        {
            "fold_index": fold.fold_index,
            "reason": fold.failure_reason or "outer fold did not complete",
        }
        for fold in result.outer_folds
        if not fold.valid
    )
    return GlobalV4Evidence(
        source_build_id=result.source_build_id,
        source_data_hash=catalog.lineage.data_sha256,
        catalog_hash=catalog.catalog_hash,
        profile_hash=profile.profile_hash,
        repository_hash=sha256(repository_commit_sha.encode("utf-8")).hexdigest(),
        outer_plan_hash=outer_plan.plan_hash,
        evidence={
            "identity": {
                "evaluation_id": GLOBAL_V4_EVALUATION_ID,
                "policy_id": "xetra_global_regime_v4",
                "profile_id": profile.profile_id,
                "profile_config_version": profile.profile_config_version,
                "outer_fold_count": len(result.outer_folds),
            },
            "lineage": {
                "source_build_id": result.source_build_id,
                "source_dataset": catalog.lineage.source_dataset,
                "source_table": catalog.lineage.source_table,
                "source_data_sha256": catalog.lineage.data_sha256,
                "source_catalog_hash": catalog.catalog_hash,
                "materialized_feature_data_sha256": snapshot.materialized_feature_data_sha256,
                "repository_commit_sha": repository_commit_sha,
            },
            "input": {
                "feature_order": list(catalog.feature_names),
                "source_row_count": len(snapshot.rows),
                "source_min_timestamp": snapshot.rows[0].timestamp.isoformat(),
                "source_max_timestamp": snapshot.rows[-1].timestamp.isoformat(),
                "data_time_semantics": catalog.lineage.data_time_semantics,
            },
            "quality": {"folds": [item["quality"] for item in selected]},
            "distance": {"folds": [item["distance"] for item in selected]},
            "clustering": {"folds": [item["clustering"] for item in selected]},
            "prototypes": {"folds": [item["prototypes"] for item in selected]},
            "teacher": {"folds": [item["teacher"] for item in selected]},
            "feature_scores": {"folds": [item["feature_scores"] for item in selected]},
            "prefix_search": {"folds": [item["prefix_search"] for item in selected]},
            "final_grid": {"folds": [item["final_grid"] for item in selected]},
            "outer_folds": [_outer_record(fold) for fold in result.outer_folds],
            "agreement": {
                "folds": [
                    {
                        "fold_index": fold.fold_index,
                        "outer_teacher_final_soft_nmi": fold.outer_teacher_final_soft_nmi,
                        "outer_shared_timestamp_count": fold.outer_shared_timestamp_count,
                        "oos_predictive_loglik_per_observation": (
                            fold.oos_predictive_loglik_per_observation
                        ),
                    }
                    for fold in result.outer_folds
                ],
                "soft_nmi_mean": result.soft_nmi_mean,
                "soft_nmi_population_std": result.soft_nmi_population_std,
                "soft_nmi_worst": result.soft_nmi_worst,
            },
            "validity": {
                "valid_fold_count": result.valid_fold_count,
                "planned_fold_count": len(result.outer_folds),
                "valid_fold_rate": result.valid_fold_rate,
                "latest_complete_fold_valid": result.latest_complete_fold_valid,
                "production_eligible": result.production_eligible,
                "failure_reason": result.failure_reason,
            },
            "stability": {
                "selection_sink": "outer_fold_train_only",
                "fold_feature_discovery_hashes": [
                    fold.final_configuration.feature_discovery_hash for fold in result.outer_folds
                ],
            },
            **({"failure": {"outer_fold_failures": list(failures)}} if failures else {}),
        },
    )


def _validate_inputs(
    evidence: GlobalV4Evidence,
    result: AdaptiveEvaluationResult,
    selections: Mapping[int, V4ConfigurationSelection],
) -> None:
    if result.source_build_id != evidence.source_build_id:
        raise ValueError("global v4 tracking source build differs from canonical evidence")
    if result.catalog_hash != evidence.catalog_hash:
        raise ValueError("global v4 tracking catalog differs from canonical evidence")
    folds = {fold.fold_index: fold for fold in result.outer_folds}
    if set(selections) - set(folds):
        raise ValueError("global v4 tracking selections reference unknown outer folds")
    missing = {fold.fold_index for fold in result.outer_folds if fold.valid} - set(selections)
    if missing:
        raise ValueError(
            "global v4 tracking requires selection evidence for every valid outer fold"
        )
    for index, selection in selections.items():
        fold = folds[index]
        if (
            selection.source_build_id != result.source_build_id
            or selection.catalog_hash != result.catalog_hash
        ):
            raise ValueError("global v4 tracked selection lineage differs from result")
        if fold.valid and (
            selection.final_candidate.candidate_id != fold.final_configuration.candidate_id
            or selection.final_candidate.feature_order != fold.final_configuration.feature_order
        ):
            raise ValueError("global v4 tracked selection differs from frozen outer configuration")


def _track_global_v4_fold(
    port: TrackingPort,
    writer: StatisticsWriter,
    *,
    evidence: GlobalV4Evidence,
    fold: OuterFoldResult,
    selection: V4ConfigurationSelection | None,
    parent_run_id: str,
    directory: Path,
    metric_ledger: MetricExportLedger | None,
) -> tuple[tuple[str, str], tuple[tuple[str, str], ...]]:
    """Track one independent outer-fold dossier and its optional models."""

    fold_id = f"outer_fold_{fold.fold_index:03d}"
    fold_evidence = (
        _selected_fold_evidence(fold, selection)
        if selection is not None
        else _failed_fold_evidence(fold)
    )
    child_id, _ = track_statistics_run(
        port,
        writer,
        run_name=fold_id,
        parent_run_id=parent_run_id,
        statistics=_running_statistics(fold_id, RunType.CANDIDATE, fold_evidence),
    )
    port.log_params(
        child_id,
        {
            "metric_catalog_version": str(METRIC_CATALOG_VERSION),
            "evaluation_id": GLOBAL_V4_EVALUATION_ID,
            "outer_fold_id": fold_id,
            "outer_fold_index": str(fold.fold_index),
            "train_start": fold.train_start.isoformat(),
            "train_end": fold.train_end.isoformat(),
            "test_start": fold.test_start.isoformat(),
            "test_end": fold.test_end.isoformat(),
            "valid": str(fold.valid).lower(),
            "failure_reason": fold.failure_reason or "",
            "candidate_id": fold.final_configuration.candidate_id,
            "feature_discovery_hash": (
                selection.feature_discovery_hash if selection is not None else ""
            ),
        },
    )
    model_ids: list[tuple[str, str]] = []
    candidates = (
        tuple(getattr(selection.final_grid.candidate_grid, "evaluations", ()))
        if selection is not None
        else ()
    )
    if selection is not None:
        final_grid_plan = getattr(selection, "final_grid_plan", None)
        if candidates and final_grid_plan is None:
            raise ValueError("selected v4 evaluation is missing its final-grid plan")
        resolved_final_grid_plan = cast(WalkForwardPlan, final_grid_plan)
        model_root = directory / "logged_models" / fold_id
        dataset_snapshot_key = (
            f"dataset:{evidence.source_build_id}:{evidence.source_data_hash}:"
            f"{evidence.catalog_hash}"
        )
        evaluation_run_key = f"{GLOBAL_V4_EVALUATION_ID}:{evidence.evidence_hash}"
        for candidate in candidates:
            candidate_dir = model_root / candidate.candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            aggregate = next(
                item
                for item in selection.final_grid.candidate_grid.aggregates
                if item.candidate_id == candidate.candidate_id
            )
            _write_json(
                candidate_dir / "candidate_evidence.json",
                {
                    "candidate_id": candidate.candidate_id,
                    "feature_order": list(candidate.feature_order),
                    "source_build_id": candidate.source_build_id,
                    "evaluation_plan_hash": candidate.evaluation_plan_hash,
                    "aggregate": _aggregate_record(aggregate),
                    "selection_context": _selected_fold_evidence(fold, selection),
                },
            )
            model_id = _project_candidate_logged_model(
                port,
                evaluation=candidate,
                source_run_id=child_id,
                source_build_id=evidence.source_build_id,
                plan=resolved_final_grid_plan,
                candidate_dir=candidate_dir,
                dataset_snapshot_key=dataset_snapshot_key,
                evaluation_run_key=evaluation_run_key,
                scope="outer_fold_candidate",
                outer_fold_id=fold_id,
                model_name=f"{evaluation_run_key}:{fold_id}:{candidate.candidate_id}",
                extra_metric_points=outer_selection_metric_points(selection, fold),
                extra_tags={
                    "regime_engine.feature_discovery_hash": selection.feature_discovery_hash,
                    "regime_engine.selected": str(
                        candidate.candidate_id == fold.final_configuration.candidate_id
                    ).lower(),
                },
                metric_ledger=metric_ledger,
            )
            model_ids.append((f"{fold_id}:{candidate.candidate_id}", model_id))
    return (fold_id, child_id), tuple(model_ids)


def track_global_v4_evaluation(
    port: TrackingPort,
    writer: StatisticsWriter,
    *,
    evidence: GlobalV4Evidence,
    result: AdaptiveEvaluationResult,
    selections: Mapping[int, V4ConfigurationSelection],
    metric_ledger_root: str | Path | None = None,
) -> GlobalV4TrackingResult:
    """Track one v4 parent and one child dossier per outer fold."""

    _validate_inputs(evidence, result, selections)
    tracked_folds: list[tuple[str, str]] = []
    logged_model_ids: list[tuple[str, str]] = []
    plot_manifest_path: Path | None = None
    metric_ledger = MetricExportLedger(metric_ledger_root) if metric_ledger_root else None

    def emit_parent(parent_run_id: str, directory: Path) -> None:
        nonlocal plot_manifest_path
        evidence_path = directory / "global_v4_evidence.json"
        evidence_path.write_bytes(evidence.canonical_json())
        if sha256(evidence_path.read_bytes()).hexdigest() != evidence.evidence_hash:
            raise ValueError("global v4 canonical evidence hash mismatch")
        identity_value = evidence.evidence.get("identity", {})
        lineage_value = evidence.evidence.get("lineage", {})
        validity_value = evidence.evidence.get("validity", {})
        identity = identity_value if isinstance(identity_value, Mapping) else {}
        lineage = lineage_value if isinstance(lineage_value, Mapping) else {}
        validity = validity_value if isinstance(validity_value, Mapping) else {}
        port.log_params(
            parent_run_id,
            {
                "global_v4_evidence_sha256": evidence.evidence_hash,
                "metric_catalog_version": str(METRIC_CATALOG_VERSION),
                "evaluation_id": GLOBAL_V4_EVALUATION_ID,
                "profile_id": str(identity.get("profile_id", "xetra")),
                "profile_config_version": str(identity.get("profile_config_version", 4)),
                "source_build_id": str(lineage.get("source_build_id", evidence.source_build_id)),
                "source_data_sha256": str(
                    lineage.get("source_data_sha256", evidence.source_data_hash)
                ),
                "source_catalog_hash": str(
                    lineage.get("source_catalog_hash", evidence.catalog_hash)
                ),
                "profile_hash": evidence.profile_hash,
                "repository_hash": evidence.repository_hash,
                "outer_plan_hash": evidence.outer_plan_hash,
                "outer_fold_count": str(identity.get("outer_fold_count", len(result.outer_folds))),
                "valid_fold_count": str(validity.get("valid_fold_count", result.valid_fold_count)),
                "valid_fold_rate": str(validity.get("valid_fold_rate", result.valid_fold_rate)),
                "production_eligible": str(
                    validity.get("production_eligible", result.production_eligible)
                ).lower(),
            },
        )
        port.log_artifact(parent_run_id, str(evidence_path), "evidence")
        configured_workers = os.environ.get("REGIME_TRACKING_WORKERS")
        requested_workers = int(configured_workers) if configured_workers else 16
        tracking_workers = cpu_worker_count(
            requested_workers,
            task_count=len(result.outer_folds),
        )
        if isinstance(port, FileMlflowTrackingPort) and tracking_workers > 1:
            with ThreadPoolExecutor(max_workers=tracking_workers) as executor:
                fold_results = list(
                    executor.map(
                        lambda fold: _track_global_v4_fold(
                            port,
                            writer,
                            evidence=evidence,
                            fold=fold,
                            selection=selections.get(fold.fold_index),
                            parent_run_id=parent_run_id,
                            directory=directory,
                            metric_ledger=metric_ledger,
                        ),
                        result.outer_folds,
                    )
                )
        else:
            fold_results = [
                _track_global_v4_fold(
                    port,
                    writer,
                    evidence=evidence,
                    fold=fold,
                    selection=selections.get(fold.fold_index),
                    parent_run_id=parent_run_id,
                    directory=directory,
                    metric_ledger=metric_ledger,
                )
                for fold in result.outer_folds
            ]
        tracked_folds.extend(item[0] for item in fold_results)
        logged_model_ids.extend(
            model_id for _fold, model_ids in fold_results for model_id in model_ids
        )
        entries = render_global_v4_diagnostics(result, selections, directory)
        manifest_entries: list[dict[str, object]] = []
        for entry in entries:
            port.log_artifact(parent_run_id, entry.png_path, "plots")
            item = entry.as_dict()
            item["png_path"] = f"plots/{Path(entry.png_path).name}"
            manifest_entries.append(item)
        plot_manifest_path = directory / "global_v4_plot_manifest.json"
        _write_json(
            plot_manifest_path,
            {
                "evaluation_id": GLOBAL_V4_EVALUATION_ID,
                "global_evidence_hash": evidence.evidence_hash,
                "entries": manifest_entries,
            },
        )
        port.log_artifact(parent_run_id, str(plot_manifest_path), "plots")

    parent_run_id, _ = track_statistics_run(
        port,
        writer,
        run_name=GLOBAL_V4_EVALUATION_ID,
        statistics=_running_statistics(GLOBAL_V4_EVALUATION_ID, RunType.PARENT, evidence.evidence),
        payload_emitter=emit_parent,
    )
    if plot_manifest_path is None:
        raise RuntimeError("global v4 plot manifest was not created")
    return GlobalV4TrackingResult(
        parent_run_id,
        tuple(tracked_folds),
        str(writer.preflight()),
        evidence.evidence_hash,
        str(plot_manifest_path),
        tuple(logged_model_ids),
    )
