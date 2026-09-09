"""Fail-closed MLflow tracking with immutable local statistics mirrors."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from market_regime_engine.evaluation.walk_forward import WalkForwardEvaluation
from market_regime_engine.evaluation_statistics.contracts import (
    GLOBAL_V4_EVALUATION_ID,
    GlobalV4Evidence,
    RunStatistics,
    RunType,
    Status,
)
from market_regime_engine.evaluation_statistics.writer import StatisticsWriter
from market_regime_engine.evaluations.contracts import EvaluationId, FeatureSpec
from market_regime_engine.evaluations.delta1_univariate import Delta1UnivariateEvaluation
from market_regime_engine.evaluations.global_regime_v4 import V4ConfigurationSelection
from market_regime_engine.evaluations.medoid_multivariate import MedoidMultivariateEvaluation
from market_regime_engine.evaluations.medoid_univariate import MedoidUnivariateEvaluation
from market_regime_engine.evaluations.plots import render_global_v4_diagnostics
from market_regime_engine.evaluations.univariate_grid import UnivariateFeatureGrid
from market_regime_engine.feature_discovery.contracts import (
    AdaptiveEvaluationResult,
    FinalSelectedConfiguration,
    OuterFoldResult,
)
from market_regime_engine.mlflow_support.plots import (
    EMCandidateConvergence,
    render_em_convergence,
    render_em_convergence_comparison,
    summarize_em_convergence,
)
from market_regime_engine.mlflow_support.ports import MetricPoint, TrackingPort
from market_regime_engine.training.candidate_grid import CandidateAggregate, CandidateGridEvaluation

EvaluationResult = (
    MedoidMultivariateEvaluation | MedoidUnivariateEvaluation | Delta1UnivariateEvaluation
)
PayloadEmitter = Callable[[str, Path], None]
_PERFORMANCE_METRICS: tuple[tuple[str, str], ...] = (
    ("train_loglik_per_obs", "TRAIN log likelihood per observation"),
    ("oos_predictive_loglik_per_obs", "OOS predictive log likelihood per observation"),
    ("aic_per_train_obs", "AIC per TRAIN observation"),
    ("bic_per_train_obs", "BIC per TRAIN observation"),
    ("multistart_success_rate", "Multistart success rate"),
)


@dataclass(frozen=True, slots=True)
class EvaluationTrackingResult:
    parent_run_id: str
    feature_run_ids: tuple[tuple[str, str], ...]
    candidate_run_ids: tuple[tuple[str, str], ...]
    statistics_root: str


@dataclass(frozen=True, slots=True)
class GlobalV4TrackingResult:
    """Tracked global-v4 parent and fold children with immutable local mirrors."""

    parent_run_id: str
    outer_fold_run_ids: tuple[tuple[str, str], ...]
    statistics_root: str
    global_evidence_hash: str
    plot_manifest_path: str


def track_statistics_run(
    port: TrackingPort,
    writer: StatisticsWriter,
    *,
    run_name: str,
    statistics: RunStatistics,
    parent_run_id: str | None = None,
    payload_emitter: PayloadEmitter | None = None,
) -> tuple[str, str]:
    """Create one MLflow run, emit payloads, then finalize its immutable local mirror."""

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
            failed = replace(
                started,
                status=Status.FAILED,
                ended_at=datetime.now(UTC),
                evidence=failed_evidence,
            )
            with suppress(BaseException):
                writer.finalize(failed)
        port.fail_run(run_id)
        raise


def _running_statistics(
    evaluation_id: EvaluationId | str,
    run_type: RunType,
    run_name: str,
    evidence: dict[str, object],
) -> RunStatistics:
    return RunStatistics(
        evaluation_id,
        "pending",
        run_type,
        run_name,
        Status.RUNNING,
        datetime.now(UTC),
        evidence=evidence,
    )


def _candidate_evidence(
    grid: CandidateGridEvaluation,
    candidate_id: str,
    *,
    include_optimization: bool = False,
) -> dict[str, object]:
    aggregate = next(item for item in grid.aggregates if item.candidate_id == candidate_id)
    evaluation = next(item for item in grid.evaluations if item.candidate_id == candidate_id)
    evidence: dict[str, object] = {
        "identity": {"candidate_id": candidate_id},
        "lineage": {
            "source_build_id": grid.source_build_id,
            "evaluation_plan_hash": grid.evaluation_plan_hash,
            "feature_selection_definition_hash": grid.feature_selection_definition_hash,
            "feature_selection_execution_hash": grid.feature_selection_execution_hash,
        },
        "input": {"feature_order": grid.feature_order},
        "model": {"state_count": evaluation.state_count},
        "folds": {
            "planned_count": len(evaluation.folds),
            "valid_count": len(evaluation.valid_folds),
        },
        "aggregate": asdict(aggregate),
    }
    if include_optimization:
        summary = summarize_em_convergence(evaluation)
        evidence["optimization"] = {
            "em_convergence": summary.as_json_dict(),
        }
    return evidence


def _metric_history(evaluation: WalkForwardEvaluation, metric_key: str) -> tuple[float | None, ...]:
    values: list[float | None] = []
    for fold in evaluation.folds:
        if not fold.valid:
            values.append(None)
            continue
        if metric_key == "train_loglik_per_obs":
            value = fold.train_log_likelihood
            count = fold.train_model_observation_count
            value = None if value is None else value / count
        elif metric_key == "oos_predictive_loglik_per_obs":
            value = fold.oos_predictive_log_likelihood_per_observation
        elif metric_key == "aic_per_train_obs":
            value = None if fold.aic is None else fold.aic / fold.train_model_observation_count
        elif metric_key == "bic_per_train_obs":
            value = None if fold.bic is None else fold.bic / fold.train_model_observation_count
        elif metric_key == "multistart_success_rate":
            value = fold.multistart_success_rate
        else:
            raise ValueError(f"unsupported model-metrics performance history: {metric_key}")
        if value is not None and not isfinite(value):
            raise ValueError(f"{metric_key} history values must be finite")
        values.append(value)
    return tuple(values)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _render_performance_history(
    evaluation: WalkForwardEvaluation,
    metric_key: str,
    label: str,
    output_path: Path,
) -> tuple[Path, tuple[float | None, ...]]:
    """Render a fold-indexed history without altering evaluation statistics."""

    values = _metric_history(evaluation, metric_key)
    x_values = np.arange(1, len(values) + 1, dtype=np.int64)
    y_values = np.asarray([np.nan if value is None else value for value in values])
    figure, axis = plt.subplots(figsize=(10.0, 5.5))
    axis.plot(x_values, y_values, marker="o", label=evaluation.candidate_id)
    axis.set_title(f"{label} — {evaluation.candidate_id}")
    axis.set_xlabel("Walk-forward fold")
    axis.set_ylabel(label)
    axis.grid(True, alpha=0.25)
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path, values


def _render_oos_comparison(grid: CandidateGridEvaluation, output_path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(11.0, 8.0))
    for evaluation in grid.evaluations:
        values = _metric_history(evaluation, "oos_predictive_loglik_per_obs")
        axis.plot(
            np.arange(1, len(values) + 1, dtype=np.int64),
            np.asarray([np.nan if value is None else value for value in values]),
            marker="o",
            label=evaluation.candidate_id,
        )
    axis.set_title(f"OOS predictive log likelihood comparison — {grid.feature_order[0]}")
    axis.set_xlabel("Walk-forward fold")
    axis.set_ylabel("OOS predictive log likelihood per observation")
    axis.grid(True, alpha=0.25)
    axis.legend(title="Canonical candidate order")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _render_train_refit_comparison(
    grid: CandidateGridEvaluation,
    output_path: Path,
) -> Path:
    """Compare normalized TRAIN likelihood over the dynamic candidate set."""
    figure, axis = plt.subplots(figsize=(13.0, 8.0))
    for evaluation in grid.evaluations:
        values = _metric_history(evaluation, "train_loglik_per_obs")
        axis.plot(
            np.arange(1, len(values) + 1, dtype=np.int64),
            np.asarray([np.nan if value is None else value for value in values]),
            marker="o",
            linewidth=1.5,
            markersize=3.5,
            label=evaluation.candidate_id,
        )
    axis.set_title(f"TRAIN log-likelihood per refit — {grid.feature_order[0]}")
    axis.set_xlabel("Walk-forward refit / fold")
    axis.set_ylabel("TRAIN log-likelihood per observation")
    axis.grid(True, alpha=0.25)
    axis.legend(title="Candidate model", fontsize="small")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _render_train_refit_comparison_all_features(
    grids: tuple[CandidateGridEvaluation, ...],
    output_path: Path,
) -> Path:
    """Compare TRAIN likelihood across every feature and dynamic candidate."""
    figure, axis = plt.subplots(figsize=(16.0, 10.0))
    labels_seen: set[str] = set()
    for grid in grids:
        for evaluation in grid.evaluations:
            values = _metric_history(evaluation, "train_loglik_per_obs")
            label = evaluation.candidate_id
            axis.plot(
                np.arange(1, len(values) + 1, dtype=np.int64),
                np.asarray([np.nan if value is None else value for value in values]),
                alpha=0.28,
                linewidth=1.0,
                label=label if label not in labels_seen else "_nolegend_",
            )
            labels_seen.add(label)
    axis.set_title("TRAIN log-likelihood per refit — all Delta1 features and models")
    axis.set_xlabel("Walk-forward refit / fold")
    axis.set_ylabel("TRAIN log-likelihood per observation")
    axis.grid(True, alpha=0.25)
    axis.legend(title="Candidate model (all feature lines shown)", fontsize="small")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _em_metric_points(summary: EMCandidateConvergence) -> tuple[MetricPoint, ...]:
    if not summary.available:
        return ()
    timestamp_ms = int(datetime.now(UTC).timestamp() * 1000)
    return tuple(
        MetricPoint(
            key="model_metrics.em_convergence.train_loglik_per_obs_median",
            value=value,
            step=iteration,
            timestamp_ms=timestamp_ms,
        )
        for iteration, value in zip(summary.iterations, summary.median, strict=True)
    )


def _model_metric_points(
    evaluation: WalkForwardEvaluation,
    aggregate: object | None,
) -> tuple[MetricPoint, ...]:
    """Return scalar summaries plus fold histories for MLflow's Model Metrics view."""
    timestamp_ms = int(datetime.now(UTC).timestamp() * 1000)
    points: list[MetricPoint] = []
    if aggregate is not None:
        for key, attribute in (
            ("oos_predictive_loglik_per_obs_mean", "oos_predictive_loglik_mean"),
            ("oos_predictive_loglik_per_obs_std", "oos_predictive_loglik_std"),
            (
                "oos_predictive_loglik_per_obs_worst_fold",
                "oos_predictive_loglik_worst_fold",
            ),
            ("bic_per_train_obs_mean", "bic_mean"),
            ("aic_per_train_obs_mean", "aic_mean"),
        ):
            value = getattr(aggregate, attribute)
            if value is not None and isfinite(value):
                points.append(MetricPoint(key=key, value=value, step=0, timestamp_ms=timestamp_ms))
    for step, value in enumerate(_metric_history(evaluation, "train_loglik_per_obs"), start=1):
        if value is not None:
            points.append(
                MetricPoint(
                    key="train_loglik_per_refit",
                    value=value,
                    step=step,
                    timestamp_ms=timestamp_ms,
                )
            )
    points.extend(_em_metric_points(summarize_em_convergence(evaluation)))
    return tuple(points)


def _emit_delta_model_metrics(
    port: TrackingPort,
    feature_run_id: str,
    directory: Path,
    grid: CandidateGridEvaluation,
) -> None:
    """Emit the delta1-only feature-run namespace before it can be finalized."""

    feature_name = grid.feature_order[0]
    root = directory / "model_metrics"
    candidates: list[dict[str, object]] = []
    for evaluation in grid.evaluations:
        aggregate = next(
            (
                item
                for item in getattr(grid, "aggregates", ())
                if item.candidate_id == evaluation.candidate_id
            ),
            None,
        )
        candidate_root = root / "models" / evaluation.candidate_id
        model_id = port.create_logged_model(
            name=f"delta1_{feature_name}_{evaluation.candidate_id}",
            source_run_id=feature_run_id,
            model_type="hmm",
            tags={
                "evaluation_id": EvaluationId.DELTA1_UNIVARIATE.value,
                "feature_name": feature_name,
                "candidate_id": evaluation.candidate_id,
            },
        )
        performance: list[dict[str, object]] = []
        for metric_key, label in _PERFORMANCE_METRICS:
            path, values = _render_performance_history(
                evaluation,
                metric_key,
                label,
                candidate_root / "performance" / f"{metric_key}.png",
            )
            destination = f"model_metrics/models/{evaluation.candidate_id}/performance"
            port.log_artifact(feature_run_id, str(path), destination)
            performance.append(
                {
                    "metric_key": metric_key,
                    "artifact_path": f"{destination}/{path.name}",
                    "source_hash": sha256(repr(values).encode()).hexdigest(),
                }
            )
        em_entry, summary = render_em_convergence(evaluation, feature_name, root / "models")
        em_destination = f"model_metrics/models/{evaluation.candidate_id}/optimization"
        port.log_artifact(feature_run_id, em_entry.png_path, em_destination)
        port.log_artifact(feature_run_id, em_entry.svg_path, em_destination)
        try:
            port.log_model_metric_points(model_id, _model_metric_points(evaluation, aggregate))
            port.log_model_artifacts(model_id, str(candidate_root))
            port.finalize_logged_model(model_id)
        except BaseException:
            with suppress(BaseException):
                port.finalize_logged_model(model_id, failed=True)
            raise
        candidates.append(
            {
                "candidate_id": evaluation.candidate_id,
                "availability": "available" if summary.available else "unavailable",
                "unavailable_reason": summary.unavailable_reason,
                "performance": performance,
                "em_convergence": {
                    "artifact_path": f"{em_destination}/{Path(em_entry.png_path).name}",
                    "source_hash": em_entry.source_artifact_hash,
                },
            }
        )
    comparison_root = root / "comparisons"
    oos_path = _render_oos_comparison(
        grid, comparison_root / "oos_predictive_loglik_per_obs_all_models.png"
    )
    port.log_artifact(
        feature_run_id,
        str(oos_path),
        "model_metrics/comparisons",
    )
    train_path = _render_train_refit_comparison(
        grid, comparison_root / "train_loglik_per_refit_all_models.png"
    )
    port.log_artifact(feature_run_id, str(train_path), "model_metrics/comparisons")
    em_entry, _ = render_em_convergence_comparison(grid.evaluations, feature_name, comparison_root)
    port.log_artifact(feature_run_id, em_entry.png_path, "model_metrics/comparisons")
    port.log_artifact(feature_run_id, em_entry.svg_path, "model_metrics/comparisons")
    manifest_path = root / "manifest.json"
    _write_json(
        manifest_path,
        {
            "feature_name": feature_name,
            "candidate_ids": [item.candidate_id for item in grid.evaluations],
            "candidates": candidates,
            "comparisons": {
                "oos_predictive_loglik_per_obs_all_models": (
                    "model_metrics/comparisons/oos_predictive_loglik_per_obs_all_models.png"
                ),
                "train_loglik_per_refit_all_models": (
                    "model_metrics/comparisons/train_loglik_per_refit_all_models.png"
                ),
                "em_convergence_all_models": (
                    "model_metrics/comparisons/em_convergence_all_models.png"
                ),
            },
        },
    )
    port.log_artifact(feature_run_id, str(manifest_path), "model_metrics")


def _emit_delta_parent_comparison(
    port: TrackingPort,
    parent_run_id: str,
    directory: Path,
    grids: tuple[CandidateGridEvaluation, ...],
) -> None:
    comparison_root = directory / "model_metrics" / "comparisons"
    plot_path = _render_train_refit_comparison_all_features(
        grids,
        comparison_root / "train_loglik_per_refit_all_features_all_models.png",
    )
    port.log_artifact(parent_run_id, str(plot_path), "model_metrics/comparisons")
    model_id = port.create_logged_model(
        name="delta1_univariate_comparison",
        source_run_id=parent_run_id,
        model_type="comparison",
        tags={
            "evaluation_id": EvaluationId.DELTA1_UNIVARIATE.value,
            "comparison_scope": "all_features_all_models",
        },
    )
    try:
        port.log_model_artifacts(model_id, str(comparison_root))
        port.finalize_logged_model(model_id)
    except BaseException:
        with suppress(BaseException):
            port.finalize_logged_model(model_id, failed=True)
        raise


def _delta_feature_payload(port: TrackingPort, grid: CandidateGridEvaluation) -> PayloadEmitter:
    def emit(run_id: str, directory: Path) -> None:
        _emit_delta_model_metrics(port, run_id, directory, grid)

    return emit


def _delta_candidate_payload(
    port: TrackingPort, candidate: WalkForwardEvaluation
) -> PayloadEmitter:
    def emit(run_id: str, _directory: Path) -> None:
        port.log_metric_points(run_id, _em_metric_points(summarize_em_convergence(candidate)))

    return emit


def _track_grid(
    port: TrackingPort,
    writer: StatisticsWriter,
    grid: CandidateGridEvaluation,
    parent_run_id: str,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            candidate.candidate_id,
            track_statistics_run(
                port,
                writer,
                run_name=candidate.candidate_id,
                parent_run_id=parent_run_id,
                statistics=_running_statistics(
                    EvaluationId.MEDOID_MULTIVARIATE,
                    RunType.CANDIDATE,
                    candidate.candidate_id,
                    _candidate_evidence(grid, candidate.candidate_id),
                ),
            )[0],
        )
        for candidate in grid.evaluations
    )


def _v4_configuration_record(configuration: FinalSelectedConfiguration) -> dict[str, object]:
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


def _v4_aggregate_record(aggregate: CandidateAggregate) -> dict[str, object]:
    return {
        "candidate_id": aggregate.candidate_id,
        "state_count": aggregate.state_count,
        "planned_fold_count": aggregate.planned_fold_count,
        "valid_fold_count": aggregate.valid_fold_count,
        "invalid_fold_count": aggregate.invalid_fold_count,
        "valid_fold_rate": aggregate.valid_fold_rate,
        "passes_valid_fold_rate_gate": aggregate.passes_valid_fold_rate_gate,
        "oos_predictive_loglik_mean": aggregate.oos_predictive_loglik_mean,
        "oos_predictive_loglik_std": aggregate.oos_predictive_loglik_std,
        "oos_predictive_loglik_worst_fold": aggregate.oos_predictive_loglik_worst_fold,
        "oos_predictive_loglik_best_fold": aggregate.oos_predictive_loglik_best_fold,
        "bic_mean": aggregate.bic_mean,
        "aic_mean": aggregate.aic_mean,
    }


def _v4_outer_record(fold: OuterFoldResult) -> dict[str, object]:
    """Persist fold outcome primitives, but no model binary or OOS probability rows."""

    return {
        "fold_id": f"outer_fold_{fold.fold_index:03d}",
        "fold_index": fold.fold_index,
        "train_start": fold.train_start.isoformat(),
        "train_end": fold.train_end.isoformat(),
        "test_start": fold.test_start.isoformat(),
        "test_end": fold.test_end.isoformat(),
        "final_configuration": _v4_configuration_record(fold.final_configuration),
        "oos_predictive_loglik_per_observation": fold.oos_predictive_loglik_per_observation,
        "state_identity": fold.state_identity,
        "teacher_reference_hash": fold.teacher_reference_hash,
        "outer_teacher_final_soft_nmi": fold.outer_teacher_final_soft_nmi,
        "outer_shared_timestamp_count": fold.outer_shared_timestamp_count,
        "valid": fold.valid,
        "failure_reason": fold.failure_reason,
    }


def _v4_selected_fold_evidence(
    fold: OuterFoldResult,
    selection: V4ConfigurationSelection,
) -> dict[str, object]:
    """Return complete fold-local v4 evidence in JSON primitives only.

    This is intentionally a compact, model-binary-free mirror of every
    TRAIN-only decision.  Prefix likelihood aggregates are retained in their
    exact local candidate records only; no cross-prefix likelihood ranking is
    constructed here or in the plots.
    """

    quality = selection.quality
    teacher = selection.teacher_evaluation
    teacher_reference = selection.teacher_reference
    final_grid = selection.final_grid
    final_selection = final_grid.selection
    if final_selection is None:
        raise ValueError("tracked global v4 selection requires a final-grid champion")
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
            "teacher_reference_hash": teacher_reference.reference_hash,
            "inner_plan_hash": teacher_reference.inner_plan_hash,
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
            "candidate_id": teacher_reference.candidate_id,
            "state_count": teacher_reference.state_count,
            "reference_hash": teacher_reference.reference_hash,
            "prototype_features": list(teacher_reference.prototype_features),
            "valid_inner_fold_ids": list(teacher_reference.valid_inner_fold_ids),
            "inner_plan_hash": teacher_reference.inner_plan_hash,
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
                _v4_aggregate_record(item) for item in final_grid.candidate_grid.aggregates
            ],
            "champion_candidate_id": final_selection.champion_candidate_id,
            "champion_state_count": final_selection.champion_state_count,
            "ranked_candidate_ids": list(final_selection.ranked_candidate_ids),
        },
        "outer_folds": [_v4_outer_record(fold)],
        "agreement": {
            "outer_teacher_final_soft_nmi": fold.outer_teacher_final_soft_nmi,
            "outer_shared_timestamp_count": fold.outer_shared_timestamp_count,
            "teacher_reference_hash": fold.teacher_reference_hash,
        },
        "validity": {
            "valid": fold.valid,
            "failure_reason": fold.failure_reason,
        },
        "stability": {"adjacent_cluster_membership_jaccard": []},
    }


def _v4_failed_fold_evidence(fold: OuterFoldResult) -> dict[str, object]:
    return {
        "identity": {
            "evaluation_id": GLOBAL_V4_EVALUATION_ID,
            "outer_fold_id": f"outer_fold_{fold.fold_index:03d}",
        },
        "outer_folds": [_v4_outer_record(fold)],
        "validity": {"valid": False, "failure_reason": fold.failure_reason},
        "failure": {
            "code": "OuterFoldFailure",
            "reason": fold.failure_reason or "outer fold did not complete",
        },
    }


def _validate_global_v4_tracking_inputs(
    evidence: GlobalV4Evidence,
    result: AdaptiveEvaluationResult,
    selections: Mapping[int, V4ConfigurationSelection],
) -> None:
    if result.source_build_id != evidence.source_build_id:
        raise ValueError("global v4 tracking source build differs from canonical evidence")
    if result.catalog_hash != evidence.catalog_hash:
        raise ValueError("global v4 tracking catalog differs from canonical evidence")
    fold_by_index = {fold.fold_index: fold for fold in result.outer_folds}
    unknown = set(selections) - set(fold_by_index)
    if unknown:
        raise ValueError("global v4 tracking selections reference unknown outer folds")
    missing_valid = {fold.fold_index for fold in result.outer_folds if fold.valid} - set(selections)
    if missing_valid:
        raise ValueError(
            "global v4 tracking requires selection evidence for every valid outer fold"
        )
    for fold_index, selection in selections.items():
        fold = fold_by_index[fold_index]
        if selection.source_build_id != result.source_build_id:
            raise ValueError("global v4 tracked selection source build differs from result")
        if selection.catalog_hash != result.catalog_hash:
            raise ValueError("global v4 tracked selection catalog differs from result")
        if fold.valid and (
            selection.final_candidate.candidate_id != fold.final_configuration.candidate_id
            or selection.final_candidate.feature_order != fold.final_configuration.feature_order
        ):
            raise ValueError("global v4 tracked selection differs from frozen outer configuration")


def track_global_v4_evaluation(
    port: TrackingPort,
    writer: StatisticsWriter,
    *,
    evidence: GlobalV4Evidence,
    result: AdaptiveEvaluationResult,
    selections: Mapping[int, V4ConfigurationSelection],
) -> GlobalV4TrackingResult:
    """Track global v4 as one parent and one fail-closed child per outer fold.

    ``evidence`` is the canonical statistical payload; its hash intentionally
    contains neither MLflow run IDs nor run timestamps.  The local evidence
    artifact is written and hash-checked before it is uploaded unchanged.
    """

    _validate_global_v4_tracking_inputs(evidence, result, selections)
    tracked_folds: list[tuple[str, str]] = []
    plot_manifest_path: Path | None = None

    def emit_parent(parent_run_id: str, directory: Path) -> None:
        nonlocal plot_manifest_path
        canonical_path = directory / "global_v4_evidence.json"
        canonical_path.write_bytes(evidence.canonical_json())
        if sha256(canonical_path.read_bytes()).hexdigest() != evidence.evidence_hash:
            raise ValueError("global v4 canonical evidence hash mismatch")
        port.log_params(parent_run_id, {"global_v4_evidence_sha256": evidence.evidence_hash})
        port.log_artifact(parent_run_id, str(canonical_path), "evidence")

        for fold in result.outer_folds:
            fold_id = f"outer_fold_{fold.fold_index:03d}"
            selection = selections.get(fold.fold_index)
            fold_evidence = (
                _v4_selected_fold_evidence(fold, selection)
                if selection is not None
                else _v4_failed_fold_evidence(fold)
            )
            # RunType has no outer-fold value; CANDIDATE is the existing generic
            # child type while the immutable identity names this exact outer fold.
            child_run_id, _ = track_statistics_run(
                port,
                writer,
                run_name=fold_id,
                parent_run_id=parent_run_id,
                statistics=_running_statistics(
                    GLOBAL_V4_EVALUATION_ID,
                    RunType.CANDIDATE,
                    fold_id,
                    fold_evidence,
                ),
            )
            tracked_folds.append((fold_id, child_run_id))

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
        statistics=_running_statistics(
            GLOBAL_V4_EVALUATION_ID,
            RunType.PARENT,
            GLOBAL_V4_EVALUATION_ID,
            evidence.evidence,
        ),
        payload_emitter=emit_parent,
    )
    if plot_manifest_path is None:
        raise RuntimeError("global v4 plot manifest was not created")
    return GlobalV4TrackingResult(
        parent_run_id=parent_run_id,
        outer_fold_run_ids=tuple(tracked_folds),
        statistics_root=str(writer.preflight()),
        global_evidence_hash=evidence.evidence_hash,
        plot_manifest_path=str(plot_manifest_path),
    )


def track_evaluation_result(
    port: TrackingPort,
    writer: StatisticsWriter,
    *,
    result: EvaluationResult,
) -> EvaluationTrackingResult:
    """Track one v3 evaluation result with a one-to-one immutable local mirror per run."""

    grids: tuple[CandidateGridEvaluation, ...]
    feature_grids: tuple[UnivariateFeatureGrid, ...]
    if isinstance(result, MedoidMultivariateEvaluation):
        evaluation_id = EvaluationId.MEDOID_MULTIVARIATE
        lineage, feature_spec, grids = result.lineage, result.feature_spec, (result.candidate_grid,)
        champion, reason = (
            result.medoid_multivariate_statistical_champion,
            result.no_champion_reason,
        )
        feature_grids = ()
    elif isinstance(result, MedoidUnivariateEvaluation):
        evaluation_id = EvaluationId.MEDOID_UNIVARIATE
        lineage, feature_spec = result.lineage, result.feature_spec
        grids = ()
        champion, reason = result.medoid_univariate_evaluation_champion, result.no_champion_reason
        feature_grids = result.feature_grids
    else:
        evaluation_id = EvaluationId.DELTA1_UNIVARIATE
        lineage = result.lineage
        feature_spec = FeatureSpec(
            evaluation_id, tuple(grid.feature_name for grid in result.feature_grids)
        )
        grids = ()
        champion, reason = result.delta1_univariate_evaluation_champion, result.no_champion_reason
        feature_grids = result.feature_grids
    parent_evidence: dict[str, object] = {
        "identity": {"evaluation_id": evaluation_id.value},
        "lineage": asdict(lineage),
        "input": {"feature_order": feature_spec.feature_order},
        "champion": {"candidate_id": champion, "no_champion_reason": reason},
    }
    parent_run_id, _ = track_statistics_run(
        port,
        writer,
        run_name=evaluation_id.value,
        statistics=_running_statistics(
            evaluation_id, RunType.PARENT, evaluation_id.value, parent_evidence
        ),
        payload_emitter=(
            (
                lambda run_id, directory: _emit_delta_parent_comparison(
                    port,
                    run_id,
                    directory,
                    tuple(feature_grid.candidate_grid for feature_grid in feature_grids),
                )
            )
            if evaluation_id is EvaluationId.DELTA1_UNIVARIATE
            else None
        ),
    )
    feature_run_ids: list[tuple[str, str]] = []
    candidate_run_ids: list[tuple[str, str]] = []
    for grid in grids:
        candidate_run_ids.extend(_track_grid(port, writer, grid, parent_run_id))
    for feature_grid in feature_grids:
        feature_run_id, _ = track_statistics_run(
            port,
            writer,
            run_name=feature_grid.feature_name,
            parent_run_id=parent_run_id,
            statistics=_running_statistics(
                evaluation_id,
                RunType.FEATURE,
                feature_grid.feature_name,
                {"input": {"feature_name": feature_grid.feature_name}},
            ),
            payload_emitter=(
                _delta_feature_payload(port, feature_grid.candidate_grid)
                if evaluation_id is EvaluationId.DELTA1_UNIVARIATE
                else None
            ),
        )
        feature_run_ids.append((feature_grid.feature_name, feature_run_id))
        for candidate in feature_grid.candidate_grid.evaluations:
            candidate_run_id, _ = track_statistics_run(
                port,
                writer,
                run_name=candidate.candidate_id,
                parent_run_id=feature_run_id,
                statistics=_running_statistics(
                    evaluation_id,
                    RunType.CANDIDATE,
                    candidate.candidate_id,
                    _candidate_evidence(
                        feature_grid.candidate_grid,
                        candidate.candidate_id,
                        include_optimization=evaluation_id is EvaluationId.DELTA1_UNIVARIATE,
                    ),
                ),
                payload_emitter=(
                    _delta_candidate_payload(port, candidate)
                    if evaluation_id is EvaluationId.DELTA1_UNIVARIATE
                    else None
                ),
            )
            candidate_run_ids.append((candidate.candidate_id, candidate_run_id))
    return EvaluationTrackingResult(
        parent_run_id, tuple(feature_run_ids), tuple(candidate_run_ids), str(writer.preflight())
    )
