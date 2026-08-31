"""Fail-closed MLflow tracking with immutable local statistics mirrors."""

from __future__ import annotations

import json
from collections.abc import Callable
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
from market_regime_engine.evaluation_statistics.contracts import RunStatistics, RunType, Status
from market_regime_engine.evaluation_statistics.writer import StatisticsWriter
from market_regime_engine.evaluations.contracts import EvaluationId, FeatureSpec
from market_regime_engine.evaluations.delta1_univariate import Delta1UnivariateEvaluation
from market_regime_engine.evaluations.medoid_multivariate import MedoidMultivariateEvaluation
from market_regime_engine.evaluations.medoid_univariate import MedoidUnivariateEvaluation
from market_regime_engine.evaluations.univariate_grid import UnivariateFeatureGrid
from market_regime_engine.mlflow_support.plots import (
    EMCandidateConvergence,
    render_em_convergence,
    render_em_convergence_comparison,
    summarize_em_convergence,
)
from market_regime_engine.mlflow_support.ports import MetricPoint, TrackingPort
from market_regime_engine.training.candidate_grid import CandidateGridEvaluation

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
    evaluation_id: EvaluationId,
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
        candidate_root = root / "models" / evaluation.candidate_id
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
                "em_convergence_all_models": (
                    "model_metrics/comparisons/em_convergence_all_models.png"
                ),
            },
        },
    )
    port.log_artifact(feature_run_id, str(manifest_path), "model_metrics")


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
