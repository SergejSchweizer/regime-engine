"""MLflow evaluation hierarchy and deterministic machine-readable evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import fmean, pstdev

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from mlflow.tracking import MlflowClient

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluation.walk_forward import (
    WalkForwardEvaluation,
    WalkForwardFoldResult,
)
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardFold, WalkForwardPlan
from market_regime_engine.mlflow_support.plots import (
    PlotManifestEntry,
    candidate_covariance_scale,
    fold_history_metric_keys,
    render_candidate_comparison,
    render_covariance_heatmap,
    render_fold_history,
    render_state_feature_influence,
    render_state_occupancy_table,
    render_state_transition_history,
    render_transition_heatmap,
)
from market_regime_engine.mlflow_support.ports import MetricPoint, TrackingPort


@dataclass(frozen=True, slots=True)
class EvaluationTrackingResult:
    parent_run_id: str
    candidate_run_ids: tuple[tuple[str, str], ...]
    parent_manifest_path: str


class FileMlflowTrackingPort:
    """Minimal concrete TrackingPort used by hermetic file-store integration tests."""

    def __init__(
        self,
        tracking_uri: str,
        *,
        experiment_name: str = "regime-engine-evaluation",
    ) -> None:
        self._client = MlflowClient(tracking_uri=tracking_uri)
        experiment = self._client.get_experiment_by_name(experiment_name)
        if experiment is not None and experiment.lifecycle_stage != "active":
            self._client.restore_experiment(experiment.experiment_id)
        self._experiment_id = (
            self._client.create_experiment(experiment_name)
            if experiment is None
            else experiment.experiment_id
        )

    def start_run(self, *, run_name: str, parent_run_id: str | None = None) -> str:
        tags = {"mlflow.runName": run_name}
        if parent_run_id is not None:
            tags["mlflow.parentRunId"] = parent_run_id
        run = self._client.create_run(self._experiment_id, tags=tags)
        run_id = run.info.run_id
        if not isinstance(run_id, str):
            raise TypeError("MLflow run_id must be a string")
        return run_id

    def log_params(self, run_id: str, params: dict[str, str]) -> None:
        for key, value in sorted(params.items()):
            self._client.log_param(run_id, key, value)

    def log_metric_points(self, run_id: str, points: tuple[MetricPoint, ...]) -> None:
        for point in points:
            self._client.log_metric(
                run_id,
                point.key,
                point.value,
                timestamp=point.timestamp_ms,
                step=point.step,
            )

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str) -> None:
        self._client.log_artifact(run_id, local_path, artifact_path)

    def end_run(self, run_id: str) -> None:
        self._client.set_terminated(run_id, status="FINISHED")


def _feature_order_hash(feature_order: tuple[str, ...]) -> str:
    payload = json.dumps(feature_order, separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


def _timestamp_ms(fold: WalkForwardFold) -> int:
    return int(fold.test_end.timestamp() * 1000)


def _scalar_fold_metrics(fold: WalkForwardFoldResult) -> dict[str, float | None]:
    hard = fold.train_hard_occupancy
    soft = fold.train_soft_occupancy
    train_count = fold.train_model_observation_count
    if train_count < 1:
        raise ValueError("normalized information criteria require positive TRAIN observations")
    return {
        "fold_train_loglik": (
            None if fold.train_log_likelihood is None else fold.train_log_likelihood / train_count
        ),
        "fold_oos_predictive_loglik": fold.oos_predictive_log_likelihood,
        "fold_oos_predictive_loglik_per_obs": (fold.oos_predictive_log_likelihood_per_observation),
        "fold_aic": fold.aic,
        "fold_bic": fold.bic,
        "fold_aic_per_train_obs": None if fold.aic is None else fold.aic / train_count,
        "fold_bic_per_train_obs": None if fold.bic is None else fold.bic / train_count,
        "fold_multistart_success_rate": fold.multistart_success_rate,
        "fold_min_train_hard_occupancy": None if hard is None else min(hard),
        "fold_min_train_soft_occupancy": None if soft is None else min(soft),
        "fold_max_state_signature_drift": fold.max_state_signature_drift,
        "fold_mean_state_duration": fold.mean_state_duration,
        "fold_switches_per_year": fold.switches_per_year,
        "fold_oos_entropy_mean": fold.oos_entropy_mean,
        "fold_oos_confidence_mean": fold.oos_confidence_mean,
    }


def _candidate_metric_points(
    evaluation: WalkForwardEvaluation,
    plan: WalkForwardPlan,
) -> tuple[MetricPoint, ...]:
    points: list[MetricPoint] = []
    for fold, planned in zip(evaluation.folds, plan.folds, strict=True):
        if not fold.valid:
            continue
        timestamp_ms = _timestamp_ms(planned)
        for key, value in _scalar_fold_metrics(fold).items():
            if value is not None:
                points.append(
                    MetricPoint(
                        key=key,
                        value=float(value),
                        step=fold.fold_index,
                        timestamp_ms=timestamp_ms,
                    )
                )
        state_vectors = {
            "fold_train_hard_occupancy": fold.train_hard_occupancy,
            "fold_train_soft_occupancy": fold.train_soft_occupancy,
            "fold_oos_hard_occupancy": fold.oos_hard_occupancy,
            "fold_oos_soft_occupancy": fold.oos_soft_occupancy,
        }
        for prefix, vector in state_vectors.items():
            if vector is None:
                continue
            for state_index, value in enumerate(vector):
                points.append(
                    MetricPoint(
                        key=f"{prefix}_state_{state_index}",
                        value=float(value),
                        step=fold.fold_index,
                        timestamp_ms=timestamp_ms,
                    )
                )
        if fold.model_artifact is not None and fold.alignment is not None:
            matrix = np.asarray(fold.model_artifact.transition_matrix, dtype=np.float64)
            mapping = fold.alignment.persistent_to_fitted
            aligned = matrix[np.ix_(mapping, mapping)]
            for state_index in range(evaluation.state_count):
                points.append(
                    MetricPoint(
                        key=f"fold_self_transition_state_{state_index}",
                        value=float(aligned[state_index, state_index]),
                        step=fold.fold_index,
                        timestamp_ms=timestamp_ms,
                    )
                )
    return tuple(points)


def _aggregate_metric_points(evaluation: WalkForwardEvaluation) -> tuple[MetricPoint, ...]:
    valid = evaluation.valid_folds
    timestamp_ms = int(evaluation.evaluation_cutoff.timestamp() * 1000)
    values = [
        fold.oos_predictive_log_likelihood_per_observation
        for fold in valid
        if fold.oos_predictive_log_likelihood_per_observation is not None
    ]
    metrics: dict[str, float] = {"candidate_valid_fold_rate": evaluation.valid_fold_rate}
    if values:
        metrics.update(
            {
                "candidate_oos_predictive_loglik_mean": fmean(values),
                "candidate_oos_predictive_loglik_std": pstdev(values),
                "candidate_oos_predictive_loglik_worst_fold": min(values),
                "candidate_oos_predictive_loglik_best_fold": max(values),
            }
        )
    bics = [fold.bic for fold in valid if fold.bic is not None]
    aics = [fold.aic for fold in valid if fold.aic is not None]
    if bics:
        metrics["candidate_bic_mean"] = fmean(bics)
    if aics:
        metrics["candidate_aic_mean"] = fmean(aics)
    return tuple(
        MetricPoint(key=key, value=value, step=0, timestamp_ms=timestamp_ms)
        for key, value in sorted(metrics.items())
    )


def _timeline_rows(
    evaluation: WalkForwardEvaluation,
    plan: WalkForwardPlan,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result, fold in zip(evaluation.folds, plan.folds, strict=True):
        rows.append(
            {
                "fold_id": fold.fold_id,
                "fold_index": fold.fold_index,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "valid": result.valid,
                "failure_reason": result.failure_reason,
                "train_source_observations": result.train_source_observation_count,
                "test_source_observations": result.test_source_observation_count,
                "train_model_observations": result.train_model_observation_count,
                "test_model_observations": result.test_model_observation_count,
                "skipped_train_incomplete": result.skipped_train_incomplete_count,
                "skipped_test_incomplete": result.skipped_test_incomplete_count,
            }
        )
    return rows


def _metric_rows(
    evaluation: WalkForwardEvaluation,
    plan: WalkForwardPlan,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result, fold in zip(evaluation.folds, plan.folds, strict=True):
        row: dict[str, object] = {
            "fold_id": fold.fold_id,
            "fold_index": fold.fold_index,
            "test_end": fold.test_end,
            "valid": result.valid,
            "failure_reason": result.failure_reason,
        }
        row.update(_scalar_fold_metrics(result))
        row["train_hard_occupancy"] = result.train_hard_occupancy
        row["train_soft_occupancy"] = result.train_soft_occupancy
        row["oos_hard_occupancy"] = result.oos_hard_occupancy
        row["oos_soft_occupancy"] = result.oos_soft_occupancy
        rows.append(row)
    return rows


def _write_parquet(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd", version="2.6")


def _aligned_parameter_payload(
    evaluation: WalkForwardEvaluation,
    fold: WalkForwardFoldResult,
) -> dict[str, object]:
    if fold.model_artifact is None or fold.alignment is None:
        raise ValueError("valid fold is missing aligned model evidence")
    mapping = fold.alignment.persistent_to_fitted
    artifact = fold.model_artifact
    transition = np.asarray(artifact.transition_matrix, dtype=np.float64)
    aligned_transition = transition[np.ix_(mapping, mapping)]
    aligned_covariances = [artifact.full_covariances[index] for index in mapping]
    aligned_means = [artifact.means[index] for index in mapping]
    return {
        "candidate_id": evaluation.candidate_id,
        "fold_id": fold.fold_id,
        "persistent_state_ids": fold.alignment.persistent_state_ids,
        "persistent_to_fitted": mapping,
        "feature_order": evaluation.feature_order,
        "transition_matrix": aligned_transition.tolist(),
        "means": aligned_means,
        "full_covariances": aligned_covariances,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n"
    path.write_text(rendered, encoding="utf-8")


def _validate_inputs(
    source_lineage: SourceLineage,
    plan: WalkForwardPlan,
    evaluations: tuple[WalkForwardEvaluation, ...],
) -> tuple[WalkForwardEvaluation, ...]:
    if not evaluations:
        raise ValueError("evaluation tracking requires at least one candidate")
    ordered = tuple(sorted(evaluations, key=lambda item: (item.state_count, item.candidate_id)))
    first = ordered[0]
    candidate_ids = tuple(item.candidate_id for item in ordered)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")
    for evaluation in ordered:
        if evaluation.profile_id != first.profile_id:
            raise ValueError("all candidates must share profile_id")
        if evaluation.profile_config_version != first.profile_config_version:
            raise ValueError("all candidates must share profile_config_version")
        if evaluation.source_build_id != source_lineage.source_build_id:
            raise ValueError("evaluation source build differs from supplied lineage")
        if evaluation.evaluation_plan_hash != plan.plan_hash:
            raise ValueError("evaluation plan hash differs from supplied plan")
        if evaluation.feature_order != first.feature_order:
            raise ValueError("all candidates must share identical frozen feature order")
        if (
            evaluation.feature_selection_definition_hash != first.feature_selection_definition_hash
            or evaluation.feature_selection_execution_hash != first.feature_selection_execution_hash
        ):
            raise ValueError("all candidates must share identical feature-selection hashes")
    return ordered


def track_walk_forward_evaluations(
    port: TrackingPort,
    *,
    source_lineage: SourceLineage,
    plan: WalkForwardPlan,
    evaluations: tuple[WalkForwardEvaluation, ...],
    statistical_selection_result: str,
    artifact_root: str | Path,
) -> EvaluationTrackingResult:
    """Persist parent/candidate/fold MLflow evidence without changing statistical decisions."""

    invalid_selection_result = (
        not statistical_selection_result
        or statistical_selection_result.strip() != statistical_selection_result
    )
    if invalid_selection_result:
        raise ValueError("statistical_selection_result must be a non-empty trimmed string")
    ordered = _validate_inputs(source_lineage, plan, evaluations)
    first = ordered[0]
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    parent_run_name = f"evaluation-{first.profile_id}-{source_lineage.source_build_id}"
    parent_run_id = port.start_run(run_name=parent_run_name)
    port.log_params(
        parent_run_id,
        {
            "profile_id": first.profile_id,
            "profile_config_version": str(first.profile_config_version),
            "source_build_id": source_lineage.source_build_id,
            "source_data_sha256": source_lineage.data_sha256,
            "source_schema_version": str(source_lineage.schema_version),
            "source_feature_version": str(source_lineage.feature_version),
            "source_synced_at_utc": source_lineage.synced_at_utc.isoformat(),
            "data_time_semantics": source_lineage.data_time_semantics,
            "evaluation_plan_hash": plan.plan_hash,
            "feature_selection_definition_hash": first.feature_selection_definition_hash,
            "feature_selection_execution_hash": first.feature_selection_execution_hash,
            "candidate_count": str(len(ordered)),
            "statistical_selection_result": statistical_selection_result,
        },
    )

    candidate_run_ids: list[tuple[str, str]] = []
    parent_entries: list[PlotManifestEntry] = []
    for evaluation in ordered:
        candidate_run_id = port.start_run(
            run_name=evaluation.candidate_id,
            parent_run_id=parent_run_id,
        )
        candidate_run_ids.append((evaluation.candidate_id, candidate_run_id))
        port.log_params(
            candidate_run_id,
            {
                "model_family": "gaussian_hmm",
                "candidate_id": evaluation.candidate_id,
                "state_count": str(evaluation.state_count),
                "covariance_type": "full",
                "feature_order": json.dumps(
                    evaluation.feature_order,
                    separators=(",", ":"),
                ),
                "feature_order_sha256": _feature_order_hash(evaluation.feature_order),
                "feature_selection_definition_hash": (evaluation.feature_selection_definition_hash),
                "feature_selection_execution_hash": (evaluation.feature_selection_execution_hash),
                "multistart_seeds": "11,23,37,53,71,89,107,131",
                "minimum_valid_starts": "6",
                "minimum_multistart_success_rate": "0.75",
            },
        )
        port.log_metric_points(candidate_run_id, _candidate_metric_points(evaluation, plan))
        port.log_metric_points(candidate_run_id, _aggregate_metric_points(evaluation))

        candidate_dir = root / evaluation.candidate_id
        timeline_path = candidate_dir / "fold_timeline.parquet"
        metrics_path = candidate_dir / "fold_metrics.parquet"
        _write_parquet(_timeline_rows(evaluation, plan), timeline_path)
        _write_parquet(_metric_rows(evaluation, plan), metrics_path)
        port.log_artifact(candidate_run_id, str(timeline_path), "evaluation")
        port.log_artifact(candidate_run_id, str(metrics_path), "evaluation")

        entries = [
            render_fold_history(evaluation, plan, metric_key, root)
            for metric_key in fold_history_metric_keys()
        ]
        if evaluation.valid_folds:
            entries.extend(
                (
                    render_state_occupancy_table(evaluation, plan, root),
                    render_state_transition_history(evaluation, plan, root),
                    render_state_feature_influence(evaluation, root),
                )
            )
        shared_scale = candidate_covariance_scale(evaluation) if evaluation.valid_folds else None
        for result, planned in zip(evaluation.folds, plan.folds, strict=True):
            fold_run_id = port.start_run(
                run_name=result.fold_id,
                parent_run_id=candidate_run_id,
            )
            port.log_params(
                fold_run_id,
                {
                    "fold_id": result.fold_id,
                    "fold_index": str(result.fold_index),
                    "train_start": planned.train_start.isoformat(),
                    "train_end": planned.train_end.isoformat(),
                    "test_start": planned.test_start.isoformat(),
                    "test_end": planned.test_end.isoformat(),
                    "valid": str(result.valid).lower(),
                    "failure_reason": result.failure_reason or "",
                },
            )
            if not result.valid:
                port.end_run(fold_run_id)
                continue
            parameter_path = candidate_dir / result.fold_id / "aligned_parameters.json"
            _write_json(parameter_path, _aligned_parameter_payload(evaluation, result))
            port.log_artifact(fold_run_id, str(parameter_path), "model_evidence")
            transition_entry = render_transition_heatmap(evaluation, result, root)
            entries.append(transition_entry)
            port.log_artifact(fold_run_id, transition_entry.png_path, "plots")
            if shared_scale is None:
                raise ValueError("valid fold requires a candidate covariance scale")
            for state_index in range(evaluation.state_count):
                covariance_entry = render_covariance_heatmap(
                    evaluation,
                    result,
                    state_index,
                    shared_scale,
                    root,
                )
                entries.append(covariance_entry)
                port.log_artifact(fold_run_id, covariance_entry.png_path, "plots")
            port.end_run(fold_run_id)

        for entry in entries:
            if entry.fold_id is None:
                port.log_artifact(candidate_run_id, entry.png_path, "plots")
        manifest_path = candidate_dir / "plot_manifest.json"
        manifest_payload = [
            entry.as_json_dict()
            for entry in sorted(
                entries,
                key=lambda item: (item.plot_type, item.fold_id or "", item.png_path),
            )
        ]
        _write_json(manifest_path, manifest_payload)
        port.log_artifact(candidate_run_id, str(manifest_path), "evaluation")
        port.end_run(candidate_run_id)

    comparison = render_candidate_comparison(ordered, plan, root)
    parent_entries.append(comparison)
    port.log_artifact(parent_run_id, comparison.png_path, "plots")
    parent_manifest_path = root / "parent" / "plot_manifest.json"
    _write_json(
        parent_manifest_path,
        [entry.as_json_dict() for entry in parent_entries],
    )
    port.log_artifact(parent_run_id, str(parent_manifest_path), "evaluation")
    port.end_run(parent_run_id)
    return EvaluationTrackingResult(
        parent_run_id=parent_run_id,
        candidate_run_ids=tuple(candidate_run_ids),
        parent_manifest_path=str(parent_manifest_path),
    )
