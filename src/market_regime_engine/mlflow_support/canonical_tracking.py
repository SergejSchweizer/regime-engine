"""Minimal MLflow tracking port used by the canonical monthly pipeline."""

from __future__ import annotations

from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient

from market_regime_engine.mlflow_support.ports import MetricPoint


class CanonicalFileMlflowTrackingPort:
    """File/HTTP MLflow adapter with no dependency on retired evaluators."""

    _LOG_BATCH_SIZE = 1000

    def __init__(
        self,
        tracking_uri: str,
        *,
        experiment_name: str = "macro-regime-evaluation",
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
        self._logged_model_source_runs: dict[str, str] = {}

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
        self._client.log_batch(
            run_id,
            params=[Param(key=key, value=value) for key, value in sorted(params.items())],  # type: ignore[no-untyped-call]
        )

    def log_metric_points(self, run_id: str, points: tuple[MetricPoint, ...]) -> None:
        for offset in range(0, len(points), self._LOG_BATCH_SIZE):
            batch = points[offset : offset + self._LOG_BATCH_SIZE]
            self._client.log_batch(
                run_id,
                metrics=[
                    Metric(
                        key=point.key,
                        value=point.value,
                        timestamp=point.timestamp_ms,
                        step=point.step,
                    )
                    for point in batch
                ],
            )

    def create_logged_model(
        self,
        *,
        name: str,
        source_run_id: str,
        model_type: str,
        tags: dict[str, str],
    ) -> str:
        existing = tuple(
            item
            for item in self._client.search_logged_models([self._experiment_id])
            if item.name == name
        )
        if len(existing) > 1:
            raise ValueError(f"multiple LoggedModels exist for logical name {name!r}")
        if existing:
            item = existing[0]
            if item.model_type != model_type or any(
                item.tags.get(key) != value for key, value in tags.items()
            ):
                raise ValueError(f"existing LoggedModel {item.model_id} has conflicting identity")
            model_id = item.model_id
            if not isinstance(model_id, str):
                raise TypeError("logged model_id must be a string")
            self._logged_model_source_runs[model_id] = source_run_id
            return model_id
        model = self._client.create_logged_model(
            self._experiment_id,
            name=name,
            source_run_id=source_run_id,
            model_type=model_type,
            tags=tags,
        )
        model_id = model.model_id
        if not isinstance(model_id, str):
            raise TypeError("MLflow logged model_id must be a string")
        self._logged_model_source_runs[model_id] = source_run_id
        return model_id

    def log_model_metric_points(self, model_id: str, points: tuple[MetricPoint, ...]) -> None:
        run_id = self._logged_model_source_runs.get(model_id)
        if run_id is None:
            run_id = self._client.get_logged_model(model_id).source_run_id
        if not isinstance(run_id, str):
            raise ValueError(f"logged model {model_id} has no source run")
        for offset in range(0, len(points), self._LOG_BATCH_SIZE):
            batch = points[offset : offset + self._LOG_BATCH_SIZE]
            self._client.log_batch(
                run_id,
                metrics=[
                    Metric(
                        key=point.key,
                        value=point.value,
                        timestamp=point.timestamp_ms,
                        step=point.step,
                        model_id=model_id,
                    )
                    for point in batch
                ],
            )

    def get_model_metric_points(self, model_id: str) -> tuple[MetricPoint, ...]:
        model = self._client.get_logged_model(model_id)
        metrics = model.metrics or ()
        return tuple(
            MetricPoint(
                key=metric.key,
                value=float(metric.value),
                step=int(metric.step),
                timestamp_ms=int(metric.timestamp),
            )
            for metric in metrics
        )

    def log_model_artifacts(self, model_id: str, local_dir: str) -> None:
        self._client.log_model_artifacts(model_id, local_dir)

    def finalize_logged_model(self, model_id: str, *, failed: bool = False) -> None:
        self._client.finalize_logged_model(model_id, "FAILED" if failed else "READY")

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str) -> None:
        self._client.log_artifact(run_id, local_path, artifact_path)

    def end_run(self, run_id: str) -> None:
        self._client.set_terminated(run_id, status="FINISHED")

    def fail_run(self, run_id: str) -> None:
        self._client.set_terminated(run_id, status="FAILED")
