"""MLflow Model Metrics for fold-local PCA and matched OOS comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from statistics import fmean

from market_regime_engine.evaluation.walk_forward import (
    WalkForwardEvaluation,
    WalkForwardFoldResult,
)
from market_regime_engine.mlflow_support.metric_catalog import (
    require_metric_definition,
    validate_metric_points,
)
from market_regime_engine.mlflow_support.ports import MetricPoint
from market_regime_engine.preprocessing.two_stage import PCATwoStageScalerArtifact


def _timestamp_ms(value: datetime) -> int:
    timestamp = int(value.timestamp() * 1000)
    if timestamp < 0:
        raise ValueError("MLflow metric timestamps must be non-negative")
    return timestamp


def _append(
    points: list[MetricPoint],
    key: str,
    value: float | int,
    *,
    step: int,
    timestamp_ms: int,
) -> None:
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"PCA metric {key} must be finite")
    require_metric_definition(key)
    points.append(MetricPoint(key=key, value=numeric, step=step, timestamp_ms=timestamp_ms))


def _valid_pca_fold(fold: WalkForwardFoldResult) -> PCATwoStageScalerArtifact:
    if not fold.valid or fold.pca_scaler_artifact is None:
        raise ValueError("PCA metrics require a valid fold-local PCA artifact")
    return fold.pca_scaler_artifact


@dataclass(frozen=True, slots=True)
class PCAMetricComparison:
    """Matched raw-only and raw-plus-PCA evaluations on one frozen source plan."""

    raw_only: WalkForwardEvaluation
    raw_plus_pca: WalkForwardEvaluation

    def __post_init__(self) -> None:
        if self.raw_only.evaluation_plan_hash != self.raw_plus_pca.evaluation_plan_hash:
            raise ValueError("PCA comparison requires one identical evaluation plan")
        if self.raw_only.source_build_id != self.raw_plus_pca.source_build_id:
            raise ValueError("PCA comparison requires one identical source build")
        if self.raw_only.state_count != self.raw_plus_pca.state_count:
            raise ValueError("PCA comparison requires the same state count")
        if self.raw_only.candidate_id != self.raw_plus_pca.candidate_id:
            raise ValueError("PCA comparison requires the same model candidate")
        if len(self.raw_only.folds) != len(self.raw_plus_pca.folds):
            raise ValueError("PCA comparison requires equal fold counts")
        for raw_fold, pca_fold in zip(
            self.raw_only.folds,
            self.raw_plus_pca.folds,
            strict=True,
        ):
            if raw_fold.fold_id != pca_fold.fold_id or raw_fold.valid != pca_fold.valid:
                raise ValueError("PCA comparison requires matching fold identities and validity")
            if not pca_fold.valid:
                continue
            pca_scaler = pca_fold.pca_scaler_artifact
            if pca_scaler is None:
                raise ValueError("PCA comparison requires a PCA artifact for every valid fold")
            if pca_scaler.raw_feature_order != self.raw_only.feature_order:
                raise ValueError("PCA raw feature order differs from the raw-only evaluation")
            if raw_fold.oos_timestamps != pca_fold.oos_timestamps:
                raise ValueError("PCA comparison requires identical retained TEST timestamps")
            if (
                raw_fold.oos_predictive_log_likelihood_per_observation is None
                or pca_fold.oos_predictive_log_likelihood_per_observation is None
            ):
                raise ValueError("PCA comparison requires OOS scores for every valid fold")
        if not self.matched_fold_indices:
            raise ValueError("PCA comparison requires at least one valid matched fold")

    @property
    def matched_fold_indices(self) -> tuple[int, ...]:
        return tuple(raw.fold_index for raw in self.raw_only.folds if raw.valid)


def pca_metric_points(
    evaluation: WalkForwardEvaluation,
    *,
    fold_timestamps: tuple[datetime, ...] | None = None,
) -> tuple[MetricPoint, ...]:
    """Project fold-local PCA loadings, explained variance and fit counts."""

    if fold_timestamps is not None and len(fold_timestamps) != len(evaluation.folds):
        raise ValueError("fold_timestamps must match the candidate fold count")
    points: list[MetricPoint] = []
    default_timestamp = _timestamp_ms(evaluation.evaluation_cutoff)
    for index, fold in enumerate(evaluation.folds):
        if not fold.valid:
            continue
        pca_scaler = _valid_pca_fold(fold)
        timestamp_ms = (
            _timestamp_ms(fold_timestamps[index])
            if fold_timestamps is not None
            else default_timestamp
        )
        pca_fit = pca_scaler.pca_fit
        pca_artifact = pca_fit.artifact
        step = fold.fold_index
        _append(
            points,
            "pca_retained_component_count",
            pca_artifact.retained_component_count,
            step=step,
            timestamp_ms=timestamp_ms,
        )
        _append(
            points,
            "pca_total_component_count",
            pca_artifact.feature_dimension,
            step=step,
            timestamp_ms=timestamp_ms,
        )
        _append(
            points,
            "pca_cumulative_explained_variance",
            pca_artifact.cumulative_explained_variance,
            step=step,
            timestamp_ms=timestamp_ms,
        )
        _append(
            points,
            "pca_fit_selected_row_count",
            pca_fit.selected_row_count,
            step=step,
            timestamp_ms=timestamp_ms,
        )
        _append(
            points,
            "pca_skipped_incomplete_row_count",
            pca_fit.skipped_incomplete_row_count,
            step=step,
            timestamp_ms=timestamp_ms,
        )
        for component_index, explained in enumerate(
            pca_artifact.explained_variance_ratio,
            start=1,
        ):
            _append(
                points,
                f"pca_explained_variance_ratio_component_{component_index:03d}",
                explained,
                step=step,
                timestamp_ms=timestamp_ms,
            )
        for component_index, component in enumerate(pca_artifact.components, start=1):
            for feature_index, loading in enumerate(component, start=1):
                _append(
                    points,
                    f"pca_loading_component_{component_index:03d}_feature_{feature_index:03d}",
                    loading,
                    step=step,
                    timestamp_ms=timestamp_ms,
                )
    result = tuple(points)
    validate_metric_points(result)
    return result


def pca_comparison_metric_points(comparison: PCAMetricComparison) -> tuple[MetricPoint, ...]:
    """Project matched raw-only versus raw-plus-PCA OOS score histories."""

    points: list[MetricPoint] = []
    default_timestamp = _timestamp_ms(comparison.raw_plus_pca.evaluation_cutoff)
    deltas: list[float] = []
    for raw_fold, pca_fold in zip(
        comparison.raw_only.folds,
        comparison.raw_plus_pca.folds,
        strict=True,
    ):
        if not raw_fold.valid:
            continue
        raw_score = raw_fold.oos_predictive_log_likelihood_per_observation
        pca_score = pca_fold.oos_predictive_log_likelihood_per_observation
        if raw_score is None or pca_score is None:
            raise ValueError("PCA comparison requires finite matched OOS scores")
        delta = pca_score - raw_score
        deltas.append(delta)
        timestamp_ms = (
            _timestamp_ms(pca_fold.oos_timestamps[-1])
            if pca_fold.oos_timestamps
            else default_timestamp
        )
        for key, value in (
            ("pca_comparison_raw_oos_predictive_loglik_per_obs", raw_score),
            ("pca_comparison_augmented_oos_predictive_loglik_per_obs", pca_score),
            ("pca_comparison_delta_oos_predictive_loglik_per_obs", delta),
        ):
            _append(points, key, value, step=raw_fold.fold_index, timestamp_ms=timestamp_ms)
    _append(
        points,
        "pca_comparison_delta_mean",
        fmean(deltas),
        step=0,
        timestamp_ms=default_timestamp,
    )
    _append(
        points,
        "pca_comparison_win_count",
        sum(delta > 0.0 for delta in deltas),
        step=0,
        timestamp_ms=default_timestamp,
    )
    result = tuple(points)
    validate_metric_points(result)
    return result


__all__ = [
    "PCAMetricComparison",
    "pca_comparison_metric_points",
    "pca_metric_points",
]
