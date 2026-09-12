"""Fit-quality and information-criterion Model Metrics for v4 candidates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from math import isfinite, log
from statistics import fmean, median, pstdev

from market_regime_engine.evaluation.diagnostics import information_criteria
from market_regime_engine.evaluation.walk_forward import WalkForwardEvaluation
from market_regime_engine.mlflow_support.metric_catalog import validate_metric_points
from market_regime_engine.mlflow_support.ports import MetricPoint


def _timestamp_ms(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("fit-quality timestamps must be timezone-aware")
    result = int(value.timestamp() * 1000)
    if result < 0:
        raise ValueError("fit-quality timestamps must be non-negative")
    return result


def _model_shape(candidate_id: str) -> tuple[str, int, int]:
    try:
        family, suffix = candidate_id.rsplit("_k", 1)
        state_count = int(suffix.split("_", 1)[0])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid candidate identity: {candidate_id}") from exc
    if family == "gmm_hmm":
        return family, state_count, 2
    if family in {"gaussian_hmm", "student_t_hmm"}:
        return family, state_count, 1
    raise ValueError(f"unsupported candidate family: {candidate_id}")


def _parameter_count(evaluation: WalkForwardEvaluation) -> int:
    family, state_count, mixture_count = _model_shape(evaluation.candidate_id)
    return information_criteria(
        train_log_likelihood=0.0,
        train_observation_count=1,
        state_count=state_count,
        feature_dimension=len(evaluation.feature_order),
        mixture_count=mixture_count,
        model_family=family,
    ).parameter_count


def _metric_point(
    key: str,
    value: float,
    *,
    step: int,
    timestamp_ms: int,
) -> MetricPoint:
    if not isfinite(value):
        raise ValueError(f"fit-quality metric {key} must be finite")
    return MetricPoint(key=key, value=float(value), step=step, timestamp_ms=timestamp_ms)


@dataclass(frozen=True, slots=True)
class FitQualityEvidence:
    """Immutable fit-quality projection for one candidate evaluation."""

    candidate_id: str
    feature_order: tuple[str, ...]
    parameter_count: int
    metric_points: tuple[MetricPoint, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.feature_order:
            raise ValueError("fit-quality identity cannot be empty")
        if self.parameter_count < 1:
            raise ValueError("fit-quality parameter count must be positive")
        validate_metric_points(self.metric_points)


def _fold_values(evaluation: WalkForwardEvaluation) -> tuple[dict[str, float], ...]:
    parameter_count = _parameter_count(evaluation)
    values: list[dict[str, float]] = []
    for fold in evaluation.valid_folds:
        train_count = fold.train_model_observation_count
        if train_count < 2:
            raise ValueError("HQC requires at least two retained TRAIN observations")
        if fold.train_log_likelihood is None or fold.oos_predictive_log_likelihood is None:
            raise ValueError("valid fold is missing log-likelihood evidence")
        if fold.oos_predictive_log_likelihood_per_observation is None:
            raise ValueError("valid fold is missing normalized OOS likelihood evidence")
        criteria = information_criteria(
            train_log_likelihood=fold.train_log_likelihood,
            train_observation_count=train_count,
            state_count=evaluation.state_count,
            feature_dimension=len(evaluation.feature_order),
            mixture_count=2 if evaluation.candidate_id.startswith("gmm_hmm_") else 1,
            model_family=_model_shape(evaluation.candidate_id)[0],
        )
        hqc = 2.0 * parameter_count * log(log(train_count)) - 2.0 * fold.train_log_likelihood
        values.append(
            {
                "train_loglik_total": fold.train_log_likelihood,
                "train_loglik_per_obs": fold.train_log_likelihood / train_count,
                "oos_predictive_loglik_total": fold.oos_predictive_log_likelihood,
                "oos_predictive_loglik_per_obs": fold.oos_predictive_log_likelihood_per_observation,
                "aic": criteria.aic,
                "bic": criteria.bic,
                "hqc": hqc,
                "aic_per_train_obs": criteria.aic / train_count,
                "bic_per_train_obs": criteria.bic / train_count,
                "hqc_per_train_obs": hqc / train_count,
                "train_observation_count": float(train_count),
            }
        )
    return tuple(values)


def build_fit_quality_evidence(evaluation: WalkForwardEvaluation) -> FitQualityEvidence:
    """Build exact per-fold and population fit-quality histories.

    Aggregates use the valid-fold population only and population standard
    deviation (``ddof=0``), matching the v4 statistical contract.  Invalid
    folds contribute to counts/rates but never to a numerical aggregate.
    """

    if not isinstance(evaluation, WalkForwardEvaluation):
        raise TypeError("fit-quality projection requires a WalkForwardEvaluation")
    timestamp_ms = _timestamp_ms(evaluation.evaluation_cutoff)
    fold_values = _fold_values(evaluation)
    keys = (
        tuple(fold_values[0])
        if fold_values
        else (
            "train_loglik_total",
            "train_loglik_per_obs",
            "oos_predictive_loglik_total",
            "oos_predictive_loglik_per_obs",
            "aic",
            "bic",
            "hqc",
            "aic_per_train_obs",
            "bic_per_train_obs",
            "hqc_per_train_obs",
            "train_observation_count",
        )
    )
    points: list[MetricPoint] = []
    for fold, values in zip(evaluation.valid_folds, fold_values, strict=True):
        for key in keys:
            points.append(
                _metric_point(
                    f"fit_quality_{key}",
                    values[key],
                    step=fold.fold_index,
                    timestamp_ms=timestamp_ms,
                )
            )
    aggregates = {
        "valid_fold_count": float(len(fold_values)),
        "invalid_fold_count": float(len(evaluation.folds) - len(fold_values)),
        "valid_fold_rate": evaluation.valid_fold_rate,
        "parameter_count": float(_parameter_count(evaluation)),
        "feature_dimension": float(len(evaluation.feature_order)),
    }
    for key, value in aggregates.items():
        points.append(_metric_point(f"fit_quality_{key}", value, step=0, timestamp_ms=timestamp_ms))
    for key in keys:
        population = tuple(values[key] for values in fold_values)
        if not population:
            continue
        for suffix, value in (
            ("mean", fmean(population)),
            ("std", pstdev(population)),
            ("minimum", min(population)),
            ("maximum", max(population)),
            ("median", median(population)),
            ("count", float(len(population))),
        ):
            points.append(
                _metric_point(
                    f"fit_quality_{key}_{suffix}",
                    value,
                    step=0,
                    timestamp_ms=timestamp_ms,
                )
            )
    result = tuple(sorted(points, key=lambda point: (point.key, point.step, point.timestamp_ms)))
    return FitQualityEvidence(
        candidate_id=evaluation.candidate_id,
        feature_order=evaluation.feature_order,
        parameter_count=_parameter_count(evaluation),
        metric_points=result,
    )


def validate_fit_quality_comparison(
    evaluations: Mapping[str, WalkForwardEvaluation],
) -> tuple[str, ...]:
    """Require one source/plan/feature vector for likelihood comparisons."""

    if not evaluations:
        raise ValueError("fit-quality comparison requires at least one evaluation")
    identities = {
        (evaluation.source_build_id, evaluation.evaluation_plan_hash, evaluation.feature_order)
        for evaluation in evaluations.values()
    }
    if len(identities) != 1:
        raise ValueError(
            "fit-quality likelihood and information-criterion comparison requires "
            "one source, plan, and feature vector"
        )
    return next(iter(identities))[2]


__all__ = [
    "FitQualityEvidence",
    "build_fit_quality_evidence",
    "validate_fit_quality_comparison",
]
