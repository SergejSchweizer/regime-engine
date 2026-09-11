"""Projection of deterministic v4 diagnostics onto MLflow Model Metrics."""

from __future__ import annotations

from datetime import datetime
from itertools import pairwise
from math import isfinite, log
from statistics import fmean, median, pstdev

import numpy as np

from market_regime_engine.evaluation.diagnostics import (
    gaussian_hmm_parameter_count,
    gmm_hmm_parameter_count,
)
from market_regime_engine.evaluation.walk_forward import (
    WalkForwardEvaluation,
    WalkForwardFoldResult,
)
from market_regime_engine.mlflow_support.metric_catalog import (
    require_metric_definition,
    validate_metric_points,
)
from market_regime_engine.mlflow_support.ports import MetricPoint


def _model_shape(candidate_id: str) -> tuple[str, int, int]:
    if "_k" not in candidate_id:
        raise ValueError(f"cannot derive model shape from candidate ID: {candidate_id}")
    state_count = int(candidate_id.split("_k", 1)[1].split("_", 1)[0])
    if candidate_id.startswith("gmm_hmm_"):
        return "gmm_hmm", state_count, 2
    if candidate_id.startswith("student_t_hmm_"):
        return "student_t_hmm", state_count, 1
    if candidate_id.startswith("gaussian_hmm_"):
        return "gaussian_hmm", state_count, 1
    raise ValueError(f"unsupported candidate ID: {candidate_id}")


def _parameter_count(evaluation: WalkForwardEvaluation) -> int:
    family, state_count, mixture_count = _model_shape(evaluation.candidate_id)
    base = (
        gaussian_hmm_parameter_count(state_count, len(evaluation.feature_order))
        if mixture_count == 1
        else gmm_hmm_parameter_count(state_count, len(evaluation.feature_order), mixture_count)
    )
    return base + (state_count if family == "student_t_hmm" else 0)


def _timestamp_ms(value: datetime) -> int:
    result = int(value.timestamp() * 1000)
    if result < 0:
        raise ValueError("MLflow metric timestamps must be non-negative")
    return result


def _point(key: str, value: float, *, step: int, timestamp_ms: int) -> MetricPoint | None:
    if not isfinite(value):
        return None
    require_metric_definition(key)
    return MetricPoint(key=key, value=float(value), step=step, timestamp_ms=timestamp_ms)


def _append(
    points: list[MetricPoint],
    key: str,
    value: float | int | None,
    *,
    step: int,
    timestamp_ms: int,
) -> None:
    if value is None:
        return
    point = _point(key, float(value), step=step, timestamp_ms=timestamp_ms)
    if point is not None:
        points.append(point)


def _state_durations(probabilities: np.ndarray, state_count: int) -> tuple[float, ...]:
    states = np.argmax(probabilities, axis=1)
    runs: list[list[int]] = [[] for _ in range(state_count)]
    current = int(states[0])
    length = 1
    for raw in states[1:]:
        state = int(raw)
        if state == current:
            length += 1
        else:
            runs[current].append(length)
            current, length = state, 1
    runs[current].append(length)
    return tuple(float(fmean(values)) if values else 0.0 for values in runs)


def _fold_points(
    evaluation: WalkForwardEvaluation,
    fold: WalkForwardFoldResult,
    *,
    timestamp_ms: int,
) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    step = fold.fold_index
    if not fold.valid:
        return points
    train_count = fold.train_model_observation_count
    assert fold.train_log_likelihood is not None
    assert fold.oos_predictive_log_likelihood is not None
    assert fold.oos_predictive_log_likelihood_per_observation is not None
    assert fold.aic is not None and fold.bic is not None
    _append(
        points,
        "train_loglik_total",
        fold.train_log_likelihood,
        step=step,
        timestamp_ms=timestamp_ms,
    )
    _append(
        points,
        "train_loglik_per_obs",
        fold.train_log_likelihood / train_count,
        step=step,
        timestamp_ms=timestamp_ms,
    )
    _append(
        points,
        "oos_predictive_loglik_total",
        fold.oos_predictive_log_likelihood,
        step=step,
        timestamp_ms=timestamp_ms,
    )
    _append(
        points,
        "oos_predictive_loglik_per_obs",
        fold.oos_predictive_log_likelihood_per_observation,
        step=step,
        timestamp_ms=timestamp_ms,
    )
    _append(points, "aic", fold.aic, step=step, timestamp_ms=timestamp_ms)
    _append(points, "bic", fold.bic, step=step, timestamp_ms=timestamp_ms)
    _append(
        points,
        "aic_per_train_obs",
        fold.aic / train_count,
        step=step,
        timestamp_ms=timestamp_ms,
    )
    _append(
        points,
        "bic_per_train_obs",
        fold.bic / train_count,
        step=step,
        timestamp_ms=timestamp_ms,
    )
    hqc = (
        2.0 * _parameter_count(evaluation) * log(log(train_count)) - 2.0 * fold.train_log_likelihood
    )
    _append(points, "hqc", hqc, step=step, timestamp_ms=timestamp_ms)
    for key, value in {
        "multistart_attempt_count": len(fold.multistart_result.diagnostics)
        if fold.multistart_result
        else None,
        "multistart_success_count": fold.multistart_result.valid_start_count
        if fold.multistart_result
        else None,
        "multistart_failure_count": (
            len(fold.multistart_result.diagnostics) - fold.multistart_result.valid_start_count
            if fold.multistart_result
            else None
        ),
        "multistart_success_rate": fold.multistart_success_rate,
        "covariance_min_diagonal_variance": None,
    }.items():
        _append(points, key, value, step=step, timestamp_ms=timestamp_ms)

    if fold.multistart_result is not None:
        successful = tuple(
            item
            for item in fold.multistart_result.diagnostics
            if item.success and item.train_log_likelihood is not None
        )
        ranked = tuple(
            sorted(
                successful, key=lambda item: (-float(item.train_log_likelihood or 0.0), item.seed)
            )
        )
        ranks = {item.seed: rank for rank, item in enumerate(ranked, start=1)}
        seed_values = tuple(
            float(item.train_log_likelihood)
            for item in successful
            if item.train_log_likelihood is not None
        )
        _append(
            points,
            "best_seed_loglik",
            max(seed_values) if seed_values else None,
            step=step,
            timestamp_ms=timestamp_ms,
        )
        _append(
            points,
            "median_seed_loglik",
            median(seed_values) if seed_values else None,
            step=step,
            timestamp_ms=timestamp_ms,
        )
        _append(
            points,
            "worst_seed_loglik",
            min(seed_values) if seed_values else None,
            step=step,
            timestamp_ms=timestamp_ms,
        )
        _append(
            points,
            "seed_loglik_spread",
            max(seed_values) - min(seed_values) if seed_values else None,
            step=step,
            timestamp_ms=timestamp_ms,
        )
        for ordinal, diagnostic in enumerate(fold.multistart_result.diagnostics, start=1):
            _append(
                points,
                f"seed_train_loglik_fold_{fold.fold_index}",
                diagnostic.train_log_likelihood,
                step=ordinal,
                timestamp_ms=timestamp_ms,
            )
            _append(
                points,
                f"em_iteration_count_fold_{fold.fold_index}",
                diagnostic.iterations,
                step=ordinal,
                timestamp_ms=timestamp_ms,
            )
            _append(
                points,
                f"em_converged_fold_{fold.fold_index}",
                float(diagnostic.converged),
                step=ordinal,
                timestamp_ms=timestamp_ms,
            )
            _append(
                points,
                f"seed_train_loglik_per_obs_fold_{fold.fold_index}",
                None
                if diagnostic.train_log_likelihood is None
                else diagnostic.train_log_likelihood / train_count,
                step=ordinal,
                timestamp_ms=timestamp_ms,
            )
            if diagnostic.success:
                _append(
                    points,
                    f"seed_rank_fold_{fold.fold_index}",
                    ranks[diagnostic.seed],
                    step=ordinal,
                    timestamp_ms=timestamp_ms,
                )
        history = fold.multistart_result.winner.em_log_likelihood_history
        for iteration, value in enumerate(history, start=1):
            _append(
                points,
                f"em_loglik_per_obs_fold_{fold.fold_index}",
                value / train_count,
                step=iteration,
                timestamp_ms=timestamp_ms,
            )
        if history:
            _append(points, "em_final_loglik", history[-1], step=step, timestamp_ms=timestamp_ms)
            _append(
                points,
                "em_final_delta",
                history[-1] - history[-2] if len(history) > 1 else 0.0,
                step=step,
                timestamp_ms=timestamp_ms,
            )
            _append(
                points,
                "em_monotonicity_violation_count",
                sum(current < previous for previous, current in pairwise(history)),
                step=step,
                timestamp_ms=timestamp_ms,
            )

    vectors = {
        "train_hard": fold.train_hard_occupancy,
        "train_soft": fold.train_soft_occupancy,
        "oos_hard": fold.oos_hard_occupancy,
        "oos_soft": fold.oos_soft_occupancy,
    }
    for prefix, vector in vectors.items():
        if vector is None:
            continue
        for state, value in enumerate(vector):
            _append(
                points,
                f"state_occupancy_{prefix}_state_{state}",
                value,
                step=step,
                timestamp_ms=timestamp_ms,
            )

    probabilities = np.asarray(fold.oos_filtered_probabilities, dtype=np.float64)
    if probabilities.size:
        state_count = probabilities.shape[1]
        entropy = -np.sum(
            np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0), axis=1
        )
        confidence = np.max(probabilities, axis=1)
        _append(
            points,
            "filtered_entropy_mean",
            float(np.mean(entropy)),
            step=step,
            timestamp_ms=timestamp_ms,
        )
        _append(
            points,
            "filtered_entropy_std",
            float(np.std(entropy)),
            step=step,
            timestamp_ms=timestamp_ms,
        )
        _append(
            points,
            "filtered_entropy_min",
            float(np.min(entropy)),
            step=step,
            timestamp_ms=timestamp_ms,
        )
        _append(
            points,
            "filtered_entropy_max",
            float(np.max(entropy)),
            step=step,
            timestamp_ms=timestamp_ms,
        )
        _append(
            points,
            "dominant_state_confidence_mean",
            float(np.mean(confidence)),
            step=step,
            timestamp_ms=timestamp_ms,
        )
        _append(
            points,
            "dominant_state_confidence_std",
            float(np.std(confidence)),
            step=step,
            timestamp_ms=timestamp_ms,
        )
        durations = _state_durations(probabilities, state_count)
        for state in range(state_count):
            _append(
                points,
                f"state_filtered_probability_mean_state_{state}",
                float(np.mean(probabilities[:, state])),
                step=step,
                timestamp_ms=timestamp_ms,
            )
            _append(
                points,
                f"state_filtered_probability_std_state_{state}",
                float(np.std(probabilities[:, state])),
                step=step,
                timestamp_ms=timestamp_ms,
            )
            _append(
                points,
                f"state_expected_duration_state_{state}",
                durations[state],
                step=step,
                timestamp_ms=timestamp_ms,
            )
        observation_offset = sum(
            item.test_model_observation_count
            for item in evaluation.folds
            if item.fold_index < fold.fold_index
        )
        for observation, row in enumerate(probabilities):
            for state, value in enumerate(row):
                _append(
                    points,
                    f"oos_filtered_probability_state_{state}",
                    float(value),
                    step=observation_offset + observation + 1,
                    timestamp_ms=(
                        _timestamp_ms(fold.oos_timestamps[observation])
                        if observation < len(fold.oos_timestamps)
                        else timestamp_ms
                    ),
                )
            _append(
                points,
                "viterbi_state",
                float(np.argmax(row)),
                step=observation_offset + observation + 1,
                timestamp_ms=(
                    _timestamp_ms(fold.oos_timestamps[observation])
                    if observation < len(fold.oos_timestamps)
                    else timestamp_ms
                ),
            )

    if fold.model_artifact is not None and fold.alignment is not None:
        mapping = fold.alignment.persistent_to_fitted
        transition = np.asarray(fold.model_artifact.transition_matrix, dtype=np.float64)
        aligned = transition[np.ix_(mapping, mapping)]
        for state_row_index in range(aligned.shape[0]):
            row_values = aligned[state_row_index]
            row_entropy = float(
                -sum(float(value) * log(float(value)) for value in row_values if value > 0.0)
            )
            _append(
                points,
                f"transition_row_entropy_state_{state_row_index}",
                row_entropy,
                step=step,
                timestamp_ms=timestamp_ms,
            )
            for column, value in enumerate(row_values):
                _append(
                    points,
                    f"transition_probability_state_{state_row_index}_to_state_{column}",
                    float(value),
                    step=step,
                    timestamp_ms=timestamp_ms,
                )
        for state, value in enumerate(np.diag(aligned)):
            _append(
                points,
                f"transition_self_probability_state_{state}",
                float(value),
                step=step,
                timestamp_ms=timestamp_ms,
            )
        _append(
            points,
            "transition_min_probability",
            float(np.min(aligned)),
            step=step,
            timestamp_ms=timestamp_ms,
        )
        _append(
            points,
            "transition_max_probability",
            float(np.max(aligned)),
            step=step,
            timestamp_ms=timestamp_ms,
        )

        covariances = np.asarray(fold.model_artifact.distribution_covariances(), dtype=np.float64)[
            list(mapping)
        ]
        eigenvalues = np.linalg.eigvalsh((covariances + np.swapaxes(covariances, 1, 2)) / 2.0)
        positive = eigenvalues[eigenvalues > 0.0]
        if positive.size:
            minimum = float(np.min(positive))
            maximum = float(np.max(positive))
            _append(
                points, "covariance_min_eigenvalue", minimum, step=step, timestamp_ms=timestamp_ms
            )
            _append(
                points, "covariance_max_eigenvalue", maximum, step=step, timestamp_ms=timestamp_ms
            )
            _append(
                points,
                "covariance_condition_number",
                maximum / minimum,
                step=step,
                timestamp_ms=timestamp_ms,
            )
        _append(
            points,
            "covariance_min_diagonal_variance",
            float(np.min(np.diagonal(covariances, axis1=1, axis2=2))),
            step=step,
            timestamp_ms=timestamp_ms,
        )
        means = np.asarray(fold.model_artifact.means, dtype=np.float64)[list(mapping)]
        variances = np.diagonal(covariances, axis1=1, axis2=2)
        for state in range(means.shape[0]):
            for feature in range(means.shape[1]):
                _append(
                    points,
                    f"emission_mean_state_{state}_feature_{feature}",
                    float(means[state, feature]),
                    step=step,
                    timestamp_ms=timestamp_ms,
                )
                _append(
                    points,
                    f"emission_variance_state_{state}_feature_{feature}",
                    float(variances[state, feature]),
                    step=step,
                    timestamp_ms=timestamp_ms,
                )

    return points


def model_metric_points(
    evaluation: WalkForwardEvaluation,
    *,
    fold_timestamps: tuple[datetime, ...] | None = None,
) -> tuple[MetricPoint, ...]:
    """Return all finite candidate diagnostics as deterministic metric points."""

    if fold_timestamps is not None and len(fold_timestamps) != len(evaluation.folds):
        raise ValueError("fold_timestamps must match the candidate fold count")
    default_timestamp = _timestamp_ms(evaluation.evaluation_cutoff)
    points: list[MetricPoint] = []
    valid = evaluation.valid_folds
    aggregate_values = [
        item.oos_predictive_log_likelihood_per_observation
        for item in valid
        if item.oos_predictive_log_likelihood_per_observation is not None
    ]
    aggregate = {
        "valid_fold_rate": evaluation.valid_fold_rate,
        "valid_fold_count": len(valid),
        "invalid_fold_count": len(evaluation.folds) - len(valid),
        "selected_K": evaluation.state_count,
        "oos_predictive_loglik_mean": fmean(aggregate_values) if aggregate_values else None,
        "oos_predictive_loglik_std": pstdev(aggregate_values) if aggregate_values else None,
        "oos_predictive_loglik_worst_fold": min(aggregate_values) if aggregate_values else None,
        "oos_predictive_loglik_best_fold": max(aggregate_values) if aggregate_values else None,
        "aic": fmean(item.aic for item in valid if item.aic is not None) if valid else None,
        "bic": fmean(item.bic for item in valid if item.bic is not None) if valid else None,
        "feature_dimension": len(evaluation.feature_order),
        "parameter_count": _parameter_count(evaluation),
    }
    for key, value in aggregate.items():
        _append(points, key, value, step=0, timestamp_ms=default_timestamp)
    for index, fold in enumerate(evaluation.folds):
        timestamp_ms = (
            _timestamp_ms(fold_timestamps[index])
            if fold_timestamps is not None
            else default_timestamp
        )
        points.extend(_fold_points(evaluation, fold, timestamp_ms=timestamp_ms))
    result = tuple(points)
    validate_metric_points(result)
    return result


def outer_selection_metric_points(selection: object, outer_fold: object) -> tuple[MetricPoint, ...]:
    """Project feature-discovery and outer-policy context onto a selected model."""

    timestamp = _timestamp_ms(outer_fold.test_end)  # type: ignore[attr-defined]
    prefix = next(
        item
        for item in selection.prefix_search.evaluations  # type: ignore[attr-defined]
        if item.prefix_length == selection.prefix_search.selected_prefix_length  # type: ignore[attr-defined]
    )
    values = {
        "eligible_feature_count_N": len(selection.quality.eligible_features),  # type: ignore[attr-defined]
        "selected_cluster_count_M": selection.clusters.selected_count,  # type: ignore[attr-defined]
        "selected_feature_count_L": selection.prefix_search.selected_prefix_length,  # type: ignore[attr-defined]
        "selected_state_count_K": selection.final_candidate.state_count,  # type: ignore[attr-defined]
        "selected_M_silhouette": selection.clusters.selected_silhouette,  # type: ignore[attr-defined]
        "prefix_soft_regime_nmi": prefix.soft_regime_nmi,
        "outer_soft_regime_nmi": outer_fold.outer_teacher_final_soft_nmi,  # type: ignore[attr-defined]
        "outer_shared_timestamp_count": outer_fold.outer_shared_timestamp_count,  # type: ignore[attr-defined]
        "outer_oos_predictive_loglik_per_obs": outer_fold.oos_predictive_loglik_per_observation,  # type: ignore[attr-defined]
    }
    points: list[MetricPoint] = []
    for key, value in values.items():
        _append(points, key, value, step=outer_fold.fold_index, timestamp_ms=timestamp)  # type: ignore[attr-defined]
    result = tuple(points)
    validate_metric_points(result)
    return result


__all__ = ["model_metric_points", "outer_selection_metric_points"]
