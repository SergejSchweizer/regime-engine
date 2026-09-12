"""Build independent-math expectations from one completed v4 evaluation.

This module prepares primitive arrays for ``verify_xetra_v4_math.py``. The
verifier remains a separate implementation; this adapter only serializes
already-persisted selection/model evidence and never performs the audited
formulas itself.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from typing import cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluation.walk_forward import (
    WalkForwardEvaluation,
    WalkForwardFoldResult,
)
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardFold
from market_regime_engine.evaluations.global_regime_v4 import V4ConfigurationSelection
from market_regime_engine.feature_discovery.contracts import AdaptiveEvaluationResult
from market_regime_engine.inference.filtering import causal_filter


def _complete_rows(source_rows: pd.DataFrame, feature_order: tuple[str, ...]) -> np.ndarray:
    frame = source_rows.loc[:, list(feature_order)]
    values = cast(np.ndarray, frame.to_numpy(dtype=np.float64))
    complete = frame.notna().all(axis=1).to_numpy() & np.isfinite(values).all(axis=1)
    return cast(np.ndarray, values[complete])


def _labels(
    feature_order: tuple[str, ...],
    memberships: tuple[tuple[str, tuple[str, ...]], ...],
) -> list[int]:
    labels = {
        feature: cluster_index
        for cluster_index, (_cluster_id, features) in enumerate(memberships)
        for feature in features
    }
    if set(labels) != set(feature_order):
        raise ValueError("cluster membership audit does not cover the feature order")
    return [labels[feature] for feature in feature_order]


def _likelihood_item(
    source_rows: pd.DataFrame,
    feature_order: tuple[str, ...],
    plan_fold: WalkForwardFold,
    fold: WalkForwardFoldResult,
) -> list[dict[str, object]]:
    model_artifact = fold.model_artifact
    scaler = fold.scaler_artifact
    if model_artifact is None or scaler is None or not fold.valid:
        raise ValueError("likelihood audit requires a valid fitted fold")
    train_source = source_rows.iloc[: plan_fold.train_source_observations]
    test_source = source_rows.iloc[
        plan_fold.train_source_observations : plan_fold.train_source_observations
        + plan_fold.test_source_observations
    ]
    train_values = scaler.transform(_complete_rows(train_source, feature_order))
    test_values = scaler.transform(_complete_rows(test_source, feature_order))
    train_filter = causal_filter(train_values, model_artifact)
    continuation_start = tuple(
        float(value)
        for value in (
            np.asarray(train_filter.terminal_probabilities, dtype=np.float64)
            @ np.asarray(model_artifact.transition_matrix, dtype=np.float64)
        )
    )
    common: dict[str, object] = {
        "model_family": model_artifact.model_family,
        "start_probabilities": list(model_artifact.start_probabilities),
        "transition_matrix": [list(row) for row in model_artifact.transition_matrix],
        "means": [list(row) for row in model_artifact.means],
        "covariances": [
            [list(row) for row in covariance] for covariance in model_artifact.full_covariances
        ],
    }
    if model_artifact.model_family == "gmm_hmm":
        if (
            model_artifact.mixture_weights is None
            or model_artifact.mixture_means is None
            or model_artifact.mixture_full_covariances is None
        ):
            raise ValueError("GMM audit artifact is missing mixture parameters")
        common.update(
            {
                "mixture_weights": [list(row) for row in model_artifact.mixture_weights],
                "mixture_means": [
                    [list(component) for component in state]
                    for state in model_artifact.mixture_means
                ],
                "mixture_covariances": [
                    [[list(row) for row in component] for component in state]
                    for state in model_artifact.mixture_full_covariances
                ],
            }
        )
    elif model_artifact.model_family == "student_t_hmm":
        if model_artifact.degrees_of_freedom is None:
            raise ValueError("Student-t audit artifact is missing degrees of freedom")
        common["degrees_of_freedom"] = list(model_artifact.degrees_of_freedom)
    train_item = {
        **common,
        "fold_index": fold.fold_index,
        "scope": "TRAIN",
        "observations": train_values.tolist(),
        "log_likelihood": fold.train_log_likelihood,
    }
    oos_item = {
        **common,
        "fold_index": fold.fold_index,
        "scope": "OOS",
        "observations": test_values.tolist(),
        "start_probabilities": list(continuation_start),
        "log_likelihood": fold.oos_predictive_log_likelihood,
    }
    return [train_item, oos_item]


def _prefix_nmi_items(
    selection: V4ConfigurationSelection,
    outer_fold_index: int,
    payloads: Mapping[str, bytes],
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    scope_prefix = f"v4_stage/scope=fold_{outer_fold_index:03d}:prefix_candidate:"
    for prefix in selection.prefix_search.evaluations:
        if not prefix.valid:
            continue
        key_prefix = f"{scope_prefix}{prefix.prefix_length}:{prefix.candidate_id}/"
        payload = next(
            (value for key, value in payloads.items() if key.startswith(key_prefix)),
            None,
        )
        if payload is None:
            raise ValueError(f"missing persisted prefix candidate payload for {key_prefix}")
        evaluation = pickle.loads(payload)
        if not isinstance(evaluation, WalkForwardEvaluation):
            raise ValueError("persisted prefix candidate payload has an invalid type")
        timestamps: list[object] = []
        probabilities: list[list[float]] = []
        for fold in evaluation.valid_folds:
            timestamps.extend(fold.oos_timestamps)
            probabilities.extend(list(row) for row in fold.oos_filtered_probabilities)
        if not timestamps:
            raise ValueError("valid prefix candidate has no persisted OOS support")
        teacher = selection.teacher_reference
        items.append(
            {
                "outer_fold_index": outer_fold_index,
                "prefix_length": prefix.prefix_length,
                "candidate_id": prefix.candidate_id,
                "candidate_timestamps": timestamps,
                "candidate_probabilities": probabilities,
                "teacher_timestamps": list(teacher.timestamps),
                "teacher_probabilities": [list(row) for row in teacher.filtered_probabilities],
                "soft_regime_nmi": prefix.soft_regime_nmi,
            }
        )
    return items


def build_math_expectations(
    source_rows: pd.DataFrame,
    result: AdaptiveEvaluationResult,
    selections: Mapping[int, V4ConfigurationSelection],
    prefix_payloads: Mapping[int, Mapping[str, bytes]] | None = None,
) -> dict[str, object]:
    """Serialize first-fold math evidence and representative final likelihoods."""

    if not result.outer_folds:
        raise ValueError("math audit requires at least one outer fold")
    first_outer = next(
        (fold for fold in result.outer_folds if fold.fold_index in selections),
        None,
    )
    if first_outer is None:
        raise ValueError("no valid outer-fold selection is available for math audit")
    selection = selections[first_outer.fold_index]
    feature_order = selection.distance.feature_order
    clusters = []
    silhouette_by_count = dict(selection.clusters.silhouette_curve)
    for count, memberships in selection.clusters.candidate_memberships:
        clusters.append(
            {
                "cluster_count": count,
                "labels": _labels(feature_order, memberships),
                "mean": silhouette_by_count[count],
            }
        )
    teacher = selection.teacher_reference
    feature_scores = [
        {
            "feature": score.feature_name,
            "teacher_timestamps": list(teacher.timestamps),
            "teacher_probabilities": [list(row) for row in teacher.filtered_probabilities],
            "state_information_ratio": score.state_information_ratio,
            "eta_squared": score.eta_squared,
        }
        for score in selection.feature_scores
        if score.eligible
        and score.state_information_ratio is not None
        and score.eta_squared is not None
    ]

    grid = selection.final_grid.grid
    final_evaluation = next(
        evaluation
        for evaluation in grid.evaluations
        if evaluation.candidate_id == selection.final_candidate.candidate_id
    )
    if selection.final_grid_plan is None:
        raise ValueError("final-grid plan is required for likelihood audit")
    valid_folds = [fold for fold in final_evaluation.folds if fold.valid]
    if not valid_folds:
        raise ValueError("final candidate has no valid folds for likelihood audit")
    selected_indices = tuple(dict.fromkeys((0, len(valid_folds) // 2, len(valid_folds) - 1)))
    likelihoods: list[dict[str, object]] = []
    for index in selected_indices:
        fold = valid_folds[index]
        plan_fold = selection.final_grid_plan.folds[fold.fold_index - 1]
        likelihoods.extend(
            _likelihood_item(source_rows, final_evaluation.feature_order, plan_fold, fold)
        )

    prefix_nmi: list[dict[str, object]] = []
    if prefix_payloads is not None:
        for outer_fold_index, selected in sorted(selections.items()):
            prefix_nmi.extend(
                _prefix_nmi_items(
                    selected,
                    outer_fold_index,
                    prefix_payloads.get(outer_fold_index, {}),
                )
            )

    return {
        "timestamp_column": "timestamp_m1",
        "feature_order": list(feature_order),
        "distance": [list(row) for row in selection.distance.distances],
        "silhouette_clusters": clusters,
        "feature_scores": feature_scores,
        "prefix_nmi": prefix_nmi,
        "likelihoods": likelihoods,
        "outer_fold_index": first_outer.fold_index,
        "final_candidate_id": selection.final_candidate.candidate_id,
    }


__all__ = ["build_math_expectations"]
