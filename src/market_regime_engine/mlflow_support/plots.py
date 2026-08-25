"""Deterministic MLflow evaluation plots governed by PLOT_STYLE.md."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.dates as mdates
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from market_regime_engine.evaluation.walk_forward import (
    WalkForwardEvaluation,
    WalkForwardFoldResult,
)
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan

PNG_DPI = 180
WIDE_FIGSIZE = (10.0, 5.5)
SQUARE_FIGSIZE = (7.0, 6.5)
COMPARISON_FIGSIZE = (11.0, 8.0)

_FOLD_METRICS: dict[str, tuple[str, str]] = {
    "fold_train_loglik": ("train_log_likelihood", "TRAIN log likelihood"),
    "fold_oos_predictive_loglik": (
        "oos_predictive_log_likelihood",
        "OOS predictive log likelihood",
    ),
    "fold_oos_predictive_loglik_per_obs": (
        "oos_predictive_log_likelihood_per_observation",
        "OOS predictive log likelihood per observation",
    ),
    "fold_aic": ("aic", "AIC"),
    "fold_bic": ("bic", "BIC"),
    "fold_aic_per_train_obs": ("aic", "AIC per TRAIN observation"),
    "fold_bic_per_train_obs": ("bic", "BIC per TRAIN observation"),
    "fold_multistart_success_rate": (
        "multistart_success_rate",
        "Multistart success rate",
    ),
    "fold_min_train_hard_occupancy": (
        "train_hard_occupancy",
        "Minimum TRAIN hard occupancy fraction",
    ),
    "fold_min_train_soft_occupancy": (
        "train_soft_occupancy",
        "Minimum TRAIN soft occupancy fraction",
    ),
    "fold_max_state_signature_drift": (
        "max_state_signature_drift",
        "State-signature drift",
    ),
    "fold_mean_state_duration": (
        "mean_state_duration",
        "Mean state duration (observations)",
    ),
    "fold_switches_per_year": ("switches_per_year", "Switches per year"),
    "fold_oos_entropy_mean": ("oos_entropy_mean", "Mean OOS entropy"),
    "fold_oos_confidence_mean": ("oos_confidence_mean", "Mean OOS confidence"),
}


@dataclass(frozen=True, slots=True)
class PlotManifestEntry:
    png_path: str
    plot_type: str
    candidate_id: str
    fold_id: str | None
    source_metric_keys: tuple[str, ...]
    x_axis_field: str
    x_axis_label: str
    y_axis_label: str
    legend_entries: tuple[str, ...]
    image_dimensions_inches: tuple[float, float]
    dpi: int
    source_artifact_hash: str
    scale_bounds: tuple[float, float] | None = None

    def as_json_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _validate_plan(evaluation: WalkForwardEvaluation, plan: WalkForwardPlan) -> None:
    if evaluation.evaluation_plan_hash != plan.plan_hash:
        raise ValueError("evaluation and walk-forward plan hashes differ")
    if len(evaluation.folds) != len(plan.folds):
        raise ValueError("evaluation and walk-forward plan fold counts differ")
    if tuple(fold.fold_id for fold in evaluation.folds) != tuple(
        fold.fold_id for fold in plan.folds
    ):
        raise ValueError("evaluation and walk-forward plan fold IDs differ")


def _metric_value(fold: WalkForwardFoldResult, metric_key: str) -> float | None:
    attribute = _FOLD_METRICS[metric_key][0]
    value = getattr(fold, attribute)
    if metric_key in {"fold_min_train_hard_occupancy", "fold_min_train_soft_occupancy"}:
        vector = cast(tuple[float, ...] | None, value)
        return None if vector is None else min(vector)
    scalar = cast(float | None, value)
    if scalar is not None and not isfinite(scalar):
        raise ValueError(f"{metric_key} must be finite when present")
    if metric_key in {"fold_aic_per_train_obs", "fold_bic_per_train_obs"}:
        if fold.train_model_observation_count < 1:
            raise ValueError("normalized information criteria require positive TRAIN observations")
        return None if scalar is None else scalar / fold.train_model_observation_count
    return scalar


def _date_axis(ax: matplotlib.axes.Axes, x_values: tuple[datetime, ...]) -> np.ndarray:
    dates = mdates.date2num(list(x_values))  # type: ignore[no-untyped-call]
    locator = mdates.AutoDateLocator()  # type: ignore[no-untyped-call]
    formatter = mdates.ConciseDateFormatter(locator)  # type: ignore[no-untyped-call]
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    return np.asarray(dates, dtype=np.float64)


def _save_figure(fig: matplotlib.figure.Figure, base_path: Path) -> Path:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = base_path.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)
    return png_path


def render_fold_history(
    evaluation: WalkForwardEvaluation,
    plan: WalkForwardPlan,
    metric_key: str,
    output_dir: str | Path,
) -> PlotManifestEntry:
    """Render one candidate history using real TEST-end UTC values and NaN invalid gaps."""

    _validate_plan(evaluation, plan)
    if metric_key not in _FOLD_METRICS:
        raise ValueError(f"unsupported fold-history metric: {metric_key}")
    x_values = tuple(fold.test_end for fold in plan.folds)
    raw_values = tuple(
        _metric_value(fold, metric_key) if fold.valid else None for fold in evaluation.folds
    )
    y_values = np.asarray(
        [float("nan") if value is None else value for value in raw_values], dtype=np.float64
    )
    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)
    x_plot = _date_axis(ax, x_values)
    ax.plot(x_plot, y_values, marker="o", label=evaluation.candidate_id)
    ax.set_title(f"{metric_key} — {evaluation.candidate_id}")
    ax.set_xlabel("Test window end (UTC)")
    ax.set_ylabel(_FOLD_METRICS[metric_key][1])
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    base = Path(output_dir) / evaluation.candidate_id / "histories" / metric_key
    png_path = _save_figure(fig, base)
    payload = {
        "candidate_id": evaluation.candidate_id,
        "metric_key": metric_key,
        "test_end": [value.isoformat() for value in x_values],
        "values": raw_values,
    }
    return PlotManifestEntry(
        png_path=str(png_path),
        plot_type="fold_history",
        candidate_id=evaluation.candidate_id,
        fold_id=None,
        source_metric_keys=(metric_key,),
        x_axis_field="test_end",
        x_axis_label="Test window end (UTC)",
        y_axis_label=_FOLD_METRICS[metric_key][1],
        legend_entries=(evaluation.candidate_id,),
        image_dimensions_inches=WIDE_FIGSIZE,
        dpi=PNG_DPI,
        source_artifact_hash=_canonical_hash(payload),
    )


def candidate_covariance_scale(evaluation: WalkForwardEvaluation) -> float:
    """Return one shared absolute covariance scale for every valid fold/state heatmap."""

    maxima: list[float] = []
    for fold in evaluation.valid_folds:
        if fold.model_artifact is None:
            raise ValueError("valid fold is missing model artifact")
        values = np.asarray(fold.model_artifact.full_covariances, dtype=np.float64)
        maxima.append(float(np.max(np.abs(values))))
    if not maxima:
        raise ValueError("candidate has no valid covariance artifacts")
    scale = max(maxima)
    if not isfinite(scale) or scale <= 0.0:
        raise ValueError("candidate covariance scale must be finite and positive")
    return scale


def render_transition_heatmap(
    evaluation: WalkForwardEvaluation,
    fold: WalkForwardFoldResult,
    output_dir: str | Path,
) -> PlotManifestEntry:
    """Render one persistent-state transition matrix on the fixed probability scale [0,1]."""

    if not fold.valid or fold.model_artifact is None or fold.alignment is None:
        raise ValueError("transition heatmap requires a valid aligned fold")
    mapping = fold.alignment.persistent_to_fitted
    raw = np.asarray(fold.model_artifact.transition_matrix, dtype=np.float64)
    matrix = raw[np.ix_(mapping, mapping)]
    state_ids = fold.alignment.persistent_state_ids
    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Transition probability")
    ax.set_title(f"Transition matrix — {evaluation.candidate_id} — {fold.fold_id}")
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    ax.set_xticks(range(len(state_ids)), labels=state_ids)
    ax.set_yticks(range(len(state_ids)), labels=state_ids)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center")
    base = Path(output_dir) / evaluation.candidate_id / fold.fold_id / "transition_matrix"
    png_path = _save_figure(fig, base)
    payload = {
        "candidate_id": evaluation.candidate_id,
        "fold_id": fold.fold_id,
        "matrix": matrix.tolist(),
    }
    return PlotManifestEntry(
        png_path=str(png_path),
        plot_type="transition_heatmap",
        candidate_id=evaluation.candidate_id,
        fold_id=fold.fold_id,
        source_metric_keys=("transition_matrix",),
        x_axis_field="persistent_state_id",
        x_axis_label="Next state",
        y_axis_label="Current state",
        legend_entries=state_ids,
        image_dimensions_inches=SQUARE_FIGSIZE,
        dpi=PNG_DPI,
        source_artifact_hash=_canonical_hash(payload),
        scale_bounds=(0.0, 1.0),
    )


def _state_persistence_history(
    evaluation: WalkForwardEvaluation,
    plan: WalkForwardPlan,
) -> tuple[tuple[str, ...], tuple[datetime, ...], np.ndarray]:
    """Return aligned self-transition probabilities for every planned fold."""

    _validate_plan(evaluation, plan)
    first_valid = next((fold for fold in evaluation.folds if fold.valid), None)
    if first_valid is None or first_valid.alignment is None:
        raise ValueError("state persistence plots require at least one valid aligned fold")
    state_ids = first_valid.alignment.persistent_state_ids
    values = np.full((evaluation.state_count, len(plan.folds)), np.nan, dtype=np.float64)
    for column, fold in enumerate(evaluation.folds):
        if not fold.valid or fold.model_artifact is None or fold.alignment is None:
            continue
        matrix = np.asarray(fold.model_artifact.transition_matrix, dtype=np.float64)
        mapping = fold.alignment.persistent_to_fitted
        aligned = matrix[np.ix_(mapping, mapping)]
        values[:, column] = np.diag(aligned)
    return state_ids, tuple(item.test_end for item in plan.folds), values


def _state_occupancy_history(
    evaluation: WalkForwardEvaluation,
    plan: WalkForwardPlan,
) -> tuple[tuple[str, ...], tuple[datetime, ...], np.ndarray]:
    """Return OOS soft-occupancy percentages for every planned fold.

    Each valid column sums to 100 percent: it is the expected share of retained
    OOS observations spent in each persistent state, not a transition probability.
    """

    _validate_plan(evaluation, plan)
    first_valid = next((fold for fold in evaluation.folds if fold.valid), None)
    if first_valid is None or first_valid.alignment is None:
        raise ValueError("state occupancy matrix requires at least one valid aligned fold")
    state_ids = first_valid.alignment.persistent_state_ids
    values = np.full((evaluation.state_count, len(plan.folds)), np.nan, dtype=np.float64)
    for column, fold in enumerate(evaluation.folds):
        if not fold.valid:
            continue
        if fold.oos_soft_occupancy is None:
            raise ValueError("valid fold is missing OOS soft occupancy")
        occupancy = np.asarray(fold.oos_soft_occupancy, dtype=np.float64)
        if occupancy.shape != (evaluation.state_count,):
            raise ValueError("OOS soft occupancy dimensions do not match the candidate")
        if not np.all(np.isfinite(occupancy)) or np.any(occupancy < 0.0):
            raise ValueError("OOS soft occupancy must be finite and nonnegative")
        if not np.isclose(np.sum(occupancy), 1.0, rtol=0.0, atol=1e-10):
            raise ValueError("OOS soft occupancy must sum to one")
        values[:, column] = occupancy * 100.0
    return state_ids, tuple(item.test_end for item in plan.folds), values


def render_state_persistence_matrix(
    evaluation: WalkForwardEvaluation,
    plan: WalkForwardPlan,
    output_dir: str | Path,
) -> PlotManifestEntry:
    """Render persistent-state OOS occupancy across Walk-forward history."""

    state_ids, test_ends, values = _state_occupancy_history(evaluation, plan)
    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)
    image = ax.imshow(values, vmin=0.0, vmax=100.0, aspect="auto")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Share of OOS observations (%)")
    ax.set_title(f"State occupancy matrix — {evaluation.candidate_id}")
    ax.set_xlabel("Test window end (UTC)")
    ax.set_ylabel("Persistent state")
    ax.set_yticks(range(len(state_ids)), labels=state_ids)
    tick_count = min(9, len(test_ends))
    tick_positions = np.unique(np.linspace(0, len(test_ends) - 1, tick_count, dtype=np.intp))
    ax.set_xticks(
        tick_positions,
        labels=[test_ends[index].date().isoformat() for index in tick_positions],
    )
    ax.tick_params(axis="x", rotation=45)
    base = Path(output_dir) / evaluation.candidate_id / "state_persistence_matrix"
    png_path = _save_figure(fig, base)
    return PlotManifestEntry(
        png_path=str(png_path),
        plot_type="state_persistence_matrix",
        candidate_id=evaluation.candidate_id,
        fold_id=None,
        source_metric_keys=("fold_oos_soft_occupancy",),
        x_axis_field="test_end",
        x_axis_label="Test window end (UTC)",
        y_axis_label="Persistent state",
        legend_entries=state_ids,
        image_dimensions_inches=WIDE_FIGSIZE,
        dpi=PNG_DPI,
        source_artifact_hash=_canonical_hash(
            {
                "test_end": [item.isoformat() for item in test_ends],
                "values_percent": values.tolist(),
            }
        ),
        scale_bounds=(0.0, 100.0),
    )


def render_state_transition_history(
    evaluation: WalkForwardEvaluation,
    plan: WalkForwardPlan,
    output_dir: str | Path,
) -> PlotManifestEntry:
    """Render each persistent state's self-transition probability through history."""

    state_ids, test_ends, values = _state_persistence_history(evaluation, plan)
    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)
    x_plot = _date_axis(ax, test_ends)
    for index, state_id in enumerate(state_ids):
        ax.plot(x_plot, values[index], marker="o", label=state_id)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"State transition history — {evaluation.candidate_id}")
    ax.set_xlabel("Test window end (UTC)")
    ax.set_ylabel("Self-transition probability")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    base = Path(output_dir) / evaluation.candidate_id / "state_transition_history"
    png_path = _save_figure(fig, base)
    return PlotManifestEntry(
        png_path=str(png_path),
        plot_type="state_transition_history",
        candidate_id=evaluation.candidate_id,
        fold_id=None,
        source_metric_keys=("fold_self_transition",),
        x_axis_field="test_end",
        x_axis_label="Test window end (UTC)",
        y_axis_label="Self-transition probability",
        legend_entries=state_ids,
        image_dimensions_inches=WIDE_FIGSIZE,
        dpi=PNG_DPI,
        source_artifact_hash=_canonical_hash(
            {"test_end": [item.isoformat() for item in test_ends], "values": values.tolist()}
        ),
        scale_bounds=(0.0, 1.0),
    )


def _state_feature_separation(
    evaluation: WalkForwardEvaluation,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return median aligned emission separation per persistent state and feature.

    A value is the state mean minus the mean of the other state means, divided by
    that state's marginal emission standard deviation.  The HMM is trained on
    standardized feature rows, so this is a dimensionless, comparable measure of
    how strongly an input feature separates a state from the remaining states.
    """

    first_valid = next((fold for fold in evaluation.folds if fold.valid), None)
    if first_valid is None or first_valid.alignment is None:
        raise ValueError("state feature influence requires at least one valid aligned fold")

    state_ids = first_valid.alignment.persistent_state_ids
    fold_values: list[np.ndarray] = []
    for fold in evaluation.valid_folds:
        if fold.model_artifact is None or fold.alignment is None:
            raise ValueError("valid fold is missing aligned model artifacts")
        mapping = np.asarray(fold.alignment.persistent_to_fitted, dtype=np.intp)
        means = np.asarray(fold.model_artifact.means, dtype=np.float64)[mapping]
        covariances = np.asarray(fold.model_artifact.full_covariances, dtype=np.float64)[mapping]
        variances = np.diagonal(covariances, axis1=1, axis2=2)
        if means.shape != (evaluation.state_count, len(evaluation.feature_order)):
            raise ValueError("aligned mean dimensions do not match the candidate")
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(variances)):
            raise ValueError("emission parameters must be finite")
        if np.any(variances <= 0.0):
            raise ValueError("emission variances must be positive")
        other_means = (np.sum(means, axis=0, keepdims=True) - means) / (evaluation.state_count - 1)
        fold_values.append((means - other_means) / np.sqrt(variances))

    summary = np.median(np.stack(fold_values), axis=0)
    if not np.all(np.isfinite(summary)):
        raise ValueError("state feature influence summary must be finite")
    return state_ids, summary


def render_state_feature_influence(
    evaluation: WalkForwardEvaluation,
    output_dir: str | Path,
) -> PlotManifestEntry:
    """Render a candidate's median state-wise standardized emission separation."""

    state_ids, values = _state_feature_separation(evaluation)
    features = evaluation.feature_order
    scale = float(np.max(np.abs(values)))
    if not isfinite(scale):
        raise ValueError("state feature influence scale must be finite")
    plot_scale = max(scale, 1.0)
    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)
    image = ax.imshow(values, vmin=-plot_scale, vmax=plot_scale, cmap="coolwarm", aspect="auto")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("State separation (state standard deviations)")
    ax.set_title(f"Input-feature influence by persistent state — {evaluation.candidate_id}")
    ax.set_xlabel("Input feature")
    ax.set_ylabel("Persistent state")
    ax.set_xticks(range(len(features)), labels=features, rotation=45, ha="right")
    ax.set_yticks(range(len(state_ids)), labels=state_ids)
    if len(features) <= 12:
        for row in range(values.shape[0]):
            strongest = int(np.argmax(np.abs(values[row])))
            for column in range(values.shape[1]):
                marker = " *" if column == strongest else ""
                ax.text(
                    column,
                    row,
                    f"{values[row, column]:.2f}{marker}",
                    ha="center",
                    va="center",
                )
    base = Path(output_dir) / evaluation.candidate_id / "state_feature_influence"
    png_path = _save_figure(fig, base)
    payload = {
        "candidate_id": evaluation.candidate_id,
        "feature_order": features,
        "state_ids": state_ids,
        "values": values.tolist(),
        "scale": plot_scale,
    }
    return PlotManifestEntry(
        png_path=str(png_path),
        plot_type="state_feature_influence",
        candidate_id=evaluation.candidate_id,
        fold_id=None,
        source_metric_keys=("aligned_emission_mean", "aligned_emission_variance"),
        x_axis_field="feature_order",
        x_axis_label="Input feature",
        y_axis_label="Persistent state",
        legend_entries=state_ids,
        image_dimensions_inches=WIDE_FIGSIZE,
        dpi=PNG_DPI,
        source_artifact_hash=_canonical_hash(payload),
        scale_bounds=(-plot_scale, plot_scale),
    )


def render_covariance_heatmap(
    evaluation: WalkForwardEvaluation,
    fold: WalkForwardFoldResult,
    persistent_state_index: int,
    shared_scale: float,
    output_dir: str | Path,
) -> PlotManifestEntry:
    """Render one aligned full covariance matrix with a candidate-wide deterministic scale."""

    if not fold.valid or fold.model_artifact is None or fold.alignment is None:
        raise ValueError("covariance heatmap requires a valid aligned fold")
    if persistent_state_index not in range(evaluation.state_count):
        raise ValueError("persistent state index is outside candidate state range")
    if not isfinite(shared_scale) or shared_scale <= 0.0:
        raise ValueError("shared covariance scale must be finite and positive")
    fitted_index = fold.alignment.persistent_to_fitted[persistent_state_index]
    matrix = np.asarray(fold.model_artifact.full_covariances[fitted_index], dtype=np.float64)
    features = evaluation.feature_order
    state_id = fold.alignment.persistent_state_ids[persistent_state_index]
    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    image = ax.imshow(matrix, vmin=-shared_scale, vmax=shared_scale, cmap="coolwarm")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Covariance")
    ax.set_title(f"Full covariance — {evaluation.candidate_id} — {fold.fold_id} — {state_id}")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")
    ax.set_xticks(range(len(features)), labels=features, rotation=45, ha="right")
    ax.set_yticks(range(len(features)), labels=features)
    if len(features) <= 8:
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center")
    base = Path(output_dir) / evaluation.candidate_id / fold.fold_id / f"covariance_{state_id}"
    png_path = _save_figure(fig, base)
    payload = {
        "candidate_id": evaluation.candidate_id,
        "fold_id": fold.fold_id,
        "state_id": state_id,
        "feature_order": features,
        "matrix": matrix.tolist(),
        "shared_scale": shared_scale,
    }
    return PlotManifestEntry(
        png_path=str(png_path),
        plot_type="full_covariance_heatmap",
        candidate_id=evaluation.candidate_id,
        fold_id=fold.fold_id,
        source_metric_keys=("full_covariance",),
        x_axis_field="feature_order",
        x_axis_label="Feature",
        y_axis_label="Feature",
        legend_entries=(state_id,),
        image_dimensions_inches=SQUARE_FIGSIZE,
        dpi=PNG_DPI,
        source_artifact_hash=_canonical_hash(payload),
        scale_bounds=(-shared_scale, shared_scale),
    )


def render_candidate_comparison(
    evaluations: tuple[WalkForwardEvaluation, ...],
    plan: WalkForwardPlan,
    output_dir: str | Path,
) -> PlotManifestEntry:
    """Compare valid-fold OOS PLL/observation histories without shifting missing folds."""

    if not evaluations:
        raise ValueError("candidate comparison requires at least one evaluation")
    ordered = tuple(sorted(evaluations, key=lambda item: (item.state_count, item.candidate_id)))
    for evaluation in ordered:
        _validate_plan(evaluation, plan)
    x_values = tuple(fold.test_end for fold in plan.folds)
    fig, (score_ax, gap_ax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=COMPARISON_FIGSIZE,
        sharex=True,
        height_ratios=(2.0, 1.0),
    )
    x_plot = _date_axis(score_ax, x_values)
    source_values: dict[str, list[float | None]] = {}
    plotted_values: list[np.ndarray] = []
    line_colors: list[str] = []
    for evaluation in ordered:
        values = [
            _metric_value(fold, "fold_oos_predictive_loglik_per_obs") if fold.valid else None
            for fold in evaluation.folds
        ]
        source_values[evaluation.candidate_id] = values
        y_values = np.asarray(
            [float("nan") if value is None else value for value in values], dtype=np.float64
        )
        line = score_ax.plot(x_plot, y_values, marker="o", label=evaluation.candidate_id)[0]
        plotted_values.append(y_values)
        line_colors.append(line.get_color())
    matrix = np.asarray(plotted_values, dtype=np.float64)
    finite_by_fold = np.any(np.isfinite(matrix), axis=0)
    best_values = np.full(len(x_values), np.nan, dtype=np.float64)
    best_values[finite_by_fold] = np.max(matrix[:, finite_by_fold], axis=0)
    relative_to_best = matrix - best_values
    leaders = np.full(len(x_values), -1, dtype=np.intp)
    leaders[finite_by_fold] = np.argmax(matrix[:, finite_by_fold], axis=0)
    if len(x_plot) == 1:
        boundaries = np.asarray((x_plot[0] - 0.5, x_plot[0] + 0.5), dtype=np.float64)
    else:
        midpoints = (x_plot[:-1] + x_plot[1:]) / 2.0
        boundaries = np.concatenate(
            (
                np.asarray((x_plot[0] - (midpoints[0] - x_plot[0]),)),
                midpoints,
                np.asarray((x_plot[-1] + (x_plot[-1] - midpoints[-1]),)),
            )
        )
    segment_start = 0
    while segment_start < len(leaders):
        leader = leaders[segment_start]
        segment_end = segment_start + 1
        while segment_end < len(leaders) and leaders[segment_end] == leader:
            segment_end += 1
        if leader >= 0:
            for axis in (score_ax, gap_ax):
                axis.axvspan(
                    boundaries[segment_start],
                    boundaries[segment_end],
                    color=line_colors[leader],
                    alpha=0.09,
                    linewidth=0,
                    zorder=0,
                )
        segment_start = segment_end
    for evaluation, values, color in zip(ordered, relative_to_best, line_colors, strict=True):
        gap_ax.plot(x_plot, values, marker="o", label=evaluation.candidate_id, color=color)
    score_ax.set_title("Walk-forward OOS candidate comparison")
    score_ax.set_ylabel("OOS predictive log likelihood per observation")
    score_ax.grid(True, alpha=0.25)
    score_ax.legend(title="Candidate / shaded best candidate", loc="lower left")
    gap_ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    gap_ax.set_xlabel("Test window end (UTC)")
    gap_ax.set_ylabel("Gap to best OOS score\n(0 = best)")
    gap_ax.grid(True, alpha=0.25)
    gap_ax.set_title("Relative score makes candidate leadership transitions visible")
    _date_axis(gap_ax, x_values)
    fig.autofmt_xdate()
    base = Path(output_dir) / "parent" / "candidate_oos_predictive_loglik_per_obs"
    png_path = _save_figure(fig, base)
    payload = {
        "test_end": [value.isoformat() for value in x_values],
        "values": source_values,
    }
    candidate_ids = tuple(evaluation.candidate_id for evaluation in ordered)
    return PlotManifestEntry(
        png_path=str(png_path),
        plot_type="candidate_comparison",
        candidate_id="all_candidates",
        fold_id=None,
        source_metric_keys=("fold_oos_predictive_loglik_per_obs",),
        x_axis_field="test_end",
        x_axis_label="Test window end (UTC)",
        y_axis_label="Absolute OOS score and gap to best OOS score",
        legend_entries=candidate_ids,
        image_dimensions_inches=COMPARISON_FIGSIZE,
        dpi=PNG_DPI,
        source_artifact_hash=_canonical_hash(payload),
    )


def fold_history_metric_keys() -> tuple[str, ...]:
    return tuple(_FOLD_METRICS)
