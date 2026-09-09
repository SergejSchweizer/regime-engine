"""Deterministic, PNG-only diagnostics for global-regime v4 evidence.

The global v4 policy deliberately separates adaptive feature selection from
the final, same-vector model comparison.  These plots preserve that boundary:
they visualise feature-discovery and agreement evidence across outer folds,
and show likelihood only within a single frozen final-grid feature vector.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from market_regime_engine.feature_discovery.contracts import (
    AdaptiveEvaluationResult,
    ClusterSolution,
    OuterFoldResult,
)

if TYPE_CHECKING:
    from market_regime_engine.evaluations.global_regime_v4 import V4ConfigurationSelection


_DPI = 180
_FIGURE_SIZE = (11.0, 6.5)
_WIDE_FIGURE_SIZE = (13.0, 7.0)
_TITLE_SIZE = 15
_LABEL_SIZE = 11
_TICK_SIZE = 9
_LEGEND_SIZE = 8
_PALETTE = (
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
)


@dataclass(frozen=True, slots=True)
class GlobalV4PlotManifestEntry:
    """Lineage for one human-facing global-v4 diagnostic PNG."""

    plot_type: str
    png_path: str
    source_artifact_hash: str
    source_metric_keys: tuple[str, ...]
    x_axis_field: str
    x_axis_label: str
    y_axis_label: str
    legend_entries: tuple[str, ...]
    candidate_id: str | None = None
    fold_id: str | None = None
    width_inches: float = _FIGURE_SIZE[0]
    height_inches: float = _FIGURE_SIZE[1]
    dpi: int = _DPI

    def as_dict(self) -> dict[str, object]:
        return {
            "plot_type": self.plot_type,
            "png_path": self.png_path,
            "source_artifact_hash": self.source_artifact_hash,
            "source_metric_keys": list(self.source_metric_keys),
            "x_axis_field": self.x_axis_field,
            "x_axis_label": self.x_axis_label,
            "y_axis_label": self.y_axis_label,
            "legend_entries": list(self.legend_entries),
            "candidate_id": self.candidate_id,
            "fold_id": self.fold_id,
            "image_dimensions_inches": [self.width_inches, self.height_inches],
            "dpi": self.dpi,
        }


def _fold_id(fold_index: int) -> str:
    return f"outer_fold_{fold_index:03d}"


def _json_hash(payload: object) -> str:
    """Hash only JSON primitives used to render a plot, never run metadata."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _configure(axis: Axes, *, title: str, x_label: str, y_label: str) -> None:
    axis.set_title(title, fontsize=_TITLE_SIZE, pad=12)
    axis.set_xlabel(x_label, fontsize=_LABEL_SIZE)
    axis.set_ylabel(y_label, fontsize=_LABEL_SIZE)
    axis.tick_params(labelsize=_TICK_SIZE)
    axis.grid(True, alpha=0.24, linewidth=0.8)


def _finish(
    root: Path,
    *,
    plot_type: str,
    filename: str,
    source: object,
    source_metric_keys: tuple[str, ...],
    x_axis_field: str,
    x_axis_label: str,
    y_axis_label: str,
    legend_entries: tuple[str, ...],
    draw: Callable[[Axes], None],
    candidate_id: str | None = None,
    fold_id: str | None = None,
    figure_size: tuple[float, float] = _FIGURE_SIZE,
) -> GlobalV4PlotManifestEntry:
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=figure_size)
    try:
        draw(axis)
        figure.tight_layout()
        figure.savefig(path, dpi=_DPI, bbox_inches="tight", format="png")
    finally:
        plt.close(figure)
    return GlobalV4PlotManifestEntry(
        plot_type=plot_type,
        png_path=str(path),
        source_artifact_hash=_json_hash(source),
        source_metric_keys=source_metric_keys,
        x_axis_field=x_axis_field,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        legend_entries=legend_entries,
        candidate_id=candidate_id,
        fold_id=fold_id,
        width_inches=figure_size[0],
        height_inches=figure_size[1],
    )


def _selections_by_fold(
    selections: Mapping[int, V4ConfigurationSelection],
    result: AdaptiveEvaluationResult,
) -> tuple[tuple[OuterFoldResult, V4ConfigurationSelection], ...]:
    result_by_index = {fold.fold_index: fold for fold in result.outer_folds}
    unknown = set(selections) - set(result_by_index)
    if unknown:
        raise ValueError("global v4 plot selection records reference unknown outer folds")
    missing_valid = {fold.fold_index for fold in result.outer_folds if fold.valid} - set(selections)
    if missing_valid:
        raise ValueError("global v4 plots require selection evidence for every valid outer fold")
    ordered = tuple((result_by_index[index], selections[index]) for index in sorted(selections))
    if not ordered:
        raise ValueError("global v4 plots require at least one outer-fold selection")
    return ordered


def _fold_dates(
    items: tuple[tuple[OuterFoldResult, V4ConfigurationSelection], ...],
) -> tuple[datetime, ...]:
    return tuple(fold.test_end for fold, _selection in items)


def _format_dates(axis: Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)  # type: ignore[no-untyped-call]
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))  # type: ignore[no-untyped-call]


def _date_numbers(dates: tuple[datetime, ...]) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Return Matplotlib epoch-day positions without relying on untyped date helpers."""

    return np.asarray([date.timestamp() / 86_400.0 for date in dates], dtype=np.float64)


def _cluster_jaccard(previous: ClusterSolution, current: ClusterSolution) -> float:
    previous_sets = tuple({*members} for _cluster, members in previous.memberships)
    current_sets = tuple({*members} for _cluster, members in current.memberships)
    if not previous_sets or not current_sets:
        raise ValueError("cluster stability requires non-empty memberships")
    best_scores = []
    for members in current_sets:
        best_scores.append(
            max(len(members & other) / len(members | other) for other in previous_sets)
        )
    return float(sum(best_scores) / len(best_scores))


def _quality_plot(
    root: Path, items: tuple[tuple[OuterFoldResult, V4ConfigurationSelection], ...]
) -> GlobalV4PlotManifestEntry:
    labels = tuple(_fold_id(fold.fold_index) for fold, _selection in items)
    total = tuple(len(selection.quality.features) for _fold, selection in items)
    eligible = tuple(len(selection.quality.eligible_features) for _fold, selection in items)
    source = {"folds": labels, "catalog_features": total, "eligible_features": eligible}

    def draw(axis: Axes) -> None:
        x_values = np.arange(len(labels))
        axis.bar(x_values - 0.2, total, width=0.4, label="Catalog features", color=_PALETTE[0])
        axis.bar(
            x_values + 0.2,
            eligible,
            width=0.4,
            label="Eligible after quality gates",
            color=_PALETTE[2],
        )
        axis.set_xticks(x_values, labels, rotation=35, ha="right")
        _configure(
            axis,
            title="Outer-TRAIN feature quality eligibility",
            x_label="Outer fold",
            y_label="Feature count",
        )
        axis.legend(fontsize=_LEGEND_SIZE)

    return _finish(
        root,
        plot_type="quality_eligibility",
        filename="quality_eligibility.png",
        source=source,
        source_metric_keys=("quality.catalog_feature_count", "quality.eligible_feature_count"),
        x_axis_field="outer_fold_id",
        x_axis_label="Outer fold",
        y_axis_label="Feature count",
        legend_entries=("Catalog features", "Eligible after quality gates"),
        draw=draw,
    )


def _silhouette_plot(
    root: Path, items: tuple[tuple[OuterFoldResult, V4ConfigurationSelection], ...]
) -> GlobalV4PlotManifestEntry:
    source = {
        _fold_id(fold.fold_index): {
            "curve": list(selection.clusters.silhouette_curve),
            "selected_m": selection.clusters.selected_count,
        }
        for fold, selection in items
    }
    legend = tuple(_fold_id(fold.fold_index) for fold, _selection in items)

    def draw(axis: Axes) -> None:
        for index, (fold, selection) in enumerate(items):
            curve = tuple(selection.clusters.silhouette_curve)
            x_values = [item[0] for item in curve]
            y_values = [item[1] for item in curve]
            label = _fold_id(fold.fold_index)
            axis.plot(
                x_values,
                y_values,
                marker=("o", "s", "^", "D", "P", "X", "v")[index % 7],
                color=_PALETTE[index % len(_PALETTE)],
                label=label,
            )
            selected = selection.clusters.selected_count
            axis.scatter(
                [selected],
                [dict(curve)[selected]],
                marker="*",
                s=130,
                color=_PALETTE[index % len(_PALETTE)],
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
        _configure(
            axis,
            title="Outer-TRAIN clustering silhouette curves (star = selected M*)",
            x_label="Candidate cluster count M",
            y_label="Silhouette score",
        )
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--", label="Zero reference")
        axis.legend(fontsize=_LEGEND_SIZE, ncol=2)

    return _finish(
        root,
        plot_type="silhouette_curve",
        filename="silhouette_curve.png",
        source=source,
        source_metric_keys=("clustering.silhouette_curve", "clustering.selected_m"),
        x_axis_field="candidate_cluster_count",
        x_axis_label="Candidate cluster count M",
        y_axis_label="Silhouette score",
        legend_entries=(*legend, "Zero reference"),
        draw=draw,
    )


def _cluster_size_plot(
    root: Path, items: tuple[tuple[OuterFoldResult, V4ConfigurationSelection], ...]
) -> GlobalV4PlotManifestEntry:
    records = tuple(
        (
            _fold_id(fold.fold_index),
            tuple(len(members) for _cluster, members in selection.clusters.memberships),
        )
        for fold, selection in items
    )
    source = {label: sizes for label, sizes in records}

    def draw(axis: Axes) -> None:
        for index, (label, sizes) in enumerate(records):
            positions = np.arange(1, len(sizes) + 1) + index * 0.10
            axis.scatter(
                positions,
                sizes,
                color=_PALETTE[index % len(_PALETTE)],
                marker=("o", "s", "^", "D", "P", "X", "v")[index % 7],
                label=label,
            )
        _configure(
            axis,
            title="Selected cluster sizes by Outer-TRAIN fold",
            x_label="Canonical cluster position within fold",
            y_label="Features in cluster",
        )
        axis.legend(fontsize=_LEGEND_SIZE, ncol=2)

    return _finish(
        root,
        plot_type="cluster_size",
        filename="cluster_size.png",
        source=source,
        source_metric_keys=("clustering.memberships",),
        x_axis_field="canonical_cluster_position",
        x_axis_label="Canonical cluster position within fold",
        y_axis_label="Features in cluster",
        legend_entries=tuple(label for label, _sizes in records),
        draw=draw,
    )


def _state_information_plots(
    root: Path, items: tuple[tuple[OuterFoldResult, V4ConfigurationSelection], ...]
) -> tuple[GlobalV4PlotManifestEntry, ...]:
    entries: list[GlobalV4PlotManifestEntry] = []
    for fold, selection in items:
        fold_id = _fold_id(fold.fold_index)
        scores = tuple(sorted(selection.feature_scores, key=lambda item: item.canonical_ordinal))
        values = tuple(
            float("nan") if item.state_information_ratio is None else item.state_information_ratio
            for item in scores
        )
        eta = tuple(
            float("nan") if item.eta_squared is None else item.eta_squared for item in scores
        )
        names = tuple(item.feature_name for item in scores)
        prototypes = set(selection.prototypes.prototypes)
        winners = set(selection.winner_selection.ranked_features)
        source = {
            "fold_id": fold_id,
            "scores": [
                {
                    "feature": name,
                    "state_information_ratio": value,
                    "eta_squared": diagnostic,
                    "prototype": name in prototypes,
                    "winner": name in winners,
                }
                for name, value, diagnostic in zip(names, values, eta, strict=True)
            ],
        }

        def draw(
            axis: Axes,
            names: tuple[str, ...] = names,
            values: tuple[float, ...] = values,
            eta: tuple[float, ...] = eta,
            prototypes: set[str] = prototypes,
            winners: set[str] = winners,
            fold_id: str = fold_id,
        ) -> None:
            positions = np.arange(len(names))
            axis.bar(positions, values, color=_PALETTE[0], label="State-information ratio")
            axis.scatter(
                positions,
                eta,
                marker="x",
                s=48,
                linewidths=1.7,
                color=_PALETTE[1],
                label="Eta-squared diagnostic",
                zorder=3,
            )
            prototype_label_emitted = False
            winner_label_emitted = False
            for position, name, value in zip(positions, names, values, strict=True):
                if name in prototypes:
                    axis.scatter(
                        [position],
                        [value],
                        marker="P",
                        s=100,
                        color=_PALETTE[4],
                        edgecolors="black",
                        linewidths=0.5,
                        label="Temporary prototype"
                        if not prototype_label_emitted
                        else "_nolegend_",
                        zorder=4,
                    )
                    prototype_label_emitted = True
                if name in winners:
                    axis.scatter(
                        [position],
                        [value],
                        marker="*",
                        s=155,
                        color=_PALETTE[2],
                        edgecolors="black",
                        linewidths=0.5,
                        label="Cluster winner" if not winner_label_emitted else "_nolegend_",
                        zorder=5,
                    )
                    winner_label_emitted = True
            axis.set_xticks(positions, names, rotation=45, ha="right")
            _configure(
                axis,
                title=f"State-information feature rank — {fold_id}",
                x_label="Feature (canonical ordinal order)",
                y_label="Score / diagnostic fraction",
            )
            axis.set_ylim(bottom=0.0)
            axis.legend(fontsize=_LEGEND_SIZE)

        entries.append(
            _finish(
                root,
                plot_type="state_information_rank",
                filename=f"state_information_rank_{fold_id}.png",
                source=source,
                source_metric_keys=(
                    "feature_scores.state_information_ratio",
                    "feature_scores.eta_squared",
                    "prototypes.features",
                    "winner_selection.ranked_features",
                ),
                x_axis_field="feature_canonical_ordinal",
                x_axis_label="Feature (canonical ordinal order)",
                y_axis_label="Score / diagnostic fraction",
                legend_entries=(
                    "State-information ratio",
                    "Eta-squared diagnostic",
                    "Temporary prototype",
                    "Cluster winner",
                ),
                draw=draw,
                fold_id=fold_id,
            )
        )
    return tuple(entries)


def _prefix_plot(
    root: Path, items: tuple[tuple[OuterFoldResult, V4ConfigurationSelection], ...]
) -> GlobalV4PlotManifestEntry:
    source = {
        _fold_id(fold.fold_index): [
            {
                "prefix_length": item.prefix_length,
                "soft_regime_nmi": item.soft_regime_nmi,
                "selected": item.prefix_length == selection.prefix_search.selected_prefix_length,
            }
            for item in selection.prefix_search.evaluations
        ]
        for fold, selection in items
    }
    legend = tuple(_fold_id(fold.fold_index) for fold, _selection in items)

    def draw(axis: Axes) -> None:
        for index, (fold, selection) in enumerate(items):
            evaluation = tuple(selection.prefix_search.evaluations)
            x_values = [item.prefix_length for item in evaluation]
            y_values = [item.soft_regime_nmi for item in evaluation]
            color = _PALETTE[index % len(_PALETTE)]
            axis.plot(
                x_values,
                y_values,
                marker=("o", "s", "^", "D", "P", "X", "v")[index % 7],
                color=color,
                label=_fold_id(fold.fold_index),
            )
            selected = selection.prefix_search.selected_prefix_length
            axis.scatter(
                [selected],
                [dict(zip(x_values, y_values, strict=True))[selected]],
                marker="*",
                s=130,
                color=color,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
        _configure(
            axis,
            title="Prefix search: soft NMI to the causal teacher (star = selected L*)",
            x_label="Prefix length L",
            y_label="Soft regime NMI",
        )
        axis.set_ylim(-0.02, 1.02)
        axis.legend(fontsize=_LEGEND_SIZE, ncol=2)

    return _finish(
        root,
        plot_type="prefix_soft_nmi_history",
        filename="prefix_soft_nmi_history.png",
        source=source,
        source_metric_keys=("prefix_search.prefix_length", "prefix_search.soft_regime_nmi"),
        x_axis_field="prefix_length",
        x_axis_label="Prefix length L",
        y_axis_label="Soft regime NMI",
        legend_entries=legend,
        draw=draw,
    )


def _final_grid_plots(
    root: Path, items: tuple[tuple[OuterFoldResult, V4ConfigurationSelection], ...]
) -> tuple[GlobalV4PlotManifestEntry, ...]:
    entries: list[GlobalV4PlotManifestEntry] = []
    for fold, selection in items:
        fold_id = _fold_id(fold.fold_index)
        aggregates = tuple(selection.final_grid.candidate_grid.aggregates)
        ids = tuple(item.candidate_id for item in aggregates)
        optional_scores = tuple(item.oos_predictive_loglik_mean for item in aggregates)
        if len(ids) != 12 or any(value is None for value in optional_scores):
            raise ValueError("final v4 plot requires all twelve comparable candidate aggregates")
        scores = tuple(float(value) for value in optional_scores if value is not None)
        source = {
            "fold_id": fold_id,
            "feature_order": list(selection.final_candidate.feature_order),
            "plan_hash": selection.final_grid.candidate_grid.evaluation_plan_hash,
            "candidate_oos_predictive_loglik_mean": list(zip(ids, scores, strict=True)),
            "champion": selection.final_candidate.candidate_id,
        }

        def draw(
            axis: Axes,
            ids: tuple[str, ...] = ids,
            scores: tuple[float, ...] = scores,
            champion: str = selection.final_candidate.candidate_id,
            fold_id: str = fold_id,
        ) -> None:
            positions = np.arange(len(ids))
            colors = [
                _PALETTE[2] if candidate_id == champion else _PALETTE[0] for candidate_id in ids
            ]
            axis.bar(positions, scores, color=colors)
            axis.set_xticks(positions, ids, rotation=45, ha="right")
            _configure(
                axis,
                title=(
                    f"Final 12-model OOS comparison — {fold_id} (one frozen feature vector / plan)"
                ),
                x_label="Canonical candidate ID",
                y_label="Mean OOS predictive log likelihood per observation",
            )
            axis.legend(
                handles=[
                    Rectangle((0, 0), 1, 1, color=_PALETTE[2], label="Selected champion"),
                    Rectangle((0, 0), 1, 1, color=_PALETTE[0], label="Other candidate"),
                ],
                fontsize=_LEGEND_SIZE,
            )

        entries.append(
            _finish(
                root,
                plot_type="final_12_model_same_vector_comparison",
                filename=f"final_12_model_same_vector_{fold_id}.png",
                source=source,
                source_metric_keys=(
                    "final_grid.candidate_aggregates.oos_predictive_loglik_mean",
                    "final_grid.feature_order",
                    "final_grid.evaluation_plan_hash",
                ),
                x_axis_field="candidate_id",
                x_axis_label="Canonical candidate ID",
                y_axis_label="Mean OOS predictive log likelihood per observation",
                legend_entries=("Selected champion", "Other candidate"),
                draw=draw,
                candidate_id=selection.final_candidate.candidate_id,
                fold_id=fold_id,
                figure_size=_WIDE_FIGURE_SIZE,
            )
        )
    return tuple(entries)


def _outer_nmi_plot(root: Path, result: AdaptiveEvaluationResult) -> GlobalV4PlotManifestEntry:
    dates = tuple(fold.test_end for fold in result.outer_folds)
    x_values = _date_numbers(dates)
    values = tuple(
        fold.outer_teacher_final_soft_nmi if fold.valid else None for fold in result.outer_folds
    )
    valid_mask = np.asarray([value is not None for value in values])
    numeric = np.asarray([np.nan if value is None else value for value in values], dtype=float)
    legend_entries = (
        ("Valid outer fold", "Invalid outer fold (no NMI)")
        if not np.all(valid_mask)
        else ("Valid outer fold",)
    )
    source = {
        "test_end": [value.isoformat() for value in dates],
        "outer_teacher_final_soft_nmi": list(values),
        "valid": [fold.valid for fold in result.outer_folds],
    }

    def draw(axis: Axes) -> None:
        axis.plot(x_values, numeric, marker="o", color=_PALETTE[0], label="Valid outer fold")
        if not np.all(valid_mask):
            invalid_x_values = [
                value for value, valid in zip(x_values, valid_mask, strict=True) if not valid
            ]
            axis.scatter(
                invalid_x_values,
                [0.0] * len(invalid_x_values),
                marker="x",
                s=55,
                color="#000000",
                label="Invalid outer fold (no NMI)",
                zorder=3,
            )
        _configure(
            axis,
            title="Outer TEST agreement with the frozen causal teacher",
            x_label="Test window end (UTC)",
            y_label="Soft regime NMI",
        )
        axis.set_ylim(-0.02, 1.02)
        _format_dates(axis)
        axis.legend(fontsize=_LEGEND_SIZE)

    return _finish(
        root,
        plot_type="outer_soft_nmi_history",
        filename="outer_soft_nmi_history.png",
        source=source,
        source_metric_keys=("agreement.outer_teacher_final_soft_nmi", "outer_folds.valid"),
        x_axis_field="outer_fold.test_end",
        x_axis_label="Test window end (UTC)",
        y_axis_label="Soft regime NMI",
        legend_entries=legend_entries,
        draw=draw,
    )


def _selection_history_plot(
    root: Path, items: tuple[tuple[OuterFoldResult, V4ConfigurationSelection], ...]
) -> GlobalV4PlotManifestEntry:
    dates = _fold_dates(items)
    x_values = _date_numbers(dates)
    m_values = tuple(selection.clusters.selected_count for _fold, selection in items)
    l_values = tuple(selection.prefix_search.selected_prefix_length for _fold, selection in items)
    source = {
        "test_end": [value.isoformat() for value in dates],
        "selected_m": list(m_values),
        "selected_l": list(l_values),
    }

    def draw(axis: Axes) -> None:
        axis.plot(x_values, m_values, marker="o", color=_PALETTE[0], label="Selected M*")
        axis.plot(x_values, l_values, marker="s", color=_PALETTE[1], label="Selected L*")
        _configure(
            axis,
            title="Outer-TRAIN selected clustering and prefix complexity",
            x_label="Test window end (UTC)",
            y_label="Selected count",
        )
        _format_dates(axis)
        axis.legend(fontsize=_LEGEND_SIZE)

    return _finish(
        root,
        plot_type="selected_m_l_history",
        filename="selected_m_l_history.png",
        source=source,
        source_metric_keys=("clustering.selected_m", "prefix_search.selected_l"),
        x_axis_field="outer_fold.test_end",
        x_axis_label="Test window end (UTC)",
        y_axis_label="Selected count",
        legend_entries=("Selected M*", "Selected L*"),
        draw=draw,
    )


def _feature_frequency_plot(
    root: Path, items: tuple[tuple[OuterFoldResult, V4ConfigurationSelection], ...]
) -> GlobalV4PlotManifestEntry:
    count = Counter(
        feature for _fold, selection in items for feature in selection.final_candidate.feature_order
    )
    ordered = tuple(sorted(count, key=lambda name: (-count[name], name)))
    values = tuple(count[name] / len(items) for name in ordered)
    source = {
        "outer_fold_count": len(items),
        "selection_frequency": list(zip(ordered, values, strict=True)),
    }

    def draw(axis: Axes) -> None:
        positions = np.arange(len(ordered))
        axis.bar(positions, values, color=_PALETTE[2])
        axis.set_xticks(positions, ordered, rotation=45, ha="right")
        _configure(
            axis,
            title="Final feature-selection frequency across Outer-TRAIN folds",
            x_label="Feature",
            y_label="Selection frequency (fraction of outer folds)",
        )
        axis.set_ylim(0.0, 1.05)

    return _finish(
        root,
        plot_type="feature_selection_frequency",
        filename="feature_selection_frequency.png",
        source=source,
        source_metric_keys=("outer_folds.final_configuration.feature_order",),
        x_axis_field="feature_name",
        x_axis_label="Feature",
        y_axis_label="Selection frequency (fraction of outer folds)",
        legend_entries=(),
        draw=draw,
    )


def _cluster_stability_plot(
    root: Path, items: tuple[tuple[OuterFoldResult, V4ConfigurationSelection], ...]
) -> GlobalV4PlotManifestEntry:
    labels = tuple(
        f"{_fold_id(previous.fold_index)}→{_fold_id(current.fold_index)}"
        for (previous, _), (current, _) in pairwise(items)
    )
    values = tuple(
        _cluster_jaccard(previous_selection.clusters, current_selection.clusters)
        for (_previous, previous_selection), (_current, current_selection) in pairwise(items)
    )
    source = {"adjacent_fold_pair": list(labels), "mean_best_cluster_jaccard": list(values)}

    def draw(axis: Axes) -> None:
        if values:
            positions = np.arange(len(values))
            axis.plot(positions, values, marker="o", color=_PALETTE[3], label="Mean best Jaccard")
            axis.set_xticks(positions, labels, rotation=35, ha="right")
            axis.legend(fontsize=_LEGEND_SIZE)
        else:
            axis.text(
                0.5,
                0.5,
                "One outer fold: no adjacent-fold stability pair",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        _configure(
            axis,
            title="Adjacent Outer-TRAIN cluster-membership stability",
            x_label="Adjacent outer-fold pair",
            y_label="Mean best cluster Jaccard similarity",
        )
        axis.set_ylim(-0.02, 1.02)

    return _finish(
        root,
        plot_type="adjacent_fold_cluster_stability",
        filename="adjacent_fold_cluster_stability.png",
        source=source,
        source_metric_keys=("stability.adjacent_fold_cluster_membership_jaccard",),
        x_axis_field="adjacent_outer_fold_pair",
        x_axis_label="Adjacent outer-fold pair",
        y_axis_label="Mean best cluster Jaccard similarity",
        legend_entries=("Mean best Jaccard",) if values else (),
        draw=draw,
    )


def render_global_v4_diagnostics(
    result: AdaptiveEvaluationResult,
    selections: Mapping[int, V4ConfigurationSelection],
    output_dir: Path,
) -> tuple[GlobalV4PlotManifestEntry, ...]:
    """Render every v4 diagnostic from fold-local selection evidence.

    The function is intentionally fail-closed.  A missing selection record or
    an incomplete same-vector final grid raises instead of emitting a partial
    success artifact.  It intentionally never renders raw PLL as a prefix or
    cross-outer-fold ranking quantity.
    """

    if not isinstance(output_dir, Path):
        raise TypeError("global v4 plot output_dir must be a pathlib.Path")
    items = _selections_by_fold(selections, result)
    root = output_dir / "plots"
    entries: list[GlobalV4PlotManifestEntry] = [
        _quality_plot(root, items),
        _silhouette_plot(root, items),
        _cluster_size_plot(root, items),
        *_state_information_plots(root, items),
        _prefix_plot(root, items),
        *_final_grid_plots(root, items),
        _outer_nmi_plot(root, result),
        _selection_history_plot(root, items),
        _feature_frequency_plot(root, items),
        _cluster_stability_plot(root, items),
    ]
    return tuple(entries)


__all__ = ["GlobalV4PlotManifestEntry", "render_global_v4_diagnostics"]
