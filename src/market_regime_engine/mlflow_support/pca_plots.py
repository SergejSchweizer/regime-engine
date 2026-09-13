"""Deterministic PNG diagnostics for PCA fits and matched OOS scores."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from market_regime_engine.mlflow_support.pca_metrics import PCAMetricComparison
from market_regime_engine.preprocessing.two_stage import PCATwoStageScalerArtifact

if TYPE_CHECKING:
    from market_regime_engine.evaluation.walk_forward import (
        WalkForwardEvaluation,
        WalkForwardFoldResult,
    )


@dataclass(frozen=True, slots=True)
class PCAPlotManifestEntry:
    """Lineage metadata for one PCA diagnostic image."""

    png_path: str
    plot_type: str
    source_metric_keys: tuple[str, ...]
    x_axis_label: str
    y_axis_label: str
    source_artifact_hash: str

    def as_dict(self) -> dict[str, object]:
        return {
            "png_path": self.png_path,
            "plot_type": self.plot_type,
            "source_metric_keys": list(self.source_metric_keys),
            "x_axis_label": self.x_axis_label,
            "y_axis_label": self.y_axis_label,
            "source_artifact_hash": self.source_artifact_hash,
        }


def _hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _valid_pca_records(
    evaluation: WalkForwardEvaluation,
) -> tuple[tuple[WalkForwardFoldResult, PCATwoStageScalerArtifact], ...]:
    records = tuple(
        (fold, fold.pca_scaler_artifact)
        for fold in evaluation.folds
        if fold.valid and fold.pca_scaler_artifact is not None
    )
    if not records:
        raise ValueError("PCA plots require at least one valid fold-local PCA artifact")
    return records


def render_pca_diagnostics(
    evaluation: WalkForwardEvaluation,
    output_dir: str | Path,
) -> tuple[PCAPlotManifestEntry, ...]:
    """Render EVR and per-fold loading plots from frozen PCA artifacts."""

    records = _valid_pca_records(evaluation)
    root = Path(output_dir) / evaluation.candidate_id / "pca"
    root.mkdir(parents=True, exist_ok=True)
    entries: list[PCAPlotManifestEntry] = []

    evr_source = {
        str(fold.fold_id): {
            "explained_variance_ratio": list(scaler.pca_fit.artifact.explained_variance_ratio),
            "retained_component_count": scaler.pca_fit.artifact.retained_component_count,
        }
        for fold, scaler in records
    }
    evr_path = root / "explained_variance.png"
    figure, axis = plt.subplots(figsize=(10.0, 5.5))
    try:
        for fold, scaler in records:
            artifact = scaler.pca_fit.artifact
            curve = np.cumsum(np.asarray(artifact.explained_variance_ratio, dtype=np.float64))
            axis.plot(
                np.arange(1, len(curve) + 1),
                curve,
                marker="o",
                label=fold.fold_id,
            )
            axis.axvline(artifact.retained_component_count, linestyle="--", alpha=0.35)
        axis.set_title("Fold-local PCA cumulative explained variance")
        axis.set_xlabel("Component")
        axis.set_ylabel("Cumulative explained variance")
        axis.set_ylim(0.0, 1.02)
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(evr_path, dpi=180, format="png")
    finally:
        plt.close(figure)
    entries.append(
        PCAPlotManifestEntry(
            png_path=str(evr_path),
            plot_type="pca_explained_variance",
            source_metric_keys=(
                "pca_explained_variance_ratio_component_*",
                "pca_retained_component_count",
            ),
            x_axis_label="Component",
            y_axis_label="Cumulative explained variance",
            source_artifact_hash=_hash(evr_source),
        )
    )

    for fold, scaler in records:
        artifact = scaler.pca_fit.artifact
        source = {
            "fold_id": fold.fold_id,
            "feature_order": list(artifact.feature_order),
            "components": [list(component) for component in artifact.components],
        }
        path = root / "loadings" / f"{fold.fold_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(10.0, 5.5))
        try:
            image = axis.imshow(np.asarray(artifact.components, dtype=np.float64), aspect="auto")
            axis.set_title(f"PCA loadings — {fold.fold_id}")
            axis.set_xlabel("Raw feature")
            axis.set_ylabel("Principal component")
            axis.set_xticks(
                np.arange(len(artifact.feature_order)),
                artifact.feature_order,
                rotation=45,
            )
            axis.set_yticks(
                np.arange(artifact.retained_component_count),
                artifact.generated_feature_names,
            )
            figure.colorbar(image, ax=axis, label="Loading")
            figure.tight_layout()
            figure.savefig(path, dpi=180, format="png")
        finally:
            plt.close(figure)
        entries.append(
            PCAPlotManifestEntry(
                png_path=str(path),
                plot_type="pca_loadings",
                source_metric_keys=("pca_loading_component_*_feature_*",),
                x_axis_label="Raw feature",
                y_axis_label="Principal component",
                source_artifact_hash=_hash(source),
            )
        )
    return tuple(entries)


def render_pca_oos_comparison(
    comparison: PCAMetricComparison,
    output_dir: str | Path,
) -> PCAPlotManifestEntry:
    """Render matched raw-only versus raw-plus-PCA OOS scores."""

    raw_scores = tuple(
        fold.oos_predictive_log_likelihood_per_observation
        for fold in comparison.raw_only.folds
        if fold.valid
    )
    pca_scores = tuple(
        fold.oos_predictive_log_likelihood_per_observation
        for fold in comparison.raw_plus_pca.folds
        if fold.valid
    )
    if any(value is None for value in (*raw_scores, *pca_scores)):
        raise ValueError("PCA OOS plot requires finite matched fold scores")
    raw_values = tuple(float(value) for value in raw_scores if value is not None)
    pca_values = tuple(float(value) for value in pca_scores if value is not None)
    if len(raw_values) != len(pca_values):
        raise ValueError("PCA OOS plot requires equal matched score counts")
    root = Path(output_dir) / comparison.raw_plus_pca.candidate_id / "pca"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "oos_comparison.png"
    figure, axis = plt.subplots(figsize=(10.0, 5.5))
    try:
        x_values = np.arange(1, len(raw_values) + 1)
        axis.plot(x_values, raw_values, marker="o", label="Raw-only")
        axis.plot(x_values, pca_values, marker="s", label="Raw + PCA")
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title("Matched OOS predictive score: raw-only vs raw + PCA")
        axis.set_xlabel("Walk-forward fold")
        axis.set_ylabel("OOS predictive log likelihood / observation")
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(path, dpi=180, format="png")
    finally:
        plt.close(figure)
    source = {
        "folds": list(comparison.matched_fold_indices),
        "raw_only": raw_values,
        "raw_plus_pca": pca_values,
    }
    return PCAPlotManifestEntry(
        png_path=str(path),
        plot_type="pca_oos_comparison",
        source_metric_keys=(
            "pca_comparison_raw_oos_predictive_loglik_per_obs",
            "pca_comparison_augmented_oos_predictive_loglik_per_obs",
        ),
        x_axis_label="Walk-forward fold",
        y_axis_label="OOS predictive log likelihood / observation",
        source_artifact_hash=_hash(source),
    )


__all__ = [
    "PCAPlotManifestEntry",
    "render_pca_diagnostics",
    "render_pca_oos_comparison",
]
