"""Canonical, deterministic evidence artifacts for feature-selection runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from market_regime_engine.feature_discovery.pipeline import FeatureSelectionPipelineResult


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_plot(
    path: Path, title: str, x: tuple[float, ...], y: tuple[float, ...], label: str
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    if x and y:
        axis.plot(x, y, marker="o", linewidth=1.5)
    axis.set_title(title)
    axis.set_xlabel("ordinal")
    axis.set_ylabel(label)
    axis.grid(True, alpha=0.25)
    figure.savefig(path, dpi=150, format="png", metadata={"Software": "regime-engine"})
    plt.close(figure)


def write_canonical_diagnostics(
    pipeline: FeatureSelectionPipelineResult, directory: str | Path
) -> tuple[Path, ...]:
    """Materialize all fold diagnostics from the immutable selection result.

    The returned files are suitable for direct tracking as MLflow artifacts.  All
    values come from the fold-local result; no network, database, or global state
    is consulted.
    """

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    family_rows = tuple(
        {
            "family": item.family,
            "feature_count": len(item.feature_order),
            "numerical_rank": item.numerical_rank,
            "explained_variance_ratio": item.explained_variance_ratio,
            "cumulative_explained_variance": item.cumulative_explained_variance,
            "fit_hash": item.fit_hash,
        }
        for item in pipeline.family_pca
    )
    funnel = {
        "quality_eligible": len(pipeline.quality_eligible_features),
        "family_retained": len(pipeline.family_reduction.retained_features),
        "family_pca_components": sum(item.numerical_rank for item in pipeline.family_pca),
        "global_representatives": len(pipeline.global_reduction.representatives),
        "sffs_selected": len(pipeline.selected_features),
        "ablation_evaluations": len(pipeline.ablation.one_feature_results),
    }
    correlation = {
        "representatives": pipeline.global_reduction.representatives,
        "removed_features": pipeline.global_reduction.removed_features,
        "evidence": tuple(
            {
                "leader": item.leader,
                "removed": item.removed,
                "full_absolute_pearson": item.full_absolute_pearson,
                "full_support_count": item.full_support_count,
                "subwindow_absolute_pearsons": item.subwindow_absolute_pearsons,
                "subwindow_support_counts": item.subwindow_support_counts,
            }
            for item in pipeline.global_reduction.evidence
        ),
    }
    sffs = {
        "selected_features": pipeline.sffs.selected_features,
        "best_singleton": pipeline.sffs.best_singleton,
        "steps": tuple(
            {
                "action": item.action,
                "selected_features": item.selected_features,
                "score": item.score.value,
            }
            for item in pipeline.sffs.steps
        ),
    }
    baseline_score = pipeline.ablation.baseline.hmm_evaluation.score
    if baseline_score is None:
        raise ValueError("ablation baseline diagnostics require a valid score")
    ablation = {
        "selected_features": pipeline.ablation.selected_features,
        "baseline_score": baseline_score.value,
        "losses": pipeline.ablation.ablation_losses,
        "removed_features": tuple(
            item.removed_feature for item in pipeline.ablation.one_feature_results
        ),
    }

    json_files = (
        ("feature-funnel.json", funnel),
        ("family-pca.json", family_rows),
        ("correlation-reduction.json", correlation),
        ("sffs-steps.json", sffs),
        ("ablation-losses.json", ablation),
    )
    paths: list[Path] = []
    for name, payload in json_files:
        path = root / name
        _write_json(path, payload)
        paths.append(path)

    explained = tuple(
        sum(item.explained_variance_ratio[: ordinal])
        for item in pipeline.family_pca
        for ordinal in range(1, item.numerical_rank + 1)
    )
    paths_to_plot = (
        ("feature-funnel.png", tuple(range(len(funnel))), tuple(funnel.values()), "count"),
        (
            "pca-explained-variance.png",
            tuple(range(1, len(explained) + 1)),
            explained,
            "cumulative variance",
        ),
        (
            "correlation-reduction.png",
            (0.0, 1.0),
            (
                float(len(correlation["representatives"])),
                float(len(correlation["removed_features"])),
            ),
            "feature count",
        ),
        (
            "sffs-scores.png",
            tuple(range(1, len(pipeline.sffs.steps) + 1)),
            tuple(item.score.value for item in pipeline.sffs.steps),
            "score",
        ),
        (
            "ablation-losses.png",
            tuple(range(1, len(pipeline.ablation.ablation_losses) + 1)),
            tuple(float(value or 0.0) for value in pipeline.ablation.ablation_losses),
            "loss",
        ),
    )
    for name, x, y, label in paths_to_plot:
        path = root / name
        _write_plot(path, name.removesuffix(".png").replace("-", " "), x, y, label)
        paths.append(path)
    return tuple(paths)


__all__ = ["write_canonical_diagnostics"]
