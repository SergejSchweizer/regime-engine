"""Human-auditable MLflow evidence for the frozen Xetra feature-selection path."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from market_regime_engine.feature_selection.contracts import FeatureSelectionResult
from market_regime_engine.feature_selection.stability import FeatureSelectionStabilityDiagnostics
from market_regime_engine.mlflow_support.ports import TrackingPort

_DPI = 180
_FIGSIZE_WIDE = (10.0, 6.0)
_FIGSIZE_SQUARE = (7.5, 7.0)


@dataclass(frozen=True, slots=True)
class FeatureSelectionArtifact:
    artifact_type: str
    png_path: str | None
    svg_path: str | None
    title: str
    x_axis_label: str
    y_axis_label: str
    legend_entries: tuple[str, ...]
    source_sha256: str


@dataclass(frozen=True, slots=True)
class FeatureSelectionTrackingResult:
    summary_path: str
    evidence_path: str
    diagnostics_path: str
    manifest_path: str
    artifacts: tuple[FeatureSelectionArtifact, ...]


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256(payload).hexdigest()


def _save_figure(fig: Any, stem: Path) -> tuple[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    fig.tight_layout()
    fig.savefig(png, dpi=_DPI, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return str(png), str(svg)


def _render_stage1_scores(result: FeatureSelectionResult, root: Path) -> FeatureSelectionArtifact:
    labels: list[str] = []
    values: list[float] = []
    for block in result.evidence.block_evidence:
        for score in block.scores:
            if score.eligible and score.medoid_score is not None:
                labels.append(f"{block.block_id}:{score.feature_name}")
                values.append(float(score.medoid_score))
    if not values:
        raise ValueError("Stage-1 audit requires at least one eligible medoid score")
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    positions = np.arange(len(values))
    ax.bar(positions, values)
    ax.set_title("Stage-1 absolute-Spearman medoid scores")
    ax.set_xlabel("Semantic block and feature")
    ax.set_ylabel("Mean distance 1 - |Spearman rho|")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.grid(axis="y", alpha=0.25)
    png, svg = _save_figure(fig, root / "stage1_medoid_scores")
    source_hash = _sha256_json(
        [
            {
                "block_id": block.block_id,
                "scores": [asdict(score) for score in block.scores],
                "winner": block.winner,
            }
            for block in result.evidence.block_evidence
        ]
    )
    return FeatureSelectionArtifact(
        artifact_type="stage1_medoid_scores",
        png_path=png,
        svg_path=svg,
        title="Stage-1 absolute-Spearman medoid scores",
        x_axis_label="Semantic block and feature",
        y_axis_label="Mean distance 1 - |Spearman rho|",
        legend_entries=(),
        source_sha256=source_hash,
    )


def _render_heatmap(
    matrix: np.ndarray,
    labels: tuple[str, ...],
    *,
    title: str,
    colorbar_label: str,
    stem: Path,
    artifact_type: str,
) -> FeatureSelectionArtifact:
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("heatmap matrix shape must match feature labels")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("heatmap matrix must be finite")
    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.set_yticklabels(labels)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    if len(labels) <= 8:
        for row in range(len(labels)):
            for column in range(len(labels)):
                ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    png, svg = _save_figure(fig, stem)
    return FeatureSelectionArtifact(
        artifact_type=artifact_type,
        png_path=png,
        svg_path=svg,
        title=title,
        x_axis_label="Feature",
        y_axis_label="Feature",
        legend_entries=(),
        source_sha256=_sha256_json({"labels": labels, "matrix": matrix.tolist()}),
    )


def _summary(result: FeatureSelectionResult) -> str:
    lines = [
        "# Feature selection summary",
        "",
        "Path: `48 source features -> 8 Stage-1 medoids -> d <= 8 frozen features`.",
        "",
        f"Definition hash: `{result.feature_selection_definition_hash}`",
        f"Execution hash: `{result.feature_selection_execution_hash}`",
        "",
        "## Stage 1 winners",
    ]
    for block in result.evidence.block_evidence:
        lines.append(f"- `{block.block_id}` -> `{block.winner}`")
    lines.extend(["", "## Stage 2 pruning"])
    if result.evidence.conflicts:
        for conflict in result.evidence.conflicts:
            lines.append(
                "- "
                f"`{conflict.feature_a}` vs `{conflict.feature_b}`: "
                f"|rho|={conflict.abs_spearman:.6f}; removed "
                f"`{conflict.removed_feature}` ({conflict.removal_reason})"
            )
    else:
        lines.append("- No cross-block pair exceeded the strict 0.85 conflict threshold.")
    lines.extend(["", "## Frozen final features"])
    lines.extend(f"- `{feature}`" for feature in result.final_features)
    lines.append("")
    return "\n".join(lines)


def track_feature_selection_evidence(
    port: TrackingPort,
    *,
    parent_run_id: str,
    result: FeatureSelectionResult,
    diagnostics: FeatureSelectionStabilityDiagnostics,
    within_block_abs_spearman: dict[str, tuple[tuple[float, ...], ...]],
    artifact_root: str | Path,
) -> FeatureSelectionTrackingResult:
    """Persist complete non-decision feature-selection evidence under one parent run."""

    if diagnostics.frozen_definition_hash != result.feature_selection_definition_hash:
        raise ValueError("diagnostics definition hash differs from frozen selection")
    if diagnostics.frozen_execution_hash != result.feature_selection_execution_hash:
        raise ValueError("diagnostics execution hash differs from frozen selection")
    if diagnostics.frozen_final_features != result.final_features:
        raise ValueError("diagnostics frozen features differ from selection result")
    root = Path(artifact_root) / "feature_selection"
    root.mkdir(parents=True, exist_ok=True)

    evidence_path = root / "selection_evidence.json"
    diagnostics_path = root / "diagnostics" / "stability.json"
    summary_path = root / "selection_summary.md"
    evidence_path.write_text(
        json.dumps(asdict(result), sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(asdict(diagnostics), sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(_summary(result), encoding="utf-8")

    artifacts: list[FeatureSelectionArtifact] = [_render_stage1_scores(result, root / "plots")]
    expected_blocks = tuple(block.block_id for block in result.evidence.block_evidence)
    if tuple(within_block_abs_spearman) != expected_blocks:
        raise ValueError("within-block matrices must preserve exact canonical block order")
    for block in result.evidence.block_evidence:
        matrix = np.asarray(within_block_abs_spearman[block.block_id], dtype=np.float64)
        labels = tuple(score.feature_name for score in block.scores if score.eligible)
        artifacts.append(
            _render_heatmap(
                matrix,
                labels,
                title=f"Stage-1 within-block absolute Spearman — {block.block_id}",
                colorbar_label="Absolute Spearman correlation",
                stem=root / "plots" / f"within_block_{block.block_id}",
                artifact_type="stage1_within_block_abs_spearman",
            )
        )
    artifacts.append(
        _render_heatmap(
            np.asarray(result.evidence.stage2_abs_spearman_matrix, dtype=np.float64),
            result.evidence.preliminary_medoids,
            title="Stage-2 cross-block absolute Spearman — preliminary medoids",
            colorbar_label="Absolute Spearman correlation",
            stem=root / "plots" / "stage2_cross_block_abs_spearman",
            artifact_type="stage2_cross_block_abs_spearman",
        )
    )

    manifest_path = root / "plots" / "manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(item) for item in artifacts], sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    for path in (evidence_path, diagnostics_path, summary_path):
        port.log_artifact(parent_run_id, str(path), "feature_selection")
    for artifact in artifacts:
        if artifact.png_path is not None:
            port.log_artifact(parent_run_id, artifact.png_path, "feature_selection/plots")
        if artifact.svg_path is not None:
            port.log_artifact(parent_run_id, artifact.svg_path, "feature_selection/plots")
    port.log_artifact(parent_run_id, str(manifest_path), "feature_selection/plots")
    return FeatureSelectionTrackingResult(
        summary_path=str(summary_path),
        evidence_path=str(evidence_path),
        diagnostics_path=str(diagnostics_path),
        manifest_path=str(manifest_path),
        artifacts=tuple(artifacts),
    )
