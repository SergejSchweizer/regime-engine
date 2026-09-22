"""Deterministic MLflow evidence for TRAIN-only feature preprocessing."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np

matplotlib.use("Agg")
import duckdb
import matplotlib.pyplot as plt

from market_regime_engine.feature_discovery.family_reduction import _absolute_pearson
from market_regime_engine.feature_discovery.feature_roles import FeatureRoleContract
from market_regime_engine.feature_discovery.metadata_store import FeatureSelectionMetadataStore
from market_regime_engine.feature_discovery.pipeline import FeatureSelectionPipelineResult
from market_regime_engine.mlflow_support.ports import TrackingPort


@dataclass(frozen=True, slots=True)
class FeatureSelectionEvidenceIdentity:
    profile_hash: str
    source_build_id: str
    fold_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "profile_hash": self.profile_hash,
            "source_build_id": self.source_build_id,
            "fold_id": self.fold_id,
        }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _plot(path: Path, title: str, labels: tuple[str, ...], values: tuple[int, ...]) -> None:
    figure, axis = plt.subplots(figsize=(10.0, 5.5))
    try:
        axis.bar(labels, values)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        figure.savefig(path, dpi=180, format="png")
    finally:
        plt.close(figure)


def _family_survival(
    result: FeatureSelectionPipelineResult,
    contract: FeatureRoleContract,
    discovered: tuple[str, ...],
) -> tuple[dict[str, object], ...]:
    quality = set(result.quality_eligible_features)
    retained = set(result.family_reduction.retained_features)
    representatives = set(result.global_reduction.representatives)
    families = sorted(
        {
            family
            for name in discovered
            for family in (contract.assignment(name).family,)
            if family is not None
        }
    )
    return tuple(
        {
            "family": family,
            "source_count": sum(contract.assignment(name).family == family for name in discovered),
            "quality_count": sum(
                contract.assignment(name).family == family and name in quality
                for name in discovered
            ),
            "retained_after_near_duplicate_count": sum(
                contract.assignment(name).family == family and name in retained for name in retained
            ),
            "representative_count": sum(
                contract.assignment(name).family == family and name in representatives
                for name in representatives
            ),
        }
        for family in families
    )


def render_feature_selection_evidence(
    result: FeatureSelectionPipelineResult,
    contract: FeatureRoleContract,
    *,
    discovered_feature_names: tuple[str, ...],
    source_build_id: str,
    fold_id: str,
    output_dir: str | Path,
    feature_values: Mapping[str, Sequence[float | None]] | None = None,
) -> tuple[Path, ...]:
    """Render deterministic preprocessing plots and complete companion tables."""

    identity = FeatureSelectionEvidenceIdentity(result.profile_hash, source_build_id, fold_id)
    root = Path(output_dir) / "feature_selection"
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    funnel_counts: dict[str, int] = {
        "discovered": len(discovered_feature_names),
        "quality_eligible": len(result.quality_eligible_features),
        "family_pc_core": len(result.global_reduction.representatives),
        "correlation_representatives": len(result.global_reduction.representatives),
        "final_selected": len(result.selected_features),
    }
    funnel = {
        **identity.as_dict(),
        "counts": funnel_counts,
    }
    funnel_json = root / "feature_funnel.json"
    _write_json(funnel_json, funnel)
    paths.append(funnel_json)
    funnel_png = root / "feature_funnel.png"
    _plot(
        funnel_png,
        "Feature-selection funnel",
        tuple(funnel_counts),
        tuple(funnel_counts.values()),
    )
    paths.append(funnel_png)

    survival = _family_survival(result, contract, discovered_feature_names)
    survival_json = root / "family_survival.json"
    _write_json(survival_json, {"identity": identity.as_dict(), "families": survival})
    paths.append(survival_json)
    survival_png = root / "family_survival.png"
    _plot(
        survival_png,
        "Family representative survival",
        tuple(str(item["family"]) for item in survival),
        tuple(cast(int, item["representative_count"]) for item in survival),
    )
    paths.append(survival_png)

    for artifact in sorted(result.family_pca, key=lambda item: item.family):
        family_root = root / "pca" / artifact.family
        family_root.mkdir(parents=True, exist_ok=True)
        curve = family_root / "explained_variance.png"
        figure, axis = plt.subplots(figsize=(10.0, 5.5))
        try:
            axis.plot(
                np.arange(1, len(artifact.explained_variance_ratio) + 1),
                np.cumsum(artifact.explained_variance_ratio),
                marker="o",
            )
            axis.set_title(f"PCA explained variance — {artifact.family}")
            axis.set_xlabel("Component")
            axis.set_ylabel("Cumulative explained variance")
            figure.tight_layout()
            figure.savefig(curve, dpi=180, format="png")
        finally:
            plt.close(figure)
        paths.append(curve)
        loading_table = family_root / "loadings.json"
        _write_json(
            loading_table,
            {
                "identity": identity.as_dict(),
                "family": artifact.family,
                "loadings": [
                    asdict(item) for item in artifact.pca_loadings(fold_id, source_build_id)
                ],
            },
        )
        paths.append(loading_table)
        retained_pcs = set(result.global_reduction.representatives)
        for component_index, component in enumerate(artifact.components, start=1):
            pc_name = artifact.generated_feature_names[component_index - 1]
            if pc_name not in retained_pcs:
                continue
            ranked = sorted(
                zip(artifact.feature_order, component, strict=True),
                key=lambda item: (-abs(item[1]), item[0]),
            )[:20]
            loading_plot = family_root / "loadings" / f"{pc_name}.png"
            figure, axis = plt.subplots(figsize=(10.0, 5.5))
            try:
                axis.bar(tuple(item[0] for item in ranked), tuple(item[1] for item in ranked))
                axis.set_title(f"Top PCA loadings — {pc_name}")
                axis.tick_params(axis="x", rotation=60)
                figure.tight_layout()
                figure.savefig(loading_plot, dpi=180, format="png")
            finally:
                plt.close(figure)
            paths.append(loading_plot)

    groups = Counter(item.leader for item in result.global_reduction.evidence)
    group_table = root / "correlation_groups.json"
    _write_json(
        group_table, {"identity": identity.as_dict(), "groups": dict(sorted(groups.items()))}
    )
    paths.append(group_table)
    group_png = root / "correlation_group_sizes.png"
    ordered_groups = tuple(sorted(groups.items(), key=lambda item: (-item[1], item[0])))
    _plot(
        group_png,
        "Correlation group sizes",
        tuple(item[0] for item in ordered_groups),
        tuple(item[1] for item in ordered_groups),
    )
    paths.append(group_png)
    selected = tuple(item[0] for item in ordered_groups[:80])
    heatmap = root / "correlation_representatives_heatmap.png"
    matrix = np.zeros((len(selected), len(selected)), dtype=np.float64)
    for index, name in enumerate(selected):
        matrix[index, index] = groups[name]
    if feature_values is not None:
        for left_index, left_name in enumerate(selected):
            for right_index, right_name in enumerate(selected[:left_index]):
                correlation = _absolute_pearson(
                    feature_values[left_name], feature_values[right_name]
                )
                value = 0.0 if correlation is None else correlation[0]
                matrix[left_index, right_index] = value
                matrix[right_index, left_index] = value
    figure, axis = plt.subplots(figsize=(8.0, 7.0))
    try:
        image = axis.imshow(matrix, aspect="auto")
        axis.set_title("Correlation representatives and covered-group size")
        axis.set_xticks(np.arange(len(selected)), selected, rotation=90)
        axis.set_yticks(np.arange(len(selected)), selected)
        figure.colorbar(image, ax=axis, label="Covered group size")
        figure.tight_layout()
        figure.savefig(heatmap, dpi=180, format="png")
    finally:
        plt.close(figure)
    paths.append(heatmap)
    artifact_records = [
        {
            "path": str(path.relative_to(root)),
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "identity": identity.as_dict(),
        }
        for path in paths
    ]
    manifest = root / "manifest.json"
    _write_json(
        manifest,
        {
            "identity": identity.as_dict(),
            "artifacts": artifact_records,
            "heatmap_representatives": list(selected),
        },
    )
    paths.append(manifest)
    return tuple(paths)


def verify_feature_selection_evidence_bundle(root_dir: str | Path) -> dict[str, object]:
    """Fail closed when a required preprocessing artifact or hash is missing."""

    root = Path(root_dir) / "feature_selection"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("feature-selection evidence manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("feature-selection evidence manifest has no artifacts")
    required = {
        "feature_funnel.png",
        "feature_funnel.json",
        "family_survival.png",
        "family_survival.json",
        "correlation_groups.json",
        "correlation_group_sizes.png",
        "correlation_representatives_heatmap.png",
    }
    paths: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("invalid feature-selection artifact record")
        relative = item["path"]
        path = root / relative
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != item.get("sha256"):
            raise ValueError(f"feature-selection artifact hash mismatch: {relative}")
        paths.add(relative)
        if item.get("identity") != manifest.get("identity"):
            raise ValueError(f"feature-selection artifact identity mismatch: {relative}")
    if not required.issubset({Path(path).name for path in paths}):
        raise ValueError("required feature-selection artifact is missing")
    selected = manifest.get("heatmap_representatives")
    if not isinstance(selected, list) or len(selected) > 80 or len(selected) != len(set(selected)):
        raise ValueError("invalid representative heatmap selection")
    return cast(dict[str, object], manifest)


def log_feature_selection_evidence(
    port: TrackingPort,
    run_id: str,
    paths: tuple[Path, ...],
    *,
    artifact_path: str = "feature_selection",
) -> bool:
    """Upload a completed local bundle without making selection depend on MLflow."""

    try:
        for path in paths:
            port.log_artifact(run_id, str(path), artifact_path)
    except Exception:
        return False
    return True


def render_cumulative_feature_stats(
    store: FeatureSelectionMetadataStore,
    *,
    fold_id: str,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Render the committed cumulative feature-statistics projection.

    The DuckDB view is the sole source for these plots; no in-memory fold
    accumulator is consulted.  Replaying a fold therefore produces the same
    artifacts and cannot double-count a contribution.
    """

    root = Path(output_dir) / "feature_selection" / "cumulative"
    root.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(store.database), read_only=True) as connection:
        rows = connection.execute(
            """
            SELECT feature_name, selected_folds, selection_rate,
                   mean_ablation_loss, total_pca_credit
            FROM feature_global_stats
            ORDER BY feature_name
            """
        ).fetchall()
    payload = {
        "fold_id": fold_id,
        "features": [
            {
                "feature_name": name,
                "selected_folds": selected,
                "selection_rate": rate,
                "mean_ablation_loss": mean_loss,
                "total_pca_credit": credit,
            }
            for name, selected, rate, mean_loss, credit in rows
        ],
    }
    table = root / "feature_global_stats.json"
    _write_json(table, payload)
    paths = [table]
    chart_specs = (
        ("selection_frequency.png", "Cumulative feature selection rate", 2),
        ("mean_ablation_loss.png", "Cumulative mean ablation loss", 3),
        ("pca_credit.png", "Cumulative PCA credit", 4),
    )
    labels = tuple(str(row[0]) for row in rows)
    for filename, title, index in chart_specs:
        figure, axis = plt.subplots(figsize=(12.0, 6.0))
        try:
            values = tuple(float(row[index] or 0.0) for row in rows)
            axis.bar(labels, values)
            axis.set_title(f"{title} — through {fold_id}")
            axis.tick_params(axis="x", rotation=60)
            figure.tight_layout()
            path = root / filename
            figure.savefig(path, dpi=180, format="png")
        finally:
            plt.close(figure)
        paths.append(path)
    return tuple(paths)


def log_cumulative_feature_stats(
    port: TrackingPort,
    run_id: str,
    store: FeatureSelectionMetadataStore,
    *,
    fold_id: str,
    output_dir: str | Path,
    artifact_path: str = "feature_selection/cumulative",
) -> bool:
    """Project committed cumulative stats to MLflow after a completed fold."""

    try:
        for path in render_cumulative_feature_stats(store, fold_id=fold_id, output_dir=output_dir):
            port.log_artifact(run_id, str(path), artifact_path)
    except Exception:
        return False
    return True


__all__ = [
    "FeatureSelectionEvidenceIdentity",
    "log_cumulative_feature_stats",
    "log_feature_selection_evidence",
    "render_cumulative_feature_stats",
    "render_feature_selection_evidence",
    "verify_feature_selection_evidence_bundle",
]
