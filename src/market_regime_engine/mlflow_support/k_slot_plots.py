"""Model-Metrics-only per-K comparison plot payloads and rendering.

The builders accept K-slot projections, never evaluation objects or source
rows.  This makes a plot a reproducible view of already published Model
Metrics and keeps rendering independent from HMM fitting.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from market_regime_engine.evaluations.process_parallel import cpu_process_pool
from market_regime_engine.mlflow_support.k_slot_metrics import (
    K_CROSS_COMPARISON_DOMAIN_ID,
    LEGAL_K,
    KSlotMetricProjection,
    validate_k_metric_comparison,
)
from market_regime_engine.mlflow_support.metric_catalog import METRIC_CATALOG_VERSION
from market_regime_engine.mlflow_support.ports import MetricPoint
from market_regime_engine.runtime.cpu import cpu_worker_count


@dataclass(frozen=True, slots=True)
class KPlotSeries:
    logged_model_id: str
    model_family: str
    metric_key: str
    points: tuple[MetricPoint, ...]


@dataclass(frozen=True, slots=True)
class KPlotPayload:
    """Canonical payload for one per-K metric plot."""

    plot_id: str
    slot_id: str
    state_count: int
    metric_key: str
    status: str
    series: tuple[KPlotSeries, ...]
    logged_model_ids: tuple[str, ...]
    feature_order_hashes: tuple[str, ...]
    source_data_hashes: tuple[str, ...]
    metric_catalog_version: int
    comparison_domain_id: str
    no_evaluation_recomputation: bool = True
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"available", "not_available"}:
            raise ValueError("K plot status must be available or not_available")
        if self.status == "available" and not self.series:
            raise ValueError("available K plot must contain metric series")
        if self.status == "not_available" and not self.unavailable_reason:
            raise ValueError("unavailable K plot requires a reason")
        if self.metric_catalog_version != METRIC_CATALOG_VERSION:
            raise ValueError("K plot metric catalog version is not current")
        if not self.no_evaluation_recomputation:
            raise ValueError("K plots must declare that no evaluation recomputation occurred")

    @property
    def canonical_json(self) -> str:
        payload = {
            "plot_id": self.plot_id,
            "slot_id": self.slot_id,
            "state_count": self.state_count,
            "metric_key": self.metric_key,
            "status": self.status,
            "series": [
                {
                    "logged_model_id": item.logged_model_id,
                    "model_family": item.model_family,
                    "metric_key": item.metric_key,
                    "points": [asdict(point) for point in item.points],
                }
                for item in self.series
            ],
            "logged_model_ids": self.logged_model_ids,
            "feature_order_hashes": self.feature_order_hashes,
            "source_data_hashes": self.source_data_hashes,
            "metric_catalog_version": self.metric_catalog_version,
            "comparison_domain_id": self.comparison_domain_id,
            "no_evaluation_recomputation": self.no_evaluation_recomputation,
            "unavailable_reason": self.unavailable_reason,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @property
    def canonical_payload_hash(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def manifest(self, *, png_path: str | None = None) -> dict[str, object]:
        """Return the immutable provenance manifest for this plot payload."""

        return {
            "plot_id": self.plot_id,
            "slot_id": self.slot_id,
            "state_count": self.state_count,
            "metric_key": self.metric_key,
            "status": self.status,
            "png_path": png_path,
            "logged_model_ids": list(self.logged_model_ids),
            "feature_order_hashes": list(self.feature_order_hashes),
            "source_data_hashes": list(self.source_data_hashes),
            "metric_catalog_version": self.metric_catalog_version,
            "comparison_domain_id": self.comparison_domain_id,
            "canonical_payload_hash": self.canonical_payload_hash,
            "no_evaluation_recomputation": self.no_evaluation_recomputation,
            "unavailable_reason": self.unavailable_reason,
        }


def _unavailable(slot: KSlotMetricProjection, metric_key: str, reason: str) -> KPlotPayload:
    return KPlotPayload(
        plot_id=f"{slot.slot_id}:{metric_key}",
        slot_id=slot.slot_id,
        state_count=slot.metadata.state_count,
        metric_key=metric_key,
        status="not_available",
        series=(),
        logged_model_ids=tuple(item.logged_model_id for item in slot.candidates),
        feature_order_hashes=(slot.metadata.feature_order_sha256,),
        source_data_hashes=(slot.metadata.source_data_sha256,),
        metric_catalog_version=METRIC_CATALOG_VERSION,
        comparison_domain_id=slot.metadata.comparison_domain_id,
        unavailable_reason=reason,
    )


def build_k_plot_payload(
    slot: KSlotMetricProjection,
    metric_key: str,
) -> KPlotPayload:
    """Build one compatible three-family plot for a single K."""

    if not metric_key or not isinstance(metric_key, str):
        raise ValueError("metric_key must be a non-empty string")
    if not slot.eligible:
        return _unavailable(
            slot,
            metric_key,
            slot.unavailable_reason or "K slot is ineligible; no selected plot is available",
        )
    points = {item.logged_model_id: item.metric_points for item in slot.candidates}
    tags = {item.logged_model_id: item.tags for item in slot.candidates}
    try:
        validate_k_metric_comparison(
            metric_key,
            {
                model_id: tuple(point for point in values if point.key == metric_key)
                for model_id, values in points.items()
            },
            tags,
            state_count=slot.metadata.state_count,
        )
    except ValueError as exc:
        return _unavailable(slot, metric_key, str(exc))
    by_id = {item.logged_model_id: item for item in slot.candidates}
    series = tuple(
        KPlotSeries(
            logged_model_id=model_id,
            model_family=by_id[model_id].model_family,
            metric_key=metric_key,
            points=tuple(
                point for point in by_id[model_id].metric_points if point.key == metric_key
            ),
        )
        for model_id in sorted(by_id)
    )
    if any(not item.points for item in series):
        return _unavailable(slot, metric_key, "one K family is missing the requested metric")
    return KPlotPayload(
        plot_id=f"{slot.slot_id}:{metric_key}",
        slot_id=slot.slot_id,
        state_count=slot.metadata.state_count,
        metric_key=metric_key,
        status="available",
        series=series,
        logged_model_ids=tuple(item.logged_model_id for item in series),
        feature_order_hashes=(slot.metadata.feature_order_sha256,),
        source_data_hashes=(slot.metadata.source_data_sha256,),
        metric_catalog_version=METRIC_CATALOG_VERSION,
        comparison_domain_id=slot.metadata.comparison_domain_id,
    )


def _build_k_plot_task(task: tuple[KSlotMetricProjection, str]) -> KPlotPayload:
    return build_k_plot_payload(*task)


def prepare_k_plot_payloads(
    slots: Sequence[KSlotMetricProjection],
    metric_keys: Sequence[str],
    *,
    max_workers: int | None = None,
) -> tuple[KPlotPayload, ...]:
    """Prepare independent K/metric payloads with canonical result ordering."""

    tasks = tuple(
        (slot, metric_key)
        for slot in sorted(slots, key=lambda item: item.metadata.state_count)
        for metric_key in sorted(set(metric_keys))
    )
    if not tasks:
        raise ValueError("K plot preparation requires at least one slot and metric")
    workers = cpu_worker_count(max_workers, task_count=len(tasks))
    if workers == 1:
        return tuple(_build_k_plot_task(task) for task in tasks)
    with cpu_process_pool(workers) as executor:
        futures = [executor.submit(_build_k_plot_task, task) for task in tasks]
        return tuple(future.result() for future in futures)


def build_cross_k_plot_payload(
    slots: Sequence[KSlotMetricProjection],
    metric_key: str,
) -> KPlotPayload:
    """Build a cross-K plot only for an explicitly dimension-independent metric."""

    ordered = tuple(sorted(slots, key=lambda item: item.metadata.state_count))
    if tuple(item.metadata.state_count for item in ordered) != LEGAL_K:
        raise ValueError("cross-K plots require all four K slots")
    if any(not item.eligible for item in ordered):
        return KPlotPayload(
            plot_id=f"cross-k:{metric_key}",
            slot_id="cross-k",
            state_count=0,
            metric_key=metric_key,
            status="not_available",
            series=(),
            logged_model_ids=(),
            feature_order_hashes=tuple(item.metadata.feature_order_sha256 for item in ordered),
            source_data_hashes=tuple(item.metadata.source_data_sha256 for item in ordered),
            metric_catalog_version=METRIC_CATALOG_VERSION,
            comparison_domain_id=K_CROSS_COMPARISON_DOMAIN_ID,
            unavailable_reason="at least one K slot is ineligible",
        )
    series: list[KPlotSeries] = []
    for slot in ordered:
        selected = {slot.selected_logged_model_id: slot.selected_metric_points}
        tags = {slot.selected_logged_model_id or "": slot.metadata.tags(scope="selected")}
        try:
            validate_k_metric_comparison(
                metric_key,
                {
                    model_id: tuple(point for point in points if point.key == metric_key)
                    for model_id, points in selected.items()
                    if model_id
                },
                tags,
                state_count=slot.metadata.state_count,
                cross_k=True,
            )
        except ValueError as exc:
            return KPlotPayload(
                plot_id=f"cross-k:{metric_key}",
                slot_id="cross-k",
                state_count=0,
                metric_key=metric_key,
                status="not_available",
                series=(),
                logged_model_ids=(),
                feature_order_hashes=tuple(item.metadata.feature_order_sha256 for item in ordered),
                source_data_hashes=tuple(item.metadata.source_data_sha256 for item in ordered),
                metric_catalog_version=METRIC_CATALOG_VERSION,
                comparison_domain_id=K_CROSS_COMPARISON_DOMAIN_ID,
                unavailable_reason=str(exc),
            )
        series.append(
            KPlotSeries(
                logged_model_id=slot.selected_logged_model_id or "",
                model_family=f"K={slot.metadata.state_count}",
                metric_key=metric_key,
                points=tuple(
                    point for point in slot.selected_metric_points if point.key == metric_key
                ),
            )
        )
    if any(not item.points for item in series):
        return KPlotPayload(
            plot_id=f"cross-k:{metric_key}",
            slot_id="cross-k",
            state_count=0,
            metric_key=metric_key,
            status="not_available",
            series=(),
            logged_model_ids=tuple(item.logged_model_id for item in series),
            feature_order_hashes=tuple(item.metadata.feature_order_sha256 for item in ordered),
            source_data_hashes=tuple(item.metadata.source_data_sha256 for item in ordered),
            metric_catalog_version=METRIC_CATALOG_VERSION,
            comparison_domain_id=K_CROSS_COMPARISON_DOMAIN_ID,
            unavailable_reason="one selected K slot is missing the requested metric",
        )
    return KPlotPayload(
        plot_id=f"cross-k:{metric_key}",
        slot_id="cross-k",
        state_count=0,
        metric_key=metric_key,
        status="available",
        series=tuple(series),
        logged_model_ids=tuple(item.logged_model_id for item in series),
        feature_order_hashes=tuple(item.metadata.feature_order_sha256 for item in ordered),
        source_data_hashes=tuple(item.metadata.source_data_sha256 for item in ordered),
        metric_catalog_version=METRIC_CATALOG_VERSION,
        comparison_domain_id=K_CROSS_COMPARISON_DOMAIN_ID,
    )


def render_k_plot_payload(payload: KPlotPayload, output_dir: str | Path) -> dict[str, object]:
    """Render only a prepared payload and write its manifest beside the PNG."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if payload.status != "available":
        manifest = payload.manifest()
        (root / f"{payload.plot_id.replace(':', '_')}.manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        return manifest
    figure, axis = plt.subplots(figsize=(11.0, 8.0))
    for series in payload.series:
        ordered = tuple(sorted(series.points, key=lambda item: item.step))
        axis.plot(
            [point.step for point in ordered],
            [point.value for point in ordered],
            marker="o",
            linewidth=1.5,
            label=f"{series.model_family} ({series.logged_model_id})",
        )
    axis.set_title(f"{payload.metric_key} — {payload.slot_id}")
    axis.set_xlabel("metric step")
    axis.set_ylabel(payload.metric_key)
    axis.grid(True, alpha=0.25)
    axis.legend()
    safe = payload.plot_id.replace(":", "_").replace("/", "_")
    png = root / f"{safe}.png"
    figure.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(figure)
    manifest = payload.manifest(png_path=str(png))
    (root / f"{safe}.manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = [
    "KPlotPayload",
    "KPlotSeries",
    "build_cross_k_plot_payload",
    "build_k_plot_payload",
    "prepare_k_plot_payloads",
    "render_k_plot_payload",
]
