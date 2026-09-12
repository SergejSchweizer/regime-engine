"""Model-Metrics-only data builders for the nine supported plot families."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from types import MappingProxyType

from market_regime_engine.mlflow_support.metric_catalog import (
    METRIC_CATALOG_VERSION,
    metric_definition,
    require_metric_definition,
)
from market_regime_engine.mlflow_support.plots import validate_model_metric_comparison
from market_regime_engine.mlflow_support.ports import MetricPoint


@dataclass(frozen=True, slots=True)
class PlotFamilySpec:
    plot_id: str
    title: str
    required_pattern_groups: tuple[tuple[str, ...], ...]
    x_axis_field: str
    y_axis_field: str
    required_tags: tuple[str, ...] = ()
    single_model_only: bool = False


PLOT_FAMILY_SPECS: Mapping[str, PlotFamilySpec] = MappingProxyType(
    {
        "likelihood": PlotFamilySpec(
            "likelihood",
            "Comparable TRAIN/OOS likelihood histories",
            (
                (r"fit_quality_train_loglik_per_obs",),
                (r"fit_quality_oos_predictive_loglik_per_obs",),
            ),
            "step",
            "metric_value",
        ),
        "information_criteria": PlotFamilySpec(
            "information_criteria",
            "AIC/BIC/HQC histories",
            ((r"fit_quality_aic",), (r"fit_quality_bic",), (r"fit_quality_hqc",)),
            "step",
            "metric_value",
        ),
        "state_posterior": PlotFamilySpec(
            "state_posterior",
            "Filtered posterior and Viterbi histories",
            (
                (r"state_diag_posterior_probability_state_[0-9]+",),
                (r"state_diag_viterbi_state",),
            ),
            "step",
            "metric_value",
            ("regime_engine.state_identity_scope",),
            True,
        ),
        "emission_fit": PlotFamilySpec(
            "emission_fit",
            "State emission fit data",
            (
                (r"state_diag_emission_mean_state_[0-9]+_feature_[0-9]+",),
                (r"state_diag_emission_variance_state_[0-9]+_feature_[0-9]+",),
            ),
            "step",
            "metric_value",
            ("regime_engine.state_identity_scope",),
            True,
        ),
        "innovations": PlotFamilySpec(
            "innovations",
            "Forecast innovation histories",
            ((r"predictive_forecast_h[0-9]+_residual",),),
            "step",
            "metric_value",
            ("regime_engine.forecast_contract_hash",),
        ),
        "rolling_oos_errors": PlotFamilySpec(
            "rolling_oos_errors",
            "Rolling out-of-sample forecast errors",
            ((r"predictive_h[0-9]+_(?:rmse|mae|mape|r2)",),),
            "step",
            "metric_value",
            ("regime_engine.forecast_contract_hash",),
        ),
        "labeled_state_quality": PlotFamilySpec(
            "labeled_state_quality",
            "Optional labeled-state quality",
            ((r"classification_(?:hard_ari|hard_nmi|hard_accuracy|hard_purity|soft_nmi)",),),
            "step",
            "metric_value",
            (
                "regime_engine.state_identity_scope",
                "regime_engine.label_identity",
            ),
            True,
        ),
        "dwell_transition": PlotFamilySpec(
            "dwell_transition",
            "Dwell and transition statistics",
            (
                (r"state_diag_expected_duration_state_[0-9]+",),
                (r"state_diag_transition_probability_state_[0-9]+_to_state_[0-9]+",),
            ),
            "step",
            "metric_value",
            ("regime_engine.state_identity_scope",),
            True,
        ),
        "backtest": PlotFamilySpec(
            "backtest",
            "Causal backtest performance",
            (
                (r"backtest_net_return",),
                (r"backtest_equity",),
                (r"backtest_drawdown",),
            ),
            "step",
            "metric_value",
            (
                "regime_engine.backtest_contract_hash",
                "regime_engine.backtest_data_snapshot_identity",
            ),
        ),
    }
)


@dataclass(frozen=True, slots=True)
class PlotSeries:
    model_id: str
    metric_key: str
    points: tuple[MetricPoint, ...]


@dataclass(frozen=True, slots=True)
class PlotData:
    plot_id: str
    title: str
    status: str
    source_metric_keys: tuple[str, ...]
    x_axis_field: str
    y_axis_field: str
    series: tuple[PlotSeries, ...]
    model_tags: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"available", "not_available"}:
            raise ValueError("plot data status must be available or not_available")
        if self.status == "available" and not self.series:
            raise ValueError("available plot data requires metric series")
        if self.status == "not_available" and not self.unavailable_reason:
            raise ValueError("unavailable plot data requires a reason")

    @property
    def canonical_json(self) -> str:
        payload = {
            "plot_id": self.plot_id,
            "title": self.title,
            "status": self.status,
            "catalog_version": METRIC_CATALOG_VERSION,
            "source_metric_keys": self.source_metric_keys,
            "x_axis_field": self.x_axis_field,
            "y_axis_field": self.y_axis_field,
            "series": [
                {
                    "model_id": item.model_id,
                    "metric_key": item.metric_key,
                    "points": [asdict(point) for point in item.points],
                }
                for item in self.series
            ],
            "model_tags": self.model_tags,
            "unavailable_reason": self.unavailable_reason,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @property
    def source_hash(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()


def _unavailable(spec: PlotFamilySpec, reason: str) -> PlotData:
    return PlotData(
        plot_id=spec.plot_id,
        title=spec.title,
        status="not_available",
        source_metric_keys=(),
        x_axis_field=spec.x_axis_field,
        y_axis_field=spec.y_axis_field,
        series=(),
        model_tags=(),
        unavailable_reason=reason,
    )


def _validate_input_points(
    model_metric_points: Mapping[str, Sequence[MetricPoint]],
    model_tags: Mapping[str, Mapping[str, str]],
) -> None:
    if not model_metric_points:
        return
    for model_id, points in model_metric_points.items():
        if not model_id or model_id not in model_tags:
            raise ValueError(f"missing tags for LoggedModel {model_id!r}")
        seen: set[tuple[str, int]] = set()
        for point in points:
            if not isinstance(point, MetricPoint):
                raise TypeError("plot data requires MetricPoint histories")
            require_metric_definition(point.key)
            identity = (point.key, point.step)
            if identity in seen:
                raise ValueError(f"duplicate Model Metrics step {point.key}@{point.step}")
            seen.add(identity)


def _matching_keys(points: Sequence[MetricPoint], patterns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                point.key
                for point in points
                if any(re.fullmatch(pattern, point.key) is not None for pattern in patterns)
            }
        )
    )


def build_plot_data(
    plot_id: str,
    model_metric_points: Mapping[str, Sequence[MetricPoint]],
    model_tags: Mapping[str, Mapping[str, str]],
) -> PlotData:
    """Build one plot payload using only standard LoggedModel metrics and tags."""

    try:
        spec = PLOT_FAMILY_SPECS[plot_id]
    except KeyError as exc:
        raise ValueError(f"unknown plot family: {plot_id}") from exc
    _validate_input_points(model_metric_points, model_tags)
    if not model_metric_points:
        return _unavailable(spec, "no LoggedModel Model Metrics were supplied")
    if spec.single_model_only and len(model_metric_points) != 1:
        return _unavailable(
            spec,
            "state-indexed or labeled metrics cannot be compared across LoggedModels",
        )

    keys_by_model = {
        model_id: tuple(
            _matching_keys(points, patterns) for patterns in spec.required_pattern_groups
        )
        for model_id, points in model_metric_points.items()
    }
    if any(not group_keys for groups in keys_by_model.values() for group_keys in groups):
        return _unavailable(spec, "required metric family is absent from at least one LoggedModel")
    for group_index in range(len(spec.required_pattern_groups)):
        common_keys = set(keys_by_model[next(iter(keys_by_model))][group_index])
        if any(set(groups[group_index]) != common_keys for groups in keys_by_model.values()):
            return _unavailable(
                spec,
                "LoggedModels do not expose the same indexed metric keys for this plot family",
            )
    selected_keys = tuple(
        sorted({key for groups in keys_by_model.values() for group in groups for key in group})
    )
    selected_points: dict[str, dict[str, tuple[MetricPoint, ...]]] = {}
    try:
        for metric_key in selected_keys:
            definition = metric_definition(metric_key)
            if definition is None:
                raise ValueError(f"metric key is not registered: {metric_key}")
            current = {
                model_id: tuple(point for point in points if point.key == metric_key)
                for model_id, points in model_metric_points.items()
            }
            validate_model_metric_comparison(metric_key, current, model_tags)
            for model_id, points in current.items():
                selected_points.setdefault(model_id, {})[metric_key] = points
    except ValueError as exc:
        return _unavailable(spec, str(exc))

    for required_tag in spec.required_tags:
        if any(not model_tags[model_id].get(required_tag) for model_id in model_metric_points):
            return _unavailable(spec, f"required plot lineage tag is missing: {required_tag}")
        values = {model_tags[model_id][required_tag] for model_id in model_metric_points}
        if len(values) != 1:
            return _unavailable(
                spec, f"plot lineage tag differs across LoggedModels: {required_tag}"
            )
    tag_identity = tuple(
        (
            model_id,
            tuple(sorted((str(key), str(value)) for key, value in model_tags[model_id].items())),
        )
        for model_id in sorted(model_tags)
        if model_id in model_metric_points
    )
    series = tuple(
        PlotSeries(model_id, metric_key, selected_points[model_id][metric_key])
        for model_id in sorted(selected_points)
        for metric_key in sorted(selected_points[model_id])
    )
    return PlotData(
        plot_id=spec.plot_id,
        title=spec.title,
        status="available",
        source_metric_keys=selected_keys,
        x_axis_field=spec.x_axis_field,
        y_axis_field=spec.y_axis_field,
        series=series,
        model_tags=tag_identity,
    )


def build_nine_plot_data(
    model_metric_points: Mapping[str, Sequence[MetricPoint]],
    model_tags: Mapping[str, Mapping[str, str]],
) -> tuple[PlotData, ...]:
    """Return all nine deterministic payloads, including explicit unavailable ones."""

    return tuple(
        build_plot_data(plot_id, model_metric_points, model_tags) for plot_id in PLOT_FAMILY_SPECS
    )


__all__ = [
    "PLOT_FAMILY_SPECS",
    "PlotData",
    "PlotFamilySpec",
    "PlotSeries",
    "build_nine_plot_data",
    "build_plot_data",
]
