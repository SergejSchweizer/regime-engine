#!/usr/bin/env python3
"""Verify a deterministic MLflow LoggedModel/Model Metrics expectation bundle."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any, cast

from mlflow.tracking import MlflowClient

from market_regime_engine.mlflow_support.metric_catalog import metric_definition
from market_regime_engine.mlflow_support.ports import MetricPoint

_REQUIRED_TAGS = {
    "regime_engine.metric_catalog_version",
    "regime_engine.evaluation_plan_hash",
    "regime_engine.dataset_snapshot_key",
    "regime_engine.evaluation_run_key",
    "regime_engine.feature_order_sha256",
    "regime_engine.feature_dimension",
    "regime_engine.scope",
}


def _comparison_domain_tags(comparison_domain: str) -> set[str]:
    """Return tags required to prove one metric's comparison domain."""

    tags = {
        "regime_engine.dataset_snapshot_key",
        "regime_engine.evaluation_plan_hash",
    }
    if comparison_domain == "same_feature_vector_source_plan":
        tags.update(
            {
                "regime_engine.feature_order_sha256",
                "regime_engine.feature_dimension",
            }
        )
    elif comparison_domain == "fold_local_only":
        tags.add("regime_engine.outer_fold_id")
    return tags


@dataclass(frozen=True, slots=True)
class AuditCounts:
    model_count: int
    expected_model_count: int
    metric_key_count: int
    metric_point_count: int
    unexpected_metric_key_count: int
    unexpected_point_count: int
    missing_model_count: int
    unexpected_model_count: int
    duplicate_model_count: int
    duplicate_expected_name_count: int
    duplicate_expected_point_count: int
    missing_point_count: int
    duplicate_point_count: int
    conflicting_point_count: int
    unknown_metric_key_count: int
    tag_violation_count: int
    comparison_domain_violation_count: int


def _point_tuple(point: Any) -> tuple[str, int, float, int]:
    if isinstance(point, dict):
        if "timestamp_ms" in point:
            timestamp = point["timestamp_ms"]
        elif "timestamp" in point:
            timestamp = point["timestamp"]
        else:
            raise ValueError("metric point must include timestamp_ms")
        key = str(point["key"])
        step = int(point["step"])
        value = float(point["value"])
        timestamp_ms = int(cast(int | float | str, timestamp))
    else:
        timestamp = getattr(point, "timestamp_ms", getattr(point, "timestamp", None))
        if timestamp is None:
            raise ValueError("metric point must include timestamp_ms")
        key = str(point.key)
        step = int(point.step)
        value = float(point.value)
        timestamp_ms = int(cast(int | float | str, timestamp))
    if not key or not isfinite(value) or step < 0 or timestamp_ms < 0:
        raise ValueError("metric point must have a non-empty key and finite non-negative fields")
    return key, step, value, timestamp_ms


def _metric_points(points: Any) -> tuple[MetricPoint, ...]:
    return tuple(
        MetricPoint(key, value, step, timestamp)
        for key, step, value, timestamp in map(_point_tuple, points)
    )


def _file_store_metric_points(client: MlflowClient, model: Any) -> tuple[MetricPoint, ...] | None:
    """Read every persisted FileStore Model Metric history entry.

    ``LoggedModel.metrics`` is intentionally a summary view in MLflow's
    FileStore: for each metric/dataset pair it returns the greatest step.  A
    completeness audit must inspect the metric files themselves, otherwise a
    missing historical step (or a duplicate at an existing step) can be
    silently hidden by that summary.  Return ``None`` for non-file-backed
    stores, where the public LoggedModel response is the only available view.
    """

    store = getattr(getattr(client, "_tracking_client", None), "store", None)
    get_model_dir = getattr(store, "_get_model_dir", None)
    parse_metric_line = getattr(store, "_get_model_metric_from_line", None)
    if not callable(get_model_dir) or not callable(parse_metric_line):
        return None

    model_dir = get_model_dir(str(model.experiment_id), str(model.model_id))
    if model_dir is None:
        raise ValueError(f"MLflow FileStore has no directory for LoggedModel {model.model_id}")
    metrics_dir = Path(model_dir) / "metrics"
    if not metrics_dir.exists():
        return ()
    if not metrics_dir.is_dir():
        raise ValueError(f"MLflow Model Metrics path is not a directory: {metrics_dir}")

    points: list[Any] = []
    for metric_file in sorted(metrics_dir.iterdir(), key=lambda path: path.name):
        if not metric_file.is_file():
            raise ValueError(f"MLflow Model Metrics entry is not a file: {metric_file}")
        for line_number, line in enumerate(
            metric_file.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                points.append(parse_metric_line(str(model.model_id), metric_file.name, line))
            except Exception as exc:
                raise ValueError(
                    f"malformed MLflow Model Metric {metric_file.name!r} line {line_number}"
                ) from exc
    return _metric_points(points)


def _model_metric_points(client: MlflowClient, model: Any) -> tuple[MetricPoint, ...]:
    """Return the complete available history for one LoggedModel."""

    file_store_points = _file_store_metric_points(client, model)
    if file_store_points is not None:
        return file_store_points
    return _metric_points(model.metrics or ())


def audit(
    tracking_uri: str,
    experiment_name: str,
    expectation: dict[str, Any],
    *,
    ledger_root: str | Path | None = None,
) -> dict[str, object]:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment does not exist: {experiment_name}")
    actual_models = tuple(client.search_logged_models([experiment.experiment_id]))
    expected_entries = tuple(expectation.get("models", ()))
    if not all(isinstance(item, dict) for item in expected_entries):
        raise ValueError("expectation models must be objects")
    expected_names = tuple(str(item.get("name", "")) for item in expected_entries)
    if any(not name for name in expected_names):
        raise ValueError("every expected LoggedModel must have a non-empty name")
    expected_by_name = {str(item["name"]): item for item in expected_entries}
    actual_by_name: dict[str, list[Any]] = {}
    for model in actual_models:
        actual_by_name.setdefault(str(model.name), []).append(model)
    actual_model_points = tuple(
        (model, _model_metric_points(client, model)) for model in actual_models
    )
    actual_points_by_name = {
        name: points
        for name, models in actual_by_name.items()
        if len(models) == 1
        for model, points in actual_model_points
        if str(model.name) == name
    }

    missing_models = set(expected_by_name) - set(actual_by_name)
    unexpected_models = set(actual_by_name) - set(expected_by_name)
    duplicate_models = {name for name, models in actual_by_name.items() if len(models) != 1}
    missing_points = duplicate_points = duplicate_expected_points = conflicting_points = 0
    unknown_keys = tag_violations = domain_violations = 0
    metric_point_count = sum(len(points) for _model, points in actual_model_points)
    unexpected_metric_key_count = 0
    unexpected_point_count = 0

    for name, expected in expected_by_name.items():
        models = actual_by_name.get(name, [])
        if len(models) != 1:
            continue
        model = models[0]
        tags = {str(key): str(value) for key, value in (model.tags or {}).items()}
        required_tag_keys = set(_REQUIRED_TAGS)
        expected_tags = expected.get("required_tags", {})
        if not isinstance(expected_tags, dict):
            raise ValueError(f"required_tags for LoggedModel {name!r} must be an object")
        required_tag_keys.update(str(key) for key in expected_tags)
        tag_violations += sum(key not in tags for key in required_tag_keys)
        tag_violations += sum(
            key in tags and tags[key] != str(value)
            for key, value in ((str(key), value) for key, value in expected_tags.items())
        )

        raw_expected_points = expected.get("points", ())
        if not isinstance(raw_expected_points, (list, tuple)):
            raise ValueError(f"points for LoggedModel {name!r} must be a list")
        expected_tuples = tuple(_point_tuple(item) for item in raw_expected_points)
        actual_tuples = tuple(_point_tuple(item) for item in actual_points_by_name[name])

        expected_by_identity: defaultdict[tuple[str, int], list[tuple[str, int, float, int]]] = (
            defaultdict(list)
        )
        actual_by_identity: defaultdict[tuple[str, int], list[tuple[str, int, float, int]]] = (
            defaultdict(list)
        )
        for point in expected_tuples:
            expected_by_identity[(point[0], point[1])].append(point)
        for point in actual_tuples:
            actual_by_identity[(point[0], point[1])].append(point)

        duplicate_expected_points += sum(
            len(points) - 1 for points in expected_by_identity.values() if len(points) > 1
        )
        duplicate_points += sum(
            len(points) - 1 for points in actual_by_identity.values() if len(points) > 1
        )
        missing_points += len(set(expected_by_identity) - set(actual_by_identity))
        unexpected_point_count += len(set(actual_by_identity) - set(expected_by_identity))
        for identity in set(expected_by_identity) & set(actual_by_identity):
            expected_values = set(expected_by_identity[identity])
            actual_values = set(actual_by_identity[identity])
            if len(expected_values) != 1 or actual_values != expected_values:
                conflicting_points += 1

        actual_keys = set(actual_by_identity)
        expected_keys = set(expected_by_identity)
        unexpected_metric_key_count += len(
            {key for key, _step in actual_keys} - {key for key, _step in expected_keys}
        )

    # Inspect every actual LoggedModel, including unexpected models.  This is
    # deliberately independent of the expectation and of any export ledger:
    # a ledger must never hide a point that is absent, duplicated, or changed
    # in MLflow itself.
    for model, points in actual_model_points:
        tags = {str(key): str(value) for key, value in (model.tags or {}).items()}
        keys = {point.key for point in points}
        unknown_keys += sum(metric_definition(key) is None for key in keys)
        for key in keys:
            definition = metric_definition(key)
            if definition is None:
                continue
            if not definition.model_metrics_visible:
                domain_violations += 1
            domain_tags = _comparison_domain_tags(definition.comparison_domain)
            if any(not tags.get(tag) for tag in domain_tags):
                domain_violations += 1

    comparison_groups = expectation.get("comparison_groups", ())
    if not isinstance(comparison_groups, (list, tuple)):
        raise ValueError("comparison_groups must be a list")
    for group in comparison_groups:
        if not isinstance(group, dict):
            raise ValueError("comparison group must be an object")
        metric_key = str(group.get("metric_key", ""))
        definition = metric_definition(metric_key)
        model_names = group.get("model_names", ())
        if definition is None or not isinstance(model_names, (list, tuple)):
            domain_violations += 1
            continue
        normalized_model_names = tuple(str(model_name) for model_name in model_names)
        if len(normalized_model_names) != len(set(normalized_model_names)):
            domain_violations += 1
            continue
        declared_domain = group.get("comparison_domain")
        if declared_domain is not None and str(declared_domain) != definition.comparison_domain:
            domain_violations += 1
        group_points: dict[str, tuple[MetricPoint, ...]] = {}
        group_tags: dict[str, dict[str, str]] = {}
        for name in normalized_model_names:
            models = actual_by_name.get(name, [])
            if len(models) != 1:
                domain_violations += 1
                continue
            model = models[0]
            group_points[name] = tuple(
                point for point in actual_points_by_name[name] if point.key == metric_key
            )
            group_tags[name] = {str(key): str(value) for key, value in (model.tags or {}).items()}
        if len(group_points) != len(normalized_model_names) or any(
            not points for points in group_points.values()
        ):
            continue
        identities = {
            tuple(
                group_tags[name].get(tag, "")
                for tag in _comparison_domain_tags(definition.comparison_domain)
            )
            for name in group_points
        }
        if any(not all(identity) for identity in identities) or len(identities) != 1:
            domain_violations += 1

    counts = AuditCounts(
        model_count=len(actual_models),
        expected_model_count=len(expected_entries),
        metric_key_count=sum(
            len({item.key for item in points}) for _model, points in actual_model_points
        ),
        metric_point_count=metric_point_count,
        unexpected_metric_key_count=unexpected_metric_key_count,
        unexpected_point_count=unexpected_point_count,
        missing_model_count=len(missing_models),
        unexpected_model_count=len(unexpected_models),
        duplicate_model_count=len(duplicate_models),
        duplicate_expected_name_count=len(expected_names) - len(expected_by_name),
        duplicate_expected_point_count=duplicate_expected_points,
        missing_point_count=missing_points,
        duplicate_point_count=duplicate_points,
        conflicting_point_count=conflicting_points,
        unknown_metric_key_count=unknown_keys,
        tag_violation_count=tag_violations,
        comparison_domain_violation_count=domain_violations,
    )
    report: dict[str, object] = {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "counts": asdict(counts),
        "status": (
            "verified"
            if not any(
                getattr(counts, field)
                for field in (
                    "missing_model_count",
                    "unexpected_model_count",
                    "duplicate_model_count",
                    "duplicate_expected_name_count",
                    "duplicate_expected_point_count",
                    "missing_point_count",
                    "unexpected_point_count",
                    "duplicate_point_count",
                    "conflicting_point_count",
                    "unexpected_metric_key_count",
                    "unknown_metric_key_count",
                    "tag_violation_count",
                    "comparison_domain_violation_count",
                )
            )
            else "failed"
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--expectation", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    expectation = json.loads(args.expectation.read_text(encoding="utf-8"))
    report = audit(
        args.tracking_uri,
        args.experiment,
        expectation,
        ledger_root=args.ledger_root,
    )
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if report["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
