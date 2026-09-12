#!/usr/bin/env python3
"""Verify a deterministic MLflow LoggedModel/Model Metrics expectation bundle."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
    missing_point_count: int
    duplicate_point_count: int
    conflicting_point_count: int
    unknown_metric_key_count: int
    tag_violation_count: int
    comparison_domain_violation_count: int


def _point_tuple(point: Any) -> tuple[str, int, float, int]:
    if isinstance(point, dict):
        return (
            str(point["key"]),
            int(point["step"]),
            float(point["value"]),
            int(point.get("timestamp_ms", point.get("timestamp", 0))),
        )
    return (str(point.key), int(point.step), float(point.value), int(point.timestamp))


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
    expected_names = tuple(str(item["name"]) for item in expected_entries)
    expected_by_name = {str(item["name"]): item for item in expected_entries}
    actual_by_name: dict[str, list[Any]] = {}
    for model in actual_models:
        actual_by_name.setdefault(model.name, []).append(model)

    missing_models = set(expected_by_name) - set(actual_by_name)
    unexpected_models = set(actual_by_name) - set(expected_by_name)
    duplicate_models = {name for name, models in actual_by_name.items() if len(models) != 1}
    missing_points = duplicate_points = conflicting_points = 0
    unknown_keys = tag_violations = domain_violations = 0
    metric_keys: set[str] = set()
    metric_point_count = 0
    unexpected_metric_key_count = 0
    unexpected_point_count = 0
    ledger_states: dict[str, tuple[Any, ...]] = {}
    if ledger_root is not None:
        from market_regime_engine.mlflow_support.metric_export import MetricExportLedger

        ledger = MetricExportLedger(ledger_root)
        ledger_states = {
            name: ledger.states(name) for name in expected_by_name if ledger.states(name)
        }

    for name, expected in expected_by_name.items():
        models = actual_by_name.get(name, [])
        if len(models) != 1:
            continue
        model = models[0]
        required_tags = set(_REQUIRED_TAGS)
        required_tags.update(str(key) for key in expected.get("required_tags", {}))
        tag_violations += len(required_tags - set(model.tags))
        for key, value in expected.get("required_tags", {}).items():
            tag_violations += int(model.tags.get(str(key)) != str(value))
        actual_points = (
            tuple(
                MetricPoint(item.key, item.value, item.step, item.timestamp_ms)
                for item in ledger_states.get(name, ())
                if item.emitted
            )
            if name in ledger_states
            else tuple(model.metrics or ())
        )
        actual_tuples = tuple(_point_tuple(item) for item in actual_points)
        expected_tuples = tuple(_point_tuple(item) for item in expected.get("points", ()))
        expected_identity = {(key, step) for key, step, _value, _timestamp in expected_tuples}
        actual_identity = {(key, step) for key, step, _value, _timestamp in actual_tuples}
        duplicate_points += len(actual_tuples) - len(actual_identity)
        missing_points += len(expected_identity - actual_identity)
        unexpected_point_count += len(actual_identity - expected_identity)
        for identity in expected_identity & actual_identity:
            expected_point = next(item for item in expected_tuples if item[:2] == identity)
            actual_matches = tuple(item for item in actual_tuples if item[:2] == identity)
            if any(item != expected_point for item in actual_matches):
                conflicting_points += 1
        metric_point_count += len(actual_tuples)
        actual_keys = {key for key, _step, _value, _timestamp in actual_tuples}
        expected_keys = {key for key, _step, _value, _timestamp in expected_tuples}
        unexpected_metric_key_count += len(actual_keys - expected_keys)

    actual_model_points = tuple(tuple(model.metrics or ()) for model in actual_models)
    for points in actual_model_points:
        metric_keys.update(str(point.key) for point in points)
        unknown_keys += sum(metric_definition(key) is None for key in metric_keys)
        for key in metric_keys:
            definition = metric_definition(key)
            if definition is None or not definition.model_metrics_visible:
                domain_violations += 1
        metric_keys.clear()

    counts = AuditCounts(
        model_count=len(actual_models),
        expected_model_count=len(expected_entries),
        metric_key_count=sum(len({item.key for item in points}) for points in actual_model_points),
        metric_point_count=metric_point_count,
        unexpected_metric_key_count=unexpected_metric_key_count,
        unexpected_point_count=unexpected_point_count,
        missing_model_count=len(missing_models),
        unexpected_model_count=len(unexpected_models),
        duplicate_model_count=len(duplicate_models),
        duplicate_expected_name_count=len(expected_names) - len(expected_by_name),
        missing_point_count=missing_points,
        duplicate_point_count=duplicate_points,
        conflicting_point_count=conflicting_points,
        unknown_metric_key_count=unknown_keys,
        tag_violation_count=tag_violations,
        comparison_domain_violation_count=domain_violations,
    )
    report = {
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
