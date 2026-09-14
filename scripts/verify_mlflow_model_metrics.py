#!/usr/bin/env python3
"""Verify a deterministic MLflow LoggedModel/Model Metrics expectation bundle."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, cast

from mlflow.entities import ViewType
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
_EXPECTATION_SCHEMA_VERSION = 1
_EXPECTATION_PROVENANCE_FIELDS = {
    "generator",
    "source_artifact",
    "source_artifact_sha256",
    "evaluation_run_key",
    "evaluation_plan_hash",
    "dataset_snapshot_key",
    "feature_order_sha256",
    "feature_dimension",
}
_PROVENANCE_TAG_FIELDS = {
    "evaluation_run_key": "regime_engine.evaluation_run_key",
    "evaluation_plan_hash": "regime_engine.evaluation_plan_hash",
    "dataset_snapshot_key": "regime_engine.dataset_snapshot_key",
    "feature_order_sha256": "regime_engine.feature_order_sha256",
    "feature_dimension": "regime_engine.feature_dimension",
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
    historical_namespace_violation_count: int
    nonempty_model_set_violation_count: int
    expectation_contract_violation_count: int
    nonready_model_count: int
    model_source_run_status_violation_count: int
    nonterminal_run_count: int
    deleted_logged_model_count: int
    deleted_logged_model_inventory_unavailable_count: int
    registered_model_count: int
    registered_model_version_count: int


@dataclass(frozen=True, slots=True)
class _LoadedModel:
    """One LoggedModel and its complete available Model Metrics history."""

    model: Any
    points: tuple[MetricPoint, ...]


def _search_all_logged_models(client: MlflowClient, experiment_id: str) -> tuple[Any, ...]:
    """Read every LoggedModel, including pages beyond MLflow's default limit."""

    models: list[Any] = []
    page_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"max_results": 1000}
        if page_token is not None:
            kwargs["page_token"] = page_token
        page = client.search_logged_models([experiment_id], **kwargs)
        models.extend(page)
        next_token = getattr(page, "token", None)
        if not next_token:
            return tuple(models)
        page_token = str(next_token)


def _search_all_runs(client: MlflowClient, experiment_id: str) -> tuple[Any, ...]:
    """Read every active and deleted run in an experiment, including all pages."""

    runs: list[Any] = []
    page_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {
            "max_results": 1000,
            "run_view_type": ViewType.ALL,
        }
        if page_token is not None:
            kwargs["page_token"] = page_token
        page = client.search_runs([experiment_id], **kwargs)
        runs.extend(page)
        next_token = getattr(page, "token", None)
        if not next_token:
            return tuple(runs)
        page_token = str(next_token)


def _search_all_registered_models(client: MlflowClient) -> tuple[Any, ...]:
    """Read the complete registry namespace, including pages past the default."""

    models: list[Any] = []
    page_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"max_results": 1000}
        if page_token is not None:
            kwargs["page_token"] = page_token
        page = client.search_registered_models(**kwargs)
        models.extend(page)
        next_token = getattr(page, "token", None)
        if not next_token:
            return tuple(models)
        page_token = str(next_token)


def _search_all_registered_model_versions(client: MlflowClient) -> tuple[Any, ...]:
    """Read all registered versions so pending legacy IDs cannot be hidden."""

    versions: list[Any] = []
    page_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"max_results": 1000}
        if page_token is not None:
            kwargs["page_token"] = page_token
        page = client.search_model_versions(**kwargs)
        versions.extend(page)
        next_token = getattr(page, "token", None)
        if not next_token:
            return tuple(versions)
        page_token = str(next_token)


def _deleted_logged_model_ids(client: MlflowClient) -> tuple[str, ...] | None:
    """Return deleted IDs, or ``None`` when the backend cannot list them.

    The SQLAlchemy store exposes a private inventory helper, while MLflow's
    HTTP RestStore has no equivalent list-deleted-LoggedModels operation.
    ``None`` is therefore different from an empty inventory: the verifier must
    fail closed instead of treating an unobservable namespace as clean.
    """

    store = getattr(getattr(client, "_tracking_client", None), "store", None)
    reader = getattr(store, "_get_deleted_logged_models", None)
    if not callable(reader):
        return None
    return tuple(sorted(str(model_id) for model_id in reader(older_than=0)))


def _canonical_expectation_bytes(expectation: Mapping[str, Any]) -> bytes:
    """Canonicalize an expectation while excluding its self-reported evidence hash."""

    payload = dict(expectation)
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        normalized_provenance = dict(provenance)
        normalized_provenance.pop("evidence_sha256", None)
        payload["provenance"] = normalized_provenance
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def build_expectation_bundle(source: Mapping[str, Any]) -> dict[str, Any]:
    """Build a hashed expectation from independent evidence, never from MLflow.

    ``source`` must contain the model/point manifest and its provenance.  The
    returned bundle is the only form accepted by the strict CLI audit.  The
    evidence hash covers every field except the hash itself, making accidental
    edits detectable without contacting the tracking server.
    """

    bundle = dict(source)
    bundle["schema_version"] = _EXPECTATION_SCHEMA_VERSION
    provenance = bundle.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("expectation provenance must be an object")
    normalized_provenance = dict(provenance)
    normalized_provenance.pop("evidence_sha256", None)
    bundle["provenance"] = normalized_provenance
    bundle["provenance"]["evidence_sha256"] = sha256(
        _canonical_expectation_bytes(bundle)
    ).hexdigest()
    violations = _expectation_contract_violations(bundle)
    if violations:
        raise ValueError("invalid expectation bundle: " + "; ".join(violations))
    return bundle


def _expectation_contract_violations(expectation: Mapping[str, Any]) -> tuple[str, ...]:
    """Return all strict expectation-contract violations deterministically."""

    violations: list[str] = []
    if expectation.get("schema_version") != _EXPECTATION_SCHEMA_VERSION:
        violations.append("unsupported expectation schema_version")
    provenance = expectation.get("provenance")
    if not isinstance(provenance, Mapping):
        return ("expectation provenance is missing or not an object",)
    missing = sorted(field for field in _EXPECTATION_PROVENANCE_FIELDS if not provenance.get(field))
    violations.extend(f"missing provenance field: {field}" for field in missing)
    source_hash = provenance.get("source_artifact_sha256")
    if source_hash and (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdef" for character in source_hash)
    ):
        violations.append("source_artifact_sha256 is not a lowercase SHA-256")
    evidence_hash = provenance.get("evidence_sha256")
    if (
        not isinstance(evidence_hash, str)
        or evidence_hash != sha256(_canonical_expectation_bytes(expectation)).hexdigest()
    ):
        violations.append("expectation evidence_sha256 does not match canonical content")
    if provenance.get("feature_dimension") is not None:
        try:
            if int(provenance["feature_dimension"]) < 1:
                violations.append("feature_dimension must be positive")
        except TypeError, ValueError:
            violations.append("feature_dimension must be an integer")

    entries = expectation.get("models")
    if not isinstance(entries, (list, tuple)):
        violations.append("expectation models must be a list")
        return tuple(violations)
    names: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            violations.append(f"model expectation {index} is not an object")
            continue
        name = str(entry.get("name", ""))
        if not name:
            violations.append(f"model expectation {index} has no name")
        names.append(name)
        expected_tags = entry.get("required_tags")
        if not isinstance(expected_tags, Mapping):
            violations.append(f"model expectation {name!r} has no required_tags")
            continue
        for key in _REQUIRED_TAGS:
            value = expected_tags.get(key)
            if value is None or not str(value).strip():
                violations.append(f"model expectation {name!r} has empty tag {key}")
        for field, tag_key in _PROVENANCE_TAG_FIELDS.items():
            if str(expected_tags.get(tag_key, "")) != str(provenance.get(field, "")):
                violations.append(f"model expectation {name!r} disagrees with provenance {field}")
    if len(names) != len(set(names)):
        violations.append("expectation contains duplicate model names")
    return tuple(violations)


def _namespace_inventory(
    experiment_id: str,
    runs: tuple[Any, ...],
    models: tuple[Any, ...],
    deleted_model_ids: tuple[str, ...] | None,
    registered_models: tuple[Any, ...],
    registered_versions: tuple[Any, ...],
) -> dict[str, object]:
    """Return read-only evidence about the complete MLflow namespace."""

    status_counts = Counter(str(run.info.status) for run in runs)
    lifecycle_counts = Counter(str(run.info.lifecycle_stage) for run in runs)
    deleted_inventory_available = deleted_model_ids is not None
    deleted_ids = () if deleted_model_ids is None else deleted_model_ids
    return {
        "experiment_id": experiment_id,
        "run_count": len(runs),
        "run_status_counts": dict(sorted(status_counts.items())),
        "run_lifecycle_counts": dict(sorted(lifecycle_counts.items())),
        "logged_model_count": len(models),
        "deleted_logged_model_inventory_available": deleted_inventory_available,
        "deleted_logged_model_count": len(deleted_ids),
        "deleted_logged_model_ids": list(deleted_ids),
        "registered_model_count": len(registered_models),
        "registered_model_names": sorted(str(model.name) for model in registered_models),
        "registered_model_version_count": len(registered_versions),
        "registered_model_version_ids": sorted(
            str(version.version) for version in registered_versions
        ),
        # Registered production packages are intentionally inventoried but
        # are not part of the evaluation-namespace zero-survivor condition.
        "historical_objects_zero": (
            deleted_inventory_available and not runs and not models and not deleted_ids
        ),
    }


def _tags(model: Any) -> dict[str, str]:
    raw_tags = getattr(model, "tags", None) or {}
    if isinstance(raw_tags, Mapping):
        return {str(key): str(value) for key, value in raw_tags.items()}
    return {str(item.key): str(item.value) for item in raw_tags}


def _load_experiment(
    tracking_uri: str,
    experiment_name: str,
) -> tuple[_LoadedModel, ...]:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment does not exist: {experiment_name}")
    loaded = tuple(
        _LoadedModel(model, _model_metric_points(client, model))
        for model in _search_all_logged_models(client, str(experiment.experiment_id))
    )
    return loaded


def _point_identity(point: MetricPoint) -> tuple[str, int, str, int]:
    """Return a byte-stable identity for cross-run history comparison."""

    return point.key, point.step, point.value.hex(), point.timestamp_ms


def _model_name_index(
    loaded: tuple[_LoadedModel, ...],
) -> dict[str, list[_LoadedModel]]:
    by_name: dict[str, list[_LoadedModel]] = {}
    for item in loaded:
        by_name.setdefault(str(item.model.name), []).append(item)
    return by_name


def compare_metric_histories(
    tracking_uri: str,
    experiment_name: str,
    baseline_tracking_uri: str,
    baseline_experiment_name: str,
) -> dict[str, object]:
    """Compare final resumed and uninterrupted Model Metrics histories.

    Model IDs and MLflow run IDs are intentionally excluded: a restarted
    evaluation may receive new operational IDs while retaining the same
    logical LoggedModel names.  Metric key, step, exact IEEE-754 value and
    timestamp must nevertheless match, and ``regime_engine.*`` lineage tags
    must remain identical.
    """

    resumed = _load_experiment(tracking_uri, experiment_name)
    baseline = _load_experiment(baseline_tracking_uri, baseline_experiment_name)
    resumed_by_name = _model_name_index(resumed)
    baseline_by_name = _model_name_index(baseline)
    missing_models = set(baseline_by_name) - set(resumed_by_name)
    unexpected_models = set(resumed_by_name) - set(baseline_by_name)
    duplicate_model_count = sum(
        len(items) - 1
        for by_name in (resumed_by_name, baseline_by_name)
        for items in by_name.values()
        if len(items) > 1
    )
    missing_points = unexpected_points = conflicting_points = duplicate_points = 0
    tag_mismatches = 0
    per_model: list[dict[str, object]] = []

    for name in sorted(set(resumed_by_name) & set(baseline_by_name)):
        resumed_items = resumed_by_name[name]
        baseline_items = baseline_by_name[name]
        if len(resumed_items) != 1 or len(baseline_items) != 1:
            continue
        resumed_item = resumed_items[0]
        baseline_item = baseline_items[0]
        resumed_points = tuple(_point_identity(point) for point in resumed_item.points)
        baseline_points = tuple(_point_identity(point) for point in baseline_item.points)
        resumed_counter = Counter(resumed_points)
        baseline_counter = Counter(baseline_points)
        missing_points += sum((baseline_counter - resumed_counter).values())
        unexpected_points += sum((resumed_counter - baseline_counter).values())
        duplicate_points += sum(max(count - 1, 0) for count in resumed_counter.values())
        duplicate_points += sum(max(count - 1, 0) for count in baseline_counter.values())
        resumed_by_key_step = {(point[0], point[1]) for point in resumed_points}
        baseline_by_key_step = {(point[0], point[1]) for point in baseline_points}
        model_conflicting_points = sum(
            1
            for identity in resumed_by_key_step & baseline_by_key_step
            if {point for point in resumed_points if point[:2] == identity}
            != {point for point in baseline_points if point[:2] == identity}
        )
        conflicting_points += model_conflicting_points
        resumed_tags = {
            key: value
            for key, value in _tags(resumed_item.model).items()
            if key.startswith("regime_engine.")
        }
        baseline_tags = {
            key: value
            for key, value in _tags(baseline_item.model).items()
            if key.startswith("regime_engine.")
        }
        tag_mismatches += int(resumed_tags != baseline_tags)
        per_model.append(
            {
                "name": name,
                "baseline_metric_point_count": len(baseline_points),
                "resumed_metric_point_count": len(resumed_points),
                "missing_point_count": sum((baseline_counter - resumed_counter).values()),
                "unexpected_point_count": sum((resumed_counter - baseline_counter).values()),
                "conflicting_point_count": model_conflicting_points,
                "lineage_tag_mismatch": resumed_tags != baseline_tags,
            }
        )

    result: dict[str, object] = {
        "baseline_tracking_uri": baseline_tracking_uri,
        "baseline_experiment_name": baseline_experiment_name,
        "resumed_tracking_uri": tracking_uri,
        "resumed_experiment_name": experiment_name,
        "baseline_model_count": len(baseline),
        "resumed_model_count": len(resumed),
        "missing_model_count": len(missing_models),
        "unexpected_model_count": len(unexpected_models),
        "duplicate_model_count": duplicate_model_count,
        "missing_point_count": missing_points,
        "unexpected_point_count": unexpected_points,
        "duplicate_point_count": duplicate_points,
        "conflicting_point_count": conflicting_points,
        "lineage_tag_mismatch_count": tag_mismatches,
        "models": per_model,
    }
    result["status"] = (
        "verified"
        if not any(
            result[field]
            for field in (
                "missing_model_count",
                "unexpected_model_count",
                "duplicate_model_count",
                "missing_point_count",
                "unexpected_point_count",
                "duplicate_point_count",
                "conflicting_point_count",
                "lineage_tag_mismatch_count",
            )
        )
        else "failed"
    )
    return result


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
    baseline_tracking_uri: str | None = None,
    baseline_experiment_name: str | None = None,
    require_clean_namespace: bool = False,
    require_nonempty_model_set: bool = False,
    require_expectation_contract: bool = False,
    require_terminal_model_runs: bool = False,
) -> dict[str, object]:
    if (baseline_tracking_uri is None) != (baseline_experiment_name is None):
        raise ValueError(
            "baseline_tracking_uri and baseline_experiment_name must be supplied together"
        )
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment does not exist: {experiment_name}")
    actual_models = _search_all_logged_models(client, str(experiment.experiment_id))
    actual_runs = _search_all_runs(client, str(experiment.experiment_id))
    deleted_model_ids = _deleted_logged_model_ids(client)
    registered_models = _search_all_registered_models(client)
    registered_versions = _search_all_registered_model_versions(client)
    namespace = _namespace_inventory(
        str(experiment.experiment_id),
        actual_runs,
        actual_models,
        deleted_model_ids,
        registered_models,
        registered_versions,
    )
    expectation_contract_violations = (
        _expectation_contract_violations(expectation) if require_expectation_contract else ()
    )
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
    run_by_id = {str(run.info.run_id): run for run in actual_runs}
    terminal_run_statuses = {"FINISHED", "FAILED", "KILLED"}
    nonterminal_run_count = sum(
        str(run.info.status) not in terminal_run_statuses for run in actual_runs
    )

    def source_run_is_not_finished(model: Any) -> bool:
        source_run_id = getattr(model, "source_run_id", None)
        source_run = run_by_id.get(str(source_run_id)) if source_run_id is not None else None
        return source_run is None or str(source_run.info.status) != "FINISHED"

    model_source_run_status_violation_count = sum(
        1 for model in actual_models if source_run_is_not_finished(model)
    )
    nonready_model_count = sum(
        str(getattr(model, "status", "")) != "READY" for model in actual_models
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
        tags = _tags(model)
        required_tag_keys = set(_REQUIRED_TAGS)
        expected_tags = expected.get("required_tags", {})
        if not isinstance(expected_tags, dict):
            raise ValueError(f"required_tags for LoggedModel {name!r} must be an object")
        required_tag_keys.update(str(key) for key in expected_tags)
        tag_violations += sum(
            key not in tags or not str(tags[key]).strip() for key in required_tag_keys
        )
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
        tags = _tags(model)
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
            group_tags[name] = _tags(model)
        missing_group_points = len(normalized_model_names) - len(group_points)
        missing_group_points += sum(not points for points in group_points.values())
        if missing_group_points:
            domain_violations += missing_group_points
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
        historical_namespace_violation_count=int(
            require_clean_namespace and not bool(namespace["historical_objects_zero"])
        ),
        nonempty_model_set_violation_count=int(
            require_nonempty_model_set and (not expected_entries or not actual_models)
        ),
        expectation_contract_violation_count=len(expectation_contract_violations),
        nonready_model_count=nonready_model_count if require_terminal_model_runs else 0,
        model_source_run_status_violation_count=(
            model_source_run_status_violation_count if require_terminal_model_runs else 0
        ),
        nonterminal_run_count=nonterminal_run_count,
        deleted_logged_model_count=len(deleted_model_ids or ()),
        deleted_logged_model_inventory_unavailable_count=int(deleted_model_ids is None),
        registered_model_count=len(registered_models),
        registered_model_version_count=len(registered_versions),
    )
    model_summaries = tuple(
        {
            "model_id": str(model.model_id),
            "name": str(model.name),
            "scope": _tags(model).get("regime_engine.scope", "<missing>"),
            "metric_key_count": len({point.key for point in points}),
            "metric_point_count": len(points),
        }
        for model, points in sorted(
            actual_model_points,
            key=lambda item: (str(item[0].name), str(item[0].model_id)),
        )
    )
    report: dict[str, object] = {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "namespace": namespace,
        "requirements": {
            "require_clean_namespace": require_clean_namespace,
            "require_nonempty_model_set": require_nonempty_model_set,
            "require_expectation_contract": require_expectation_contract,
            "require_terminal_model_runs": require_terminal_model_runs,
        },
        "expectation_contract_violations": list(expectation_contract_violations),
        "counts": asdict(counts),
        "model_count_by_scope": dict(
            sorted(Counter(str(item["scope"]) for item in model_summaries).items())
        ),
        "models": model_summaries,
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
                    "historical_namespace_violation_count",
                    "nonempty_model_set_violation_count",
                    "expectation_contract_violation_count",
                    "nonready_model_count",
                    "model_source_run_status_violation_count",
                )
            )
            else "failed"
        ),
    }
    if baseline_tracking_uri is not None and baseline_experiment_name is not None:
        resume_parity = compare_metric_histories(
            tracking_uri,
            experiment_name,
            baseline_tracking_uri,
            baseline_experiment_name,
        )
        report["resume_parity"] = resume_parity
        if resume_parity["status"] != "verified":
            report["status"] = "failed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--expectation",
        type=Path,
        help="JSON expectation; omit only for --require-clean-namespace preflight",
    )
    parser.add_argument(
        "--require-clean-namespace",
        action="store_true",
        help="fail unless the experiment has zero active or deleted runs and LoggedModels",
    )
    parser.add_argument(
        "--require-nonempty",
        action="store_true",
        help="fail unless both the expectation and MLflow contain LoggedModels",
    )
    parser.add_argument(
        "--require-expectation-contract",
        action="store_true",
        help="require schema, provenance and canonical evidence-hash fields",
    )
    parser.add_argument(
        "--require-terminal-model-runs",
        action="store_true",
        help="fail unless every audited LoggedModel is READY and sourced by a FINISHED run",
    )
    parser.add_argument(
        "--baseline-tracking-uri",
        help="uninterrupted MLflow URI for final resumed-vs-baseline parity evidence",
    )
    parser.add_argument(
        "--baseline-experiment",
        help="uninterrupted MLflow experiment for final resumed-vs-baseline parity evidence",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.expectation is None and not args.require_clean_namespace:
        parser.error("--expectation is required unless --require-clean-namespace is set")
    expectation = (
        {} if args.expectation is None else json.loads(args.expectation.read_text(encoding="utf-8"))
    )
    report = audit(
        args.tracking_uri,
        args.experiment,
        expectation,
        baseline_tracking_uri=args.baseline_tracking_uri,
        baseline_experiment_name=args.baseline_experiment,
        require_clean_namespace=args.require_clean_namespace,
        require_nonempty_model_set=args.require_nonempty,
        require_expectation_contract=args.require_expectation_contract,
        require_terminal_model_runs=args.require_terminal_model_runs,
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
