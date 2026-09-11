"""Deterministic, scope-checked cleanup plans for the MLflow service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from market_regime_engine.mlflow_support.registry import REGISTERED_MODEL_NAME

EVALUATION_EXPERIMENT_NAME = "regime-engine-evaluation"
V4_PACKAGE_SCHEMA = "RegimeEngineProductionModel.v4"
ALLOWED_ALIASES = frozenset({"challenger", "champion"})


class CleanupClient(Protocol):
    def get_experiment_by_name(self, name: str) -> Any: ...

    def search_runs(self, experiment_ids: list[str]) -> list[Any]: ...

    def search_logged_models(self, experiment_ids: list[str]) -> list[Any]: ...


def _is_missing_error(error: Exception) -> bool:
    """Recognize only provider errors that prove an object is already absent."""

    if isinstance(error, KeyError):
        return True
    error_code = str(getattr(error, "error_code", ""))
    if error_code in {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"}:
        return True
    message = str(error).lower()
    return "does not exist" in message or "not found" in message


def _delete_if_present(delete: Any, identifier: str) -> None:
    try:
        delete(identifier)
    except Exception as error:
        if not _is_missing_error(error):
            raise


@dataclass(frozen=True, slots=True)
class EvaluationCleanupManifest:
    tracking_uri: str
    experiment_name: str
    experiment_id: str | None
    logged_model_ids: tuple[str, ...]
    run_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "scope": "evaluation_runs",
            "tracking_uri": self.tracking_uri,
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "logged_model_ids": list(self.logged_model_ids),
            "run_ids": list(self.run_ids),
            "deletion_order": ["logged_models", "runs"],
            "backend_gc_required": True,
            "backend_gc_guidance": (
                "After verification, run the MLflow backend-supported permanent cleanup/GC "
                "procedure for the configured tracking store."
            ),
        }


def _identity(value: object, field: str) -> str:
    result = str(value)
    if not result or result.strip() != result:
        raise ValueError(f"{field} must be non-empty and trimmed")
    return result


def collect_evaluation_manifest(
    client: CleanupClient,
    *,
    tracking_uri: str,
    experiment_name: str = EVALUATION_EXPERIMENT_NAME,
) -> EvaluationCleanupManifest:
    """Enumerate only the exact evaluation experiment and its objects."""

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return EvaluationCleanupManifest(tracking_uri, experiment_name, None, (), ())
    experiment_id = _identity(experiment.experiment_id, "experiment_id")
    runs = tuple(
        sorted(
            (_identity(run.info.run_id, "run_id") for run in client.search_runs([experiment_id])),
        )
    )
    models = tuple(
        sorted(
            _identity(model.model_id, "logged_model_id")
            for model in client.search_logged_models([experiment_id])
        )
    )
    return EvaluationCleanupManifest(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        experiment_id=experiment_id,
        logged_model_ids=models,
        run_ids=runs,
    )


def write_json(path: str | Path, payload: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return destination


def execute_evaluation_cleanup(
    client: Any,
    manifest: EvaluationCleanupManifest,
    *,
    expected_tracking_uri: str,
) -> dict[str, object]:
    """Delete exactly the manifest objects and verify no listed object remains."""

    if manifest.tracking_uri != expected_tracking_uri:
        raise ValueError("cleanup tracking URI does not match the explicit confirmation URI")
    for model_id in manifest.logged_model_ids:
        _delete_if_present(client.delete_logged_model, model_id)
    for run_id in manifest.run_ids:
        _delete_if_present(client.delete_run, run_id)
    remaining = collect_evaluation_manifest(
        client,
        tracking_uri=manifest.tracking_uri,
        experiment_name=manifest.experiment_name,
    )
    surviving_models = sorted(set(manifest.logged_model_ids) & set(remaining.logged_model_ids))
    surviving_runs = sorted(set(manifest.run_ids) & set(remaining.run_ids))
    proof: dict[str, object] = {
        **manifest.as_dict(),
        "executed": True,
        "surviving_logged_model_ids": surviving_models,
        "surviving_run_ids": surviving_runs,
        "status": "verified" if not surviving_models and not surviving_runs else "failed",
    }
    if proof["status"] != "verified":
        raise RuntimeError("MLflow evaluation cleanup left targeted objects behind")
    return proof


def _version_is_v4(version: Any) -> bool:
    tags = getattr(version, "tags", {}) or {}
    return (
        tags.get("regime_engine.package_schema") == V4_PACKAGE_SCHEMA
        and tags.get("regime_engine.profile_config_version") == "4"
    )


def _require_retired_version(version: Any, expected_version: str) -> None:
    actual = _identity(version.version, "model version")
    if actual != expected_version:
        raise ValueError("legacy inventory version does not match MLflow")
    if _version_is_v4(version):
        raise ValueError(f"refusing to delete accepted v4 model version {expected_version}")


def execute_retired_registered_model_cleanup(
    client: Any,
    inventory: dict[str, object],
    *,
    expected_tracking_uri: str,
) -> dict[str, object]:
    """Apply an explicit legacy inventory after proving a v4 champion exists."""

    if inventory.get("tracking_uri") != expected_tracking_uri:
        raise ValueError("legacy cleanup URI does not match explicit confirmation URI")
    if inventory.get("model_name") != REGISTERED_MODEL_NAME:
        raise ValueError("legacy cleanup is limited to regime-xetra")
    champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    if not _version_is_v4(champion):
        raise ValueError("destructive cleanup requires a v4 production champion")
    versions = inventory.get("versions", [])
    aliases = inventory.get("aliases", [])
    if not isinstance(versions, list) or not isinstance(aliases, list):
        raise ValueError("legacy inventory versions/aliases must be lists")
    plan: list[dict[str, str]] = []
    for item in versions:
        if not isinstance(item, dict) or not isinstance(item.get("version"), str):
            raise ValueError("legacy inventory contains an invalid model version")
        try:
            version = client.get_model_version(REGISTERED_MODEL_NAME, item["version"])
        except Exception as error:
            if _is_missing_error(error):
                continue
            raise
        _require_retired_version(version, item["version"])
        plan.append(
            {
                "operation": "delete_model_version",
                "model_name": REGISTERED_MODEL_NAME,
                "version": item["version"],
                "source": str(getattr(version, "source", "")),
            }
        )
    for item in aliases:
        if not isinstance(item, dict) or item.get("alias") not in ALLOWED_ALIASES:
            raise ValueError("legacy inventory contains an unsupported alias")
        alias = str(item["alias"])
        try:
            current = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
        except Exception as error:
            if _is_missing_error(error):
                continue
            raise
        current_version = _identity(current.version, "alias version")
        retarget = item.get("retarget_version")
        if isinstance(retarget, str) and current_version == retarget:
            # The manifest was already applied; make retries idempotent.
            continue
        if current_version != item.get("version"):
            raise ValueError(f"alias {alias} changed since inventory capture")
        if not isinstance(retarget, str) or not retarget:
            plan.append({"operation": "delete_alias", "alias": alias})
        else:
            target = client.get_model_version(REGISTERED_MODEL_NAME, retarget)
            if not _version_is_v4(target):
                raise ValueError(f"alias {alias} may only be retargeted to v4")
            plan.append(
                {
                    "operation": "retarget_alias",
                    "alias": alias,
                    "from_version": current_version,
                    "to_version": retarget,
                }
            )
    for operation in plan:
        if operation["operation"] == "delete_model_version":
            try:
                client.delete_model_version(REGISTERED_MODEL_NAME, operation["version"])
            except Exception as error:
                if not _is_missing_error(error):
                    raise
        elif operation["operation"] == "delete_alias":
            try:
                client.delete_registered_model_alias(REGISTERED_MODEL_NAME, operation["alias"])
            except Exception as error:
                if not _is_missing_error(error):
                    raise
        else:
            try:
                client.set_registered_model_alias(
                    REGISTERED_MODEL_NAME,
                    operation["alias"],
                    operation["to_version"],
                )
            except Exception as error:
                if not _is_missing_error(error):
                    raise
    survivors: list[str] = []
    for item in versions:
        try:
            client.get_model_version(REGISTERED_MODEL_NAME, item["version"])
        except Exception:
            continue
        survivors.append(item["version"])
    proof: dict[str, object] = {
        "scope": "retired_registered_models",
        "tracking_uri": expected_tracking_uri,
        "model_name": REGISTERED_MODEL_NAME,
        "plan": plan,
        "surviving_versions": sorted(survivors),
        "status": "verified" if not survivors else "failed",
    }
    if proof["status"] != "verified":
        raise RuntimeError("legacy registered-model cleanup left targeted versions behind")
    return proof


__all__ = [
    "ALLOWED_ALIASES",
    "EVALUATION_EXPERIMENT_NAME",
    "V4_PACKAGE_SCHEMA",
    "EvaluationCleanupManifest",
    "collect_evaluation_manifest",
    "execute_evaluation_cleanup",
    "execute_retired_registered_model_cleanup",
    "write_json",
]
