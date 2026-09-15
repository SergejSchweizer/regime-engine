#!/usr/bin/env python3
"""Read-only external MLflow acceptance check for the four K-slot aliases.

This command is intentionally manual/scheduled-only.  It never mutates an
external registry unless a separate publisher explicitly uses the registry
API after this read-back succeeds.  Ordinary CI should not invoke it.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from typing import Any

from mlflow.tracking import MlflowClient

from market_regime_engine.contracts import K_CHAMPION_ALIASES

_SLOT_TAGS = (
    "regime_engine.slot_id",
    "regime_engine.alias",
    "regime_engine.state_count",
    "regime_engine.model_family",
    "regime_engine.policy_id",
    "regime_engine.policy_version",
    "regime_engine.source_snapshot_id",
    "regime_engine.feature_order_sha256",
    "regime_engine.comparison_domain_id",
    "regime_engine.promotion_score_version",
    "regime_engine.reference_teacher_id",
    "regime_engine.validation_cutoff",
    "regime_engine.deployment_cutoff",
    "regime_engine.selection_sha256",
    "regime_engine.artifact_sha256",
    "regime_engine.idempotency_key",
)
_HASH_TAGS = {
    "regime_engine.feature_order_sha256",
    "regime_engine.selection_sha256",
    "regime_engine.artifact_sha256",
    "regime_engine.idempotency_key",
}


def _tag_dict(model_version: Any) -> dict[str, str]:
    raw = getattr(model_version, "tags", {}) or {}
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    return {str(item.key): str(item.value) for item in raw}


def _is_sha256(value: str) -> bool:
    return (
        len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def inspect_external_registry(
    client: Any,
    *,
    model_name: str = "regime-xetra",
    expected_default_champion: str | None = None,
) -> dict[str, object]:
    """Read aliases and versions, failing closed on cross-K targets."""

    registered = client.get_registered_model(model_name)
    aliases = dict(getattr(registered, "aliases", {}) or {})
    result: dict[str, object] = {
        "checked_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model_name": model_name,
        "aliases": {},
        "default_champion": aliases.get("champion"),
    }
    if (
        expected_default_champion is not None
        and aliases.get("champion") != expected_default_champion
    ):
        raise RuntimeError("external acceptance would observe a changed legacy champion alias")
    checked: dict[str, dict[str, str]] = {}
    for alias in sorted(K_CHAMPION_ALIASES):
        version = aliases.get(alias)
        if version is None:
            checked[alias] = {"status": "absent"}
            continue
        model_version = client.get_model_version_by_alias(model_name, alias)
        version_id = str(version)
        if str(getattr(model_version, "version", "")) != version_id:
            raise RuntimeError(f"{alias} alias readback returned a different model version")
        tags = _tag_dict(model_version)
        expected_k = alias.rsplit("-k", 1)[1]
        if tags.get("regime_engine.state_count") != expected_k:
            raise RuntimeError(f"{alias} points to a model with mismatched state count")
        if tags.get("regime_engine.slot_id") != f"k{expected_k}":
            raise RuntimeError(f"{alias} target has a mismatched slot tag")
        if tags.get("regime_engine.alias") != alias:
            raise RuntimeError(f"{alias} target has a mismatched alias tag")
        missing = tuple(key for key in _SLOT_TAGS if not tags.get(key))
        if missing:
            raise RuntimeError(f"{alias} target is missing required tags: {', '.join(missing)}")
        invalid_hashes = tuple(key for key in _HASH_TAGS if not _is_sha256(tags[key]))
        if invalid_hashes:
            raise RuntimeError(f"{alias} target has invalid hash tags: {', '.join(invalid_hashes)}")
        checked[alias] = {
            "status": "present",
            "version": version_id,
            "state_count": expected_k,
            "model_family": tags["regime_engine.model_family"],
            "source_snapshot_id": tags["regime_engine.source_snapshot_id"],
            "feature_order_sha256": tags["regime_engine.feature_order_sha256"],
            "selection_sha256": tags["regime_engine.selection_sha256"],
            "artifact_sha256": tags["regime_engine.artifact_sha256"],
            "idempotency_key": tags["regime_engine.idempotency_key"],
        }
    result["aliases"] = checked
    present = [item for item in checked.values() if item["status"] == "present"]
    for key in (
        "regime_engine.source_snapshot_id",
        "regime_engine.policy_id",
        "regime_engine.policy_version",
        "regime_engine.comparison_domain_id",
    ):
        values = {
            tags[key]
            for alias, item in checked.items()
            if item["status"] == "present"
            for tags in (_tag_dict(client.get_model_version_by_alias(model_name, alias)),)
        }
        if len(values) > 1:
            field = key.removeprefix("regime_engine.")
            raise RuntimeError(f"K-slot aliases have incompatible {field} lineage")
    result["present_slot_count"] = len(present)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--model-name", default="regime-xetra")
    parser.add_argument("--expected-default-champion")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    if not args.tracking_uri:
        parser.error("--tracking-uri or MLFLOW_TRACKING_URI is required")
    client = MlflowClient(tracking_uri=args.tracking_uri, registry_uri=args.tracking_uri)
    report = inspect_external_registry(
        client,
        model_name=args.model_name,
        expected_default_champion=args.expected_default_champion,
    )
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
