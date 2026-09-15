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
        tags = dict(getattr(model_version, "tags", {}) or {})
        expected_k = alias.rsplit("-k", 1)[1]
        if tags.get("regime_engine.state_count") != expected_k:
            raise RuntimeError(f"{alias} points to a model with mismatched state count")
        if tags.get("regime_engine.alias") not in {alias, None}:
            raise RuntimeError(f"{alias} target has a mismatched alias tag")
        checked[alias] = {
            "status": "present",
            "version": str(model_version.version),
            "state_count": expected_k,
            "artifact_sha256": tags.get("regime_engine.artifact_sha256", ""),
        }
    result["aliases"] = checked
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
