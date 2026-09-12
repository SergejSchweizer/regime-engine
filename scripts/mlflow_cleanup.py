#!/usr/bin/env python3
"""Plan or explicitly execute narrowly scoped MLflow cleanup operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlflow.tracking import MlflowClient

from market_regime_engine.mlflow_support.cleanup import (
    EVALUATION_EXPERIMENT_NAME,
    collect_evaluation_manifest,
    execute_evaluation_cleanup,
    execute_retired_registered_model_cleanup,
    require_production_tracking_uri,
    write_json,
)
from market_regime_engine.mlflow_support.settings import MLflowSettings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("evaluation-runs", "retired-registered-models"),
        required=True,
    )
    parser.add_argument("--tracking-uri", default=MLflowSettings.from_environment().tracking_uri)
    parser.add_argument("--confirm-tracking-uri")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--inventory", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/qa/mlflow_cleanup_report.json"),
    )
    args = parser.parse_args()
    try:
        require_production_tracking_uri(args.tracking_uri)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    client = MlflowClient(tracking_uri=args.tracking_uri, registry_uri=args.tracking_uri)
    if args.scope == "evaluation-runs":
        manifest = collect_evaluation_manifest(
            client,
            tracking_uri=args.tracking_uri,
            experiment_name=EVALUATION_EXPERIMENT_NAME,
        )
        report: dict[str, object] = {**manifest.as_dict(), "executed": False, "status": "planned"}
        if args.execute:
            if args.confirm_tracking_uri != args.tracking_uri:
                raise SystemExit("--execute requires an exact --confirm-tracking-uri match")
            report = execute_evaluation_cleanup(
                client,
                manifest,
                expected_tracking_uri=args.confirm_tracking_uri,
            )
    else:
        if args.inventory is None:
            raise SystemExit("--inventory is required for retired-registered-model cleanup")
        inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
        if not isinstance(inventory, dict):
            raise SystemExit("retired inventory must be a JSON object")
        report = {
            **inventory,
            "scope": "retired_registered_models",
            "executed": False,
            "status": "planned",
        }
        if args.execute:
            if args.confirm_tracking_uri != args.tracking_uri:
                raise SystemExit("--execute requires an exact --confirm-tracking-uri match")
            report = execute_retired_registered_model_cleanup(
                client,
                inventory,
                expected_tracking_uri=args.confirm_tracking_uri,
            )
    write_json(args.report, report)
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
