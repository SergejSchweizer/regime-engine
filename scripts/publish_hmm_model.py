#!/usr/bin/env python3
"""Upload and register one final-refit HMM package in NAS MLflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from market_regime_engine.mlflow_support.model_package import load_production_package
from market_regime_engine.mlflow_support.model_publishing import publish_production_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_directory", type=Path)
    args = parser.parse_args()
    artifact = load_production_package(args.package_directory)
    published = publish_production_package(artifact, args.package_directory)
    print(
        {
            "tracking_uri": published.tracking_uri,
            "run_id": published.run_id,
            "model_name": published.registered.model_name,
            "exact_version": published.registered.exact_version,
            "package_uri": published.registered.package_uri,
        }
    )


if __name__ == "__main__":
    main()
