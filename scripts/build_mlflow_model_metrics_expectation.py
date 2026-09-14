#!/usr/bin/env python3
"""Build a strict MLflow Model Metrics expectation from independent evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verify_mlflow_model_metrics import build_expectation_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="independent JSON evidence manifest; it must not be read from MLflow",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        parser.error("source evidence must be a JSON object")
    bundle = build_expectation_bundle(source, source_artifact_path=args.source)
    rendered = json.dumps(bundle, sort_keys=True, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(bundle["provenance"]["evidence_sha256"])


if __name__ == "__main__":
    main()
