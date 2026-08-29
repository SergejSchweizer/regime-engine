"""Exit 0 when an equivalent Xetra V3 evaluation already completed in MLflow.

Exit 3 means that no completed equivalent batch exists and the caller should run it.
Any other non-zero exit is a fail-closed operational error.
"""

from __future__ import annotations

import json
import os
import subprocess

import psycopg
from mlflow.tracking import MlflowClient
from psycopg import IsolationLevel

from market_regime_engine.features.postgres_settings import FeaturePostgresSettings
from market_regime_engine.mlflow_support.evaluation_dedup import (
    completed_xetra_v3_evaluation_exists,
    xetra_v3_evaluation_fingerprint,
)


def _git_commit(root: str) -> str:
    return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()


def _source_data_sha256(settings: FeaturePostgresSettings) -> str:
    with psycopg.connect(**settings.connection_kwargs()) as connection:
        connection.read_only = True
        connection.isolation_level = IsolationLevel.REPEATABLE_READ
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT data_sha256 FROM regime_loader_sync.gold_sync_state WHERE dataset_id = %s",
                ("regime_features_daily",),
            )
            row = cursor.fetchone()
    if row is None or not isinstance(row[0], str) or not row[0]:
        raise ValueError("missing data_sha256 for regime_features_daily")
    return row[0]


def main() -> int:
    root = os.environ["REGIME_ENGINE_ROOT"]
    git_commit = _git_commit(root)
    data_sha256 = _source_data_sha256(FeaturePostgresSettings.from_env(os.environ))
    fingerprint = xetra_v3_evaluation_fingerprint(git_commit=git_commit, data_sha256=data_sha256)
    exists = completed_xetra_v3_evaluation_exists(
        MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"]), fingerprint
    )
    print(
        json.dumps(
            {
                "data_sha256": data_sha256,
                "decision": "skip" if exists else "run",
                "fingerprint": fingerprint,
                "git_commit": git_commit,
            },
            sort_keys=True,
        )
    )
    return 0 if exists else 3


if __name__ == "__main__":
    raise SystemExit(main())
