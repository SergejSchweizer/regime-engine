"""Run, track, and summarize one complete Xetra v4 evaluation snapshot."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import psycopg

from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.evaluation_runs.snapshot import ArrowDatasetSnapshotStore
from market_regime_engine.evaluation_runs.store import SQLiteEvaluationRunStore
from market_regime_engine.evaluation_statistics.writer import StatisticsWriter
from market_regime_engine.evaluations.global_regime_v4 import (
    V4ConfigurationSelection,
    evaluate_global_regime_v4,
    evaluate_global_regime_v4_from_source,
)
from market_regime_engine.features.ports import FeatureRequest
from market_regime_engine.features.postgres_settings import FeaturePostgresSettings
from market_regime_engine.features.postgres_source import PostgresFeatureSource
from market_regime_engine.mlflow_support.evaluation_tracking import (
    build_global_v4_evidence,
    track_global_v4_evaluation,
)
from market_regime_engine.mlflow_support.settings import MLflowSettings
from market_regime_engine.mlflow_support.tracking import FileMlflowTrackingPort
from market_regime_engine.profiles.loader import load_profile


def _root() -> Path:
    return Path(os.environ.get("REGIME_ENGINE_ROOT", Path(__file__).resolve().parents[1]))


def _commit(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _configured_checkpoint_root(root: Path) -> Path:
    """Require resumable state on an explicit persistent volume."""

    configured = os.environ.get("REGIME_EVALUATION_CHECKPOINT_ROOT") or os.environ.get(
        "REGIME_ENGINE_STATE_ROOT"
    )
    if not configured:
        raise RuntimeError(
            "REGIME_EVALUATION_CHECKPOINT_ROOT or REGIME_ENGINE_STATE_ROOT must be configured"
        )
    checkpoint_root = Path(configured).expanduser()
    if not checkpoint_root.is_absolute():
        raise RuntimeError("evaluation checkpoint root must be an absolute path")
    resolved_checkpoint_root = checkpoint_root.resolve()
    resolved_root = root.resolve()
    if (
        resolved_checkpoint_root == resolved_root
        or resolved_root in resolved_checkpoint_root.parents
    ):
        raise RuntimeError("evaluation checkpoint root must be outside the repository")
    return resolved_checkpoint_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-key",
        help="resume exactly this durable evaluation run without reading live PostgreSQL",
    )
    arguments = parser.parse_args()
    root = _root()
    profile = load_profile(root / "configs/profiles/xetra_v4.yaml")
    settings = FeaturePostgresSettings.from_env(os.environ)
    source = PostgresFeatureSource(
        lambda: cast(Any, psycopg.connect(**cast(Any, settings.connection_kwargs())))
    )
    observed: dict[str, Any] = {}

    class RecordingSource:
        def read_schema_wide_with_catalog(self, request: Any) -> Any:
            catalog, snapshot = source.read_schema_wide_with_catalog(request)
            observed["catalog"] = catalog
            observed["snapshot"] = snapshot
            return catalog, snapshot

    selections: dict[int, V4ConfigurationSelection] = {}
    checkpoint_root = _configured_checkpoint_root(root)
    commit = _commit(root)
    snapshot_store = ArrowDatasetSnapshotStore(checkpoint_root / "snapshots")
    run_store = SQLiteEvaluationRunStore(checkpoint_root / "runs")
    if arguments.run_key is None:
        result = evaluate_global_regime_v4_from_source(
            RecordingSource(),
            profile=profile,
            snapshot_store=snapshot_store,
            run_store=run_store,
            repository_commit_sha=commit,
            uv_lock_sha256=_sha256_file(root / "uv.lock"),
            python_version=platform.python_version(),
            selection_sink=selections.__setitem__,
        )
    else:
        run_identity = run_store.load_identity(arguments.run_key)
        if run_identity.evaluation_id != "global_regime_v4":
            raise RuntimeError("requested run key is not a global v4 evaluation")
        if run_identity.profile_hash != profile.profile_hash:
            raise RuntimeError("requested run key was created with a different profile")
        dataset_identity = snapshot_store.load_identity(run_identity.dataset_snapshot_key)
        snapshot = snapshot_store.load(dataset_identity)
        catalog = snapshot_store.load_catalog(dataset_identity)
        observed["catalog"] = catalog
        observed["snapshot"] = snapshot
        rows = pd.DataFrame(
            [row.values for row in snapshot.rows],
            columns=snapshot.feature_names,
        )
        rows.insert(0, "timestamp_m1", [row.timestamp for row in snapshot.rows])
        result = evaluate_global_regime_v4(
            rows,
            catalog=catalog,
            profile=profile,
            source_build_id=catalog.lineage.source_build_id,
            run_store=run_store,
            run_identity=run_identity,
            selection_sink=selections.__setitem__,
        )
    catalog = observed["catalog"]
    snapshot = observed["snapshot"]
    initial_source_identity = (
        catalog.lineage.source_build_id,
        catalog.lineage.data_sha256,
        catalog.catalog_hash,
        snapshot.materialized_feature_data_sha256,
        snapshot.rows[0].timestamp,
        snapshot.rows[-1].timestamp,
    )
    if arguments.run_key is None:
        audit_catalog, audit_snapshot = source.read_schema_wide_with_catalog(
            FeatureRequest.all_features()
        )
        audit_source_identity = (
            audit_catalog.lineage.source_build_id,
            audit_catalog.lineage.data_sha256,
            audit_catalog.catalog_hash,
            audit_snapshot.materialized_feature_data_sha256,
            audit_snapshot.rows[0].timestamp,
            audit_snapshot.rows[-1].timestamp,
        )
        if audit_source_identity != initial_source_identity:
            raise RuntimeError(
                "source build/data/catalog changed during evaluation; refusing to audit or track"
            )
    tracking_uri = MLflowSettings.from_environment().tracking_uri
    port = FileMlflowTrackingPort(tracking_uri, experiment_name="regime-engine-evaluation")
    evidence = build_global_v4_evidence(
        result,
        catalog=catalog,
        snapshot=snapshot,
        profile=profile,
        selections=selections,
        repository_commit_sha=commit,
    )
    tracked = track_global_v4_evaluation(
        port,
        StatisticsWriter(os.environ.get("REGIME_EVALUATION_STATISTICS_ROOT", ".state/evaluations")),
        evidence=evidence,
        result=result,
        selections=selections,
        metric_ledger_root=checkpoint_root / "metric-export",
    )
    outer_plan = plan_walk_forward(
        tuple(row.timestamp for row in snapshot.rows), profile.walk_forward
    )
    summary = {
        "evaluation_id": "global_regime_v4",
        "source_build_id": result.source_build_id,
        "source_data_sha256": catalog.lineage.data_sha256,
        "source_catalog_hash": catalog.catalog_hash,
        "profile_hash": profile.profile_hash,
        "repository_commit_sha": commit,
        "uv_lock_sha256": _sha256_file(root / "uv.lock"),
        "python_version": platform.python_version(),
        "source_row_count": len(snapshot.rows),
        "source_min_timestamp": snapshot.rows[0].timestamp.isoformat(),
        "source_max_timestamp": snapshot.rows[-1].timestamp.isoformat(),
        "outer_plan_hash": outer_plan.plan_hash,
        "outer_fold_count": len(result.outer_folds),
        "folds": [
            {
                "fold_index": fold.fold_index,
                "eligible_feature_count": len(selections[fold.fold_index].quality.eligible_features)
                if fold.fold_index in selections
                else None,
                "m_star": selections[fold.fold_index].clusters.selected_count
                if fold.fold_index in selections
                else None,
                "l_star": selections[fold.fold_index].prefix_search.selected_prefix_length
                if fold.fold_index in selections
                else None,
                "candidate_id": fold.final_configuration.candidate_id,
                "state_count": fold.final_configuration.state_count,
                "outer_soft_nmi": fold.outer_teacher_final_soft_nmi,
                "outer_oos_pll_per_observation": fold.oos_predictive_loglik_per_observation,
                "valid": fold.valid,
            }
            for fold in result.outer_folds
        ],
        "valid_fold_rate": result.valid_fold_rate,
        "production_eligible": result.production_eligible,
        "mlflow_parent_run_id": tracked.parent_run_id,
        "evidence_hash": tracked.global_evidence_hash,
    }
    summary_path = os.environ.get("REGIME_EVALUATION_SUMMARY_PATH")
    if summary_path:
        destination = Path(summary_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
