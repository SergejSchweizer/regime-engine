"""Run, track, and summarize one complete, non-resumable Xetra v4 evaluation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import psycopg

from market_regime_engine.evaluation.walk_forward import WalkForwardEvaluation
from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.evaluation_runs.contracts import (
    DatasetSnapshotIdentity,
    EvaluationRunIdentity,
)
from market_regime_engine.evaluation_runs.math_audit import build_math_expectations
from market_regime_engine.evaluation_runs.snapshot import ArrowDatasetSnapshotStore
from market_regime_engine.evaluation_statistics.writer import StatisticsWriter
from market_regime_engine.evaluations.global_regime_v4 import (
    V4ConfigurationSelection,
    evaluate_global_regime_v4,
    evaluate_global_regime_v4_from_source,
)
from market_regime_engine.feature_discovery.contracts import AdaptiveEvaluationResult
from market_regime_engine.features.ports import FeatureRequest, SourceMode
from market_regime_engine.features.postgres_settings import FeaturePostgresSettings
from market_regime_engine.features.postgres_source import PostgresFeatureSource
from market_regime_engine.mlflow_support.evaluation_tracking import (
    build_global_v4_evidence,
    track_global_v4_evaluation,
)
from market_regime_engine.mlflow_support.settings import MLflowSettings
from market_regime_engine.mlflow_support.tracking import FileMlflowTrackingPort
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.runtime.cpu import available_cpu_count, cpu_worker_count
from market_regime_engine.runtime.performance import PerformanceRecorder

_CURRENT_AUDIT_SCHEMA_VERSION = 2


def _root() -> Path:
    return Path(os.environ.get("REGIME_ENGINE_ROOT", Path(__file__).resolve().parents[1]))


def _commit(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(payload).hexdigest()


def _durable_evidence_path(root: Path, environment_name: str) -> Path:
    configured = os.environ.get(environment_name)
    if not configured:
        raise RuntimeError(f"{environment_name} must point to durable audit evidence")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{environment_name} must be an absolute path")
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved == resolved_root or resolved_root in resolved.parents:
        raise RuntimeError(f"{environment_name} must be outside the repository")
    return resolved


def _source_request_evidence(request: FeatureRequest) -> dict[str, object]:
    """Require the current audit to use the complete unbounded source request."""

    if request.mode is not SourceMode.SCHEMA_DISCOVERY:
        raise RuntimeError("current Xetra audit requires schema-wide source discovery")
    if request.feature_names:
        raise RuntimeError("current Xetra audit cannot use a feature allowlist")
    if request.start is not None or request.end is not None:
        raise RuntimeError("current Xetra audit cannot use source timestamp bounds")
    return {
        "mode": request.mode.value,
        "feature_names": [],
        "start": None,
        "end": None,
        "all_source_rows": True,
        "dynamic_catalog": True,
    }


def _source_bounds(catalog: Any, snapshot: Any) -> dict[str, object]:
    lineage = catalog.lineage
    if lineage.row_count is None or lineage.min_timestamp is None or lineage.max_timestamp is None:
        raise RuntimeError("current Xetra audit requires complete source lineage bounds")
    return {
        "source_dataset": lineage.source_dataset,
        "source_table": lineage.source_table,
        "source_row_count": lineage.row_count,
        "source_min_timestamp": lineage.min_timestamp.isoformat(),
        "source_max_timestamp": lineage.max_timestamp.isoformat(),
        "feature_count": len(catalog.feature_names),
        "feature_names_sha256": _sha256_json(list(catalog.feature_names)),
        "materialized_row_count": len(snapshot.rows),
        "materialized_min_timestamp": (
            None if not snapshot.rows else snapshot.rows[0].timestamp.isoformat()
        ),
        "materialized_max_timestamp": (
            None if not snapshot.rows else snapshot.rows[-1].timestamp.isoformat()
        ),
        "skipped_incomplete_row_count": snapshot.skipped_incomplete_row_count,
    }


def _search_bounds(profile: Any) -> dict[str, object]:
    canonical = profile.canonical_dict()
    discovery = cast(dict[str, object], canonical["feature_discovery"])
    walk_forward = cast(dict[str, object], canonical["walk_forward"])
    return {
        "feature_discovery": discovery,
        "walk_forward": walk_forward,
        "candidate_families": {
            "gaussian_state_counts": list(profile.gaussian_hmm.candidate_states),
            "gaussian_multistart_seeds": list(profile.gaussian_hmm.seeds),
            "gmm_state_mixture_pairs": [
                [candidate.state_count, candidate.mixture_count] for candidate in profile.gmm_hmms
            ],
            "student_t_state_counts": list(profile.student_t_hmm.candidate_states),
            "final_candidate_ids": list(profile.feature_discovery.final_candidate_ids),
        },
    }


def _selection_search_evidence(
    profile: Any,
    selections: dict[int, V4ConfigurationSelection],
    fold_indices: tuple[int, ...],
) -> dict[str, object]:
    """Validate and expose the exact bounded search actually executed per fold."""

    per_fold: list[dict[str, object]] = []
    for fold_index in fold_indices:
        selection = selections[fold_index]
        feature_count = len(selection.distance.feature_order)
        expected_cluster_counts = tuple(
            range(
                profile.feature_discovery.cluster_count_min,
                min(profile.feature_discovery.cluster_count_max, feature_count - 1) + 1,
            )
        )
        actual_cluster_counts = tuple(
            count for count, _memberships in selection.clusters.candidate_memberships
        )
        if actual_cluster_counts != expected_cluster_counts:
            raise RuntimeError(
                f"outer fold {fold_index} did not evaluate every bounded cluster count: "
                f"expected={expected_cluster_counts}, actual={actual_cluster_counts}"
            )
        ranked_count = len(selection.prefix_search.ranked_features)
        expected_prefix_lengths = tuple(
            range(
                profile.feature_discovery.minimum_prefix_length,
                min(profile.feature_discovery.maximum_prefix_length, ranked_count) + 1,
            )
        )
        actual_prefix_lengths = tuple(
            item.prefix_length for item in selection.prefix_search.evaluations
        )
        if actual_prefix_lengths != expected_prefix_lengths:
            raise RuntimeError(
                f"outer fold {fold_index} did not evaluate every bounded prefix: "
                f"expected={expected_prefix_lengths}, actual={actual_prefix_lengths}"
            )
        final_ids = tuple(item.candidate_id for item in selection.final_grid.grid.evaluations)
        expected_final_ids = tuple(profile.feature_discovery.final_candidate_ids)
        if final_ids != expected_final_ids:
            raise RuntimeError(
                f"outer fold {fold_index} final candidate grid is incomplete or reordered"
            )
        teacher_ids = tuple(
            item.candidate_id for item in selection.teacher_evaluation.candidate_evaluations
        )
        expected_teacher_ids = tuple(
            f"gaussian_hmm_k{state_count}_full"
            for state_count in profile.feature_discovery.provisional_state_counts
        )
        if teacher_ids != expected_teacher_ids:
            raise RuntimeError(f"outer fold {fold_index} provisional teacher grid is incomplete")
        per_fold.append(
            {
                "outer_fold_index": fold_index,
                "eligible_feature_count": feature_count,
                "eligible_feature_names": list(selection.distance.feature_order),
                "ranked_feature_count": ranked_count,
                "cluster_count_candidates": list(actual_cluster_counts),
                "prefix_length_candidates": list(actual_prefix_lengths),
                "provisional_candidate_ids": list(teacher_ids),
                "final_candidate_ids": list(final_ids),
            }
        )
    return {
        "declared_bounds": _search_bounds(profile),
        "per_fold": per_fold,
    }


def _require_current_audit_eligibility(
    result: AdaptiveEvaluationResult,
    selections: dict[int, V4ConfigurationSelection],
    profile: Any,
) -> tuple[int, ...]:
    valid_indices = tuple(sorted(fold.fold_index for fold in result.outer_folds if fold.valid))
    minimum = profile.feature_discovery.minimum_outer_valid_folds
    if len(valid_indices) < minimum:
        raise RuntimeError(
            f"current Xetra audit requires at least {minimum} valid outer folds; "
            f"actual={valid_indices}"
        )
    if len(valid_indices) != len(result.outer_folds):
        raise RuntimeError(
            "current Xetra audit requires every planned outer fold to be valid; "
            f"valid={valid_indices}, planned={len(result.outer_folds)}"
        )
    if tuple(sorted(selections)) != valid_indices:
        raise RuntimeError("current Xetra audit selections do not cover every valid outer fold")
    if not result.production_eligible:
        raise RuntimeError("current Xetra audit requires a production-eligible outer policy")
    return valid_indices


def _configured_checkpoint_root(root: Path) -> Path:
    """Require evaluation artifacts on an explicit persistent volume."""

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


def _tracking_worker_count(task_count: int) -> int:
    """Resolve the same bounded tracking budget used by the tracker itself."""

    configured = os.environ.get("REGIME_TRACKING_WORKERS")
    return cpu_worker_count(
        int(configured) if configured else None,
        task_count=task_count,
    )


def _require_full_audit_eligibility(
    result: AdaptiveEvaluationResult,
    selections: dict[int, V4ConfigurationSelection],
) -> None:
    """Fail closed unless every valid outer fold has a complete policy selection."""

    valid_fold_indices = tuple(sorted(fold.fold_index for fold in result.outer_folds if fold.valid))
    selected_fold_indices = tuple(sorted(selections))
    if not valid_fold_indices:
        raise RuntimeError("full Xetra v4 audit requires at least one valid outer fold")
    if selected_fold_indices != valid_fold_indices:
        raise RuntimeError(
            "full Xetra v4 audit requires one policy selection for every valid outer fold; "
            f"valid={valid_fold_indices}, selections={selected_fold_indices}"
        )
    if not result.production_eligible:
        raise RuntimeError("full Xetra v4 audit requires a production-eligible outer-policy result")


def _run(performance: PerformanceRecorder) -> None:
    evaluation_started_at = datetime.now(UTC)
    evaluation_start_monotonic = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-root",
        help="recompute from an existing immutable snapshot root (benchmark mode)",
    )
    parser.add_argument(
        "--snapshot-key",
        help="dataset snapshot key to use with --snapshot-root",
    )
    arguments = parser.parse_args()
    if arguments.snapshot_root is not None and arguments.snapshot_key is None:
        parser.error("--snapshot-key is required with --snapshot-root")
    root = _root()
    profile = load_profile(root / "configs/profiles/xetra_v4.yaml")
    summary_path = _durable_evidence_path(root, "REGIME_EVALUATION_SUMMARY_PATH")
    performance_report_path = _durable_evidence_path(root, "REGIME_PERFORMANCE_REPORT_PATH")
    source: PostgresFeatureSource | None = None
    if arguments.snapshot_root is None:
        settings = FeaturePostgresSettings.from_env(os.environ)
        source = PostgresFeatureSource(
            lambda: cast(Any, psycopg.connect(**cast(Any, settings.connection_kwargs())))
        )
    observed: dict[str, Any] = {}
    run_identity: EvaluationRunIdentity | None = None

    class RecordingSource:
        def read_schema_wide_with_catalog(self, request: Any) -> Any:
            if source is None:
                raise RuntimeError("live source is not configured")
            observed["source_request"] = _source_request_evidence(request)
            with performance.stage("postgres_snapshot", worker_count=1, task_count=1):
                catalog, snapshot = source.read_schema_wide_with_catalog(request)
            observed["catalog"] = catalog
            observed["snapshot"] = snapshot
            return catalog, snapshot

    selections: dict[int, V4ConfigurationSelection] = {}
    prefix_evaluations: dict[int, dict[tuple[int, str], WalkForwardEvaluation]] = {}

    def record_prefix_evaluation(
        outer_fold_index: int,
        prefix_length: int,
        candidate_id: str,
        evaluation: WalkForwardEvaluation,
    ) -> None:
        prefix_evaluations.setdefault(outer_fold_index, {})[prefix_length, candidate_id] = (
            evaluation
        )

    checkpoint_root = _configured_checkpoint_root(root)
    commit = _commit(root)
    snapshot_store = ArrowDatasetSnapshotStore(checkpoint_root / "snapshots")
    if arguments.snapshot_root is not None:
        assert arguments.snapshot_key is not None
        input_store = ArrowDatasetSnapshotStore(arguments.snapshot_root)
        dataset_identity = input_store.load_identity(arguments.snapshot_key)
        snapshot = input_store.load(dataset_identity)
        catalog = input_store.load_catalog(dataset_identity)
        observed["catalog"] = catalog
        observed["snapshot"] = snapshot
        rows = pd.DataFrame(
            [row.values for row in snapshot.rows],
            columns=snapshot.feature_names,
        )
        rows.insert(0, "timestamp_m1", [row.timestamp for row in snapshot.rows])
        outer_plan = plan_walk_forward(
            tuple(row.timestamp for row in snapshot.rows), profile.walk_forward
        )
        run_identity = EvaluationRunIdentity(
            evaluation_id="global_regime_v4",
            profile_id=profile.profile_id,
            profile_config_version=profile.profile_config_version,
            profile_hash=profile.profile_hash,
            evaluation_contract_version=1,
            evaluation_plan_hash=outer_plan.plan_hash,
            dataset_snapshot_key=dataset_identity.key,
            evaluation_cutoff=cast(datetime, outer_plan.evaluation_cutoff),
            repository_commit_sha=commit,
            uv_lock_sha256=_sha256_file(root / "uv.lock"),
            python_version=platform.python_version(),
        )
        with performance.stage(
            "statistical_evaluation_snapshot",
            worker_count=available_cpu_count(),
            task_count=len(outer_plan.folds),
        ):
            result = evaluate_global_regime_v4(
                rows,
                catalog=catalog,
                profile=profile,
                source_build_id=catalog.lineage.source_build_id,
                selection_sink=selections.__setitem__,
                prefix_evaluation_sink=record_prefix_evaluation,
            )
    else:
        assert source is not None
        with performance.stage(
            "statistical_evaluation",
            worker_count=available_cpu_count(),
        ):
            result = evaluate_global_regime_v4_from_source(
                RecordingSource(),
                profile=profile,
                snapshot_store=snapshot_store,
                repository_commit_sha=commit,
                uv_lock_sha256=_sha256_file(root / "uv.lock"),
                python_version=platform.python_version(),
                selection_sink=selections.__setitem__,
                prefix_evaluation_sink=record_prefix_evaluation,
                require_production_eligible_source_clock=True,
            )
    valid_outer_fold_indices = _require_current_audit_eligibility(result, selections, profile)
    selection_search_evidence = _selection_search_evidence(
        profile,
        selections,
        valid_outer_fold_indices,
    )
    if run_identity is None:
        catalog = observed["catalog"]
        snapshot = observed["snapshot"]
        dataset_identity = DatasetSnapshotIdentity.from_catalog(catalog)
        outer_plan = plan_walk_forward(
            tuple(row.timestamp for row in snapshot.rows), profile.walk_forward
        )
        run_identity = EvaluationRunIdentity(
            evaluation_id="global_regime_v4",
            profile_id=profile.profile_id,
            profile_config_version=profile.profile_config_version,
            profile_hash=profile.profile_hash,
            evaluation_contract_version=1,
            evaluation_plan_hash=outer_plan.plan_hash,
            dataset_snapshot_key=dataset_identity.key,
            evaluation_cutoff=cast(datetime, outer_plan.evaluation_cutoff),
            repository_commit_sha=commit,
            uv_lock_sha256=_sha256_file(root / "uv.lock"),
            python_version=platform.python_version(),
        )
    performance.update_metadata(
        {
            "evaluation_id": "global_regime_v4",
            "outer_fold_count": len(result.outer_folds),
            "valid_fold_count": result.valid_fold_count,
            "selection_count": len(selections),
        }
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
    initial_source_request = cast(
        dict[str, object],
        observed.get(
            "source_request",
            {
                "mode": "immutable_snapshot_replay",
                "feature_names": [],
                "start": None,
                "end": None,
                "all_source_rows": True,
                "dynamic_catalog": True,
            },
        ),
    )
    audit_request_evidence: dict[str, object] | None = None
    if arguments.snapshot_root is None:
        assert source is not None
        audit_request_evidence = _source_request_evidence(FeatureRequest.all_features())
        with performance.stage("postgres_audit", worker_count=1, task_count=1):
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
        if audit_request_evidence != initial_source_request:
            raise RuntimeError("source request bounds changed between evaluation and audit")
    audit_root = Path(
        os.environ.get("REGIME_EVALUATION_AUDIT_ROOT", str(checkpoint_root / "audit"))
    )
    audit_root.mkdir(mode=0o750, parents=True, exist_ok=True)
    assert run_identity is not None
    audit_rows = pd.DataFrame(
        [row.values for row in snapshot.rows],
        columns=snapshot.feature_names,
    )
    audit_rows.insert(0, "timestamp_m1", [row.timestamp for row in snapshot.rows])
    snapshot_root = (
        Path(arguments.snapshot_root)
        if arguments.snapshot_root is not None
        else checkpoint_root / "snapshots"
    )
    snapshot_path = snapshot_root / run_identity.dataset_snapshot_key / "snapshot.arrow"
    source_bounds = _source_bounds(catalog, snapshot)
    identity_hashes = {
        "source_build_id": catalog.lineage.source_build_id,
        "source_data_sha256": catalog.lineage.data_sha256,
        "source_catalog_hash": catalog.catalog_hash,
        "materialized_feature_data_sha256": snapshot.materialized_feature_data_sha256,
        "dataset_snapshot_key": run_identity.dataset_snapshot_key,
        "snapshot_sha256": _sha256_file(snapshot_path),
        "profile_hash": profile.profile_hash,
        "outer_plan_hash": run_identity.evaluation_plan_hash,
        "repository_commit_sha": commit,
        "uv_lock_sha256": run_identity.uv_lock_sha256,
    }
    if any(not isinstance(value, str) or not value for value in identity_hashes.values()):
        raise RuntimeError("current Xetra audit identity hashes are incomplete")
    valid_audit_indices = tuple(
        dict.fromkeys(
            (
                valid_outer_fold_indices[0],
                valid_outer_fold_indices[len(valid_outer_fold_indices) // 2],
                valid_outer_fold_indices[-1],
            )
        )
    )
    raw_feature_names = tuple(
        name for name in catalog.feature_names if not name.startswith("pca_pc_")
    )
    generated_pca_names = tuple(
        name for name in catalog.feature_names if name.startswith("pca_pc_")
    )
    audit_contract: dict[str, object] = {
        "schema_version": _CURRENT_AUDIT_SCHEMA_VERSION,
        "source_request": initial_source_request,
        "audit_source_request": audit_request_evidence,
        "source_bounds": source_bounds,
        "search_bounds": selection_search_evidence,
        "identity_hashes": identity_hashes,
        "pca": {
            # PCA is mandatory in the canonical Xetra v4 profile.
            "mandatory": True,
            "variance_threshold": profile.pca.variance_threshold,
            "component_count": profile.pca.component_count,
            "raw_feature_names": list(raw_feature_names),
            "generated_feature_names": list(generated_pca_names),
            "raw_feature_names_sha256": _sha256_json(list(raw_feature_names)),
            "generated_feature_names_sha256": _sha256_json(list(generated_pca_names)),
            "universe_feature_names_sha256": _sha256_json(list(catalog.feature_names)),
        },
        "valid_outer_fold_indices": list(valid_outer_fold_indices),
        "audit_outer_fold_indices": list(valid_audit_indices),
        "outer_fold_result_hashes": [
            fold.result_hash
            for fold in sorted(result.outer_folds, key=lambda item: item.fold_index)
        ],
        "resource_evidence": {
            "performance_report_path": str(performance_report_path),
            "available_logical_cpus": available_cpu_count(),
            "requested_cpu_workers": os.environ.get("REGIME_CPU_WORKERS"),
            "requested_tracking_workers": os.environ.get("REGIME_TRACKING_WORKERS"),
            "native_thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
                if os.environ.get(name) is not None
            },
        },
    }
    expectations = build_math_expectations(
        audit_rows,
        result,
        selections,
        prefix_evaluations=prefix_evaluations,
        max_workers=cpu_worker_count(None),
        source_identity={
            "source_build_id": audit_catalog.lineage.source_build_id,
            "source_data_sha256": audit_catalog.lineage.data_sha256,
            "source_catalog_hash": audit_catalog.catalog_hash,
            "dataset_snapshot_key": run_identity.dataset_snapshot_key,
            "snapshot_sha256": sha256(snapshot_path.read_bytes()).hexdigest(),
        },
    )
    expectations["schema_version"] = 3
    expectations["audit_contract"] = audit_contract
    expectations["valid_outer_fold_indices"] = list(valid_outer_fold_indices)
    expectations_path = audit_root / f"{run_identity.key}.json"
    expectations_path.write_text(
        json.dumps(expectations, default=str, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    audit_output = subprocess.check_output(
        [
            str(root / ".venv/bin/python"),
            str(root / "scripts/verify_xetra_v4_math.py"),
            "--snapshot",
            str(snapshot_path),
            "--expectations",
            str(expectations_path),
            "--workers",
            str(cpu_worker_count(None)),
        ]
        + (["--require-current-xetra-contract"] if arguments.snapshot_root is None else []),
        text=True,
    )
    audit_report_path = audit_root / f"{run_identity.key}.report.json"
    audit_report_path.write_text(audit_output, encoding="utf-8")
    performance.update_metadata(
        {
            "math_audit_expectations": str(expectations_path),
            "math_audit_report": str(audit_report_path),
        }
    )
    tracking_uri = MLflowSettings.from_environment().tracking_uri
    port = FileMlflowTrackingPort(tracking_uri, experiment_name="macro-regime-evaluation")
    evidence_workers = cpu_worker_count(None, task_count=len(result.outer_folds))
    with performance.stage(
        "evidence_assembly",
        worker_count=evidence_workers,
        task_count=len(result.outer_folds),
    ):
        evidence = build_global_v4_evidence(
            result,
            catalog=catalog,
            snapshot=snapshot,
            profile=profile,
            selections=selections,
            repository_commit_sha=commit,
            max_workers=evidence_workers,
        )
    tracking_workers = _tracking_worker_count(len(result.outer_folds))
    with performance.stage(
        "mlflow_tracking",
        worker_count=tracking_workers,
        task_count=len(result.outer_folds),
    ):
        tracked = track_global_v4_evaluation(
            port,
            StatisticsWriter(
                os.environ.get("REGIME_EVALUATION_STATISTICS_ROOT", ".state/evaluations")
            ),
            evidence=evidence,
            result=result,
            selections=selections,
            metric_ledger_root=checkpoint_root / "metric-export",
            evaluation_started_at=evaluation_started_at,
            evaluation_start_monotonic=evaluation_start_monotonic,
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
                "feature_count": (
                    len(selections[fold.fold_index].distance.feature_order)
                    if fold.fold_index in selections
                    else None
                ),
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
                "model_family": fold.final_configuration.model_family,
                "final_features": list(fold.final_configuration.feature_order),
                "state_count": fold.final_configuration.state_count,
                "train_start": fold.train_start.isoformat(),
                "train_end": fold.train_end.isoformat(),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "train_source_observations": outer_plan.folds[
                    fold.fold_index - 1
                ].train_source_observations,
                "test_source_observations": outer_plan.folds[
                    fold.fold_index - 1
                ].test_source_observations,
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
        "math_audit_expectations": str(expectations_path),
        "math_audit_report": str(audit_report_path),
        "audit_contract": audit_contract,
        "source_request": initial_source_request,
        "source_bounds": source_bounds,
        "search_bounds": selection_search_evidence,
        "identity_hashes": identity_hashes,
        "valid_outer_fold_indices": list(valid_outer_fold_indices),
        "audit_outer_fold_indices": list(valid_audit_indices),
        "resource_evidence": audit_contract["resource_evidence"],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    performance = PerformanceRecorder.from_environment()
    try:
        _run(performance)
    except BaseException as exc:
        performance.finish(status="FAILED", error=exc)
        raise
    else:
        performance.finish(status="COMPLETE")


if __name__ == "__main__":
    main()
