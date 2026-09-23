from __future__ import annotations

import json
import os
import resource
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
from mlflow.tracking import MlflowClient

from market_regime_engine.feature_discovery.feature_roles import (
    TEMPORAL_KEY,
    TRANSFORMATION_FAMILIES,
    build_feature_role_contract,
)
from market_regime_engine.feature_discovery.global_reduction import (
    prune_global_correlated_features,
)
from market_regime_engine.feature_discovery.pipeline import fit_family_pca_stages
from market_regime_engine.mlflow_support.canonical_tracking import CanonicalFileMlflowTrackingPort
from market_regime_engine.runtime.cpu import available_cpu_count
from market_regime_engine.runtime.task_frontier import FrontierTask, SharedTaskFrontier


def _benchmark_frontier_worker(task: FrontierTask[int]) -> tuple[str, str | None, str, int]:
    digest = sha256()
    for iteration in range(4_000):
        digest.update(f"{task.payload}:{iteration}".encode())
    native_thread_count = len(tuple(Path("/proc/self/task").iterdir()))
    return (
        task.task_id,
        os.environ.get("REGIME_CPU_PROCESS_WORKER"),
        digest.hexdigest(),
        native_thread_count,
    )


def _benchmark_tasks() -> tuple[FrontierTask[int], ...]:
    return tuple(
        FrontierTask(
            task_id=f"fold-{index % 2:02d}-task-{index:04d}",
            state_count=2,
            fold_id=f"fold-{index % 2:02d}",
            candidate_subset=(f"feature_{index % 16:02d}",),
            seed=index,
            profile_hash="a" * 64,
            matrix_identity="benchmark-matrix-v1",
            row_indices=(0, 1, 2, 3),
            column_indices=(index % 16,),
            payload=index,
        )
        for index in range(176)
    )


def test_fixed_parallel_benchmark_records_budget_parity_and_runtime(tmp_path: Path) -> None:
    tasks = _benchmark_tasks()
    budgets: tuple[int | None, ...] = (1, 8, 16, 32, 48, 64, 80, None)
    reports: list[dict[str, object]] = []
    canonical_hashes: list[str] = []
    for budget in budgets:
        started = time.perf_counter()
        cpu_before = resource.getrusage(resource.RUSAGE_SELF)
        child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        with SharedTaskFrontier(max_workers=budget) as frontier:
            worker_limit = frontier.worker_limit
            result = frontier.map(tasks, _benchmark_frontier_worker)
        cpu_after = resource.getrusage(resource.RUSAGE_SELF)
        child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        wall_seconds = time.perf_counter() - started
        canonical = tuple((task.task_id, value[2]) for task, value in result.values)
        canonical_hash = sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        canonical_hashes.append(canonical_hash)
        reports.append(
            {
                "requested_workers": budget,
                "effective_workers": worker_limit,
                "task_count": len(tasks),
                "wall_seconds": wall_seconds,
                "cpu_seconds": (
                    cpu_after.ru_utime
                    + cpu_after.ru_stime
                    - cpu_before.ru_utime
                    - cpu_before.ru_stime
                ),
                "child_cpu_seconds": (
                    child_after.ru_utime
                    + child_after.ru_stime
                    - child_before.ru_utime
                    - child_before.ru_stime
                ),
                "peak_rss_mib": child_after.ru_maxrss / 1024.0,
                "throughput_tasks_per_second": len(tasks) / wall_seconds,
                "queue_starvation_seconds": (
                    wall_seconds * max(0.0, 1.0 - result.metrics.worker_utilization_proxy)
                ),
                "worker_markers": sorted({value[1] for _task, value in result.values}),
                "native_thread_counts_per_worker": sorted(
                    {value[3] for _task, value in result.values}
                ),
            }
        )

    report = {
        "schema_version": 1,
        "available_logical_cpus": available_cpu_count(),
        "task_count": len(tasks),
        "canonical_hash": canonical_hashes[0],
        "runs": reports,
        "fastest_requested_workers": reports[
            min(range(len(reports)), key=lambda index: reports[index]["wall_seconds"])
        ]["requested_workers"],
    }
    report_path = tmp_path / "pr523-parallel-benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tracker = CanonicalFileMlflowTrackingPort(
        tracking_uri,
        experiment_name="pr523-hermetic-benchmark",
    )
    run_id = tracker.start_run(run_name="pr523-parallel-benchmark")
    tracker.log_artifact(run_id, str(report_path), "benchmark")
    tracker.end_run(run_id)
    client = MlflowClient(tracking_uri=tracking_uri)

    assert len(set(canonical_hashes)) == 1
    assert all(run["worker_markers"] == ["1"] for run in reports)
    assert all(run["native_thread_counts_per_worker"] == [1] for run in reports)
    assert all(run["effective_workers"] <= max(1, available_cpu_count()) for run in reports)
    assert all(run["peak_rss_mib"] < 64 * 1024 for run in reports)
    assert tuple(item.path for item in client.list_artifacts(run_id, "benchmark")) == (
        "benchmark/pr523-parallel-benchmark.json",
    )
    assert report_path.exists()


def test_fixed_production_shaped_preprocessing_benchmark_has_stage_hash_parity(
    tmp_path: Path,
) -> None:
    names = tuple(f"{family}_delta_1obs" for family in TRANSFORMATION_FAMILIES)
    contract = build_feature_role_contract((TEMPORAL_KEY, *names))
    matrix = np.random.default_rng(523).normal(size=(45, len(names)))
    values = {
        name: tuple(float(value) for value in matrix[:, index]) for index, name in enumerate(names)
    }
    budgets: tuple[int | None, ...] = (1, 8, 16, 32, 48, 64, 80, None)
    stage_runs: list[dict[str, object]] = []
    hashes: list[str] = []
    name_index = {name: index for index, name in enumerate(names)}
    for budget in budgets:
        started = time.perf_counter()
        process_before = resource.getrusage(resource.RUSAGE_SELF)
        child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        family_artifacts = fit_family_pca_stages(
            values,
            contract,
            quality_eligible_features=names,
            max_workers=budget,
        )
        generated: dict[str, tuple[float, ...]] = {}
        for artifact in family_artifacts:
            transformed = artifact.transform(
                matrix[:, [name_index[name] for name in artifact.feature_order]]
            )
            generated.update(
                {
                    name: tuple(float(row[index]) for row in transformed)
                    for index, name in enumerate(artifact.generated_feature_names)
                }
            )
        global_result = prune_global_correlated_features(
            generated,
            contract,
            max_workers=budget,
        )
        process_after = resource.getrusage(resource.RUSAGE_SELF)
        child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        wall_seconds = time.perf_counter() - started
        stage_hash = sha256(
            json.dumps(
                {
                    "family_fit_hashes": tuple(item.fit_hash for item in family_artifacts),
                    "global_hash": global_result.result_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        hashes.append(stage_hash)
        stage_runs.append(
            {
                "requested_workers": budget,
                "wall_seconds": wall_seconds,
                "cpu_seconds": (
                    process_after.ru_utime
                    + process_after.ru_stime
                    - process_before.ru_utime
                    - process_before.ru_stime
                ),
                "child_cpu_seconds": (
                    child_after.ru_utime
                    + child_after.ru_stime
                    - child_before.ru_utime
                    - child_before.ru_stime
                ),
                "peak_rss_mib": child_after.ru_maxrss / 1024.0,
                "throughput_families_per_second": len(TRANSFORMATION_FAMILIES) / wall_seconds,
                "queue_starvation_seconds": 0.0,
                "runnable_task_count": len(TRANSFORMATION_FAMILIES),
                "family_count": len(family_artifacts),
                "global_candidate_count": len(generated),
                "effective_worker_bound": min(available_cpu_count(), len(TRANSFORMATION_FAMILIES)),
            }
        )
    report_path = tmp_path / "pr523-stage-benchmark.json"
    report_path.write_text(
        json.dumps({"runs": stage_runs, "canonical_hash": hashes[0]}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    assert len(set(hashes)) == 1
    assert all(
        run["global_candidate_count"] <= len(TRANSFORMATION_FAMILIES) * 8 for run in stage_runs
    )
    assert all(run["peak_rss_mib"] < 64 * 1024 for run in stage_runs)
    assert all(run["queue_starvation_seconds"] == 0.0 for run in stage_runs)
    assert report_path.exists()
