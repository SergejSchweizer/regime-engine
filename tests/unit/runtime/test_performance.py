from __future__ import annotations

import json
from pathlib import Path

from market_regime_engine.runtime.performance import PerformanceRecorder


def test_performance_recorder_writes_stage_report(tmp_path: Path) -> None:
    path = tmp_path / "performance.json"
    recorder = PerformanceRecorder(path)
    with recorder.stage("statistical_evaluation", task_count=4, worker_count=2):
        pass
    recorder.finish(status="COMPLETE", metadata={"source": "test"})

    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "COMPLETE"
    assert report["schema_version"] == 2
    assert report["physical_core_count"] >= 1
    assert report["affinity_logical_cpus"] >= report["available_logical_cpus"]
    assert isinstance(report["numa_nodes"], list)
    assert report["metadata"] == {"source": "test"}
    assert report["stages"][0]["name"] == "statistical_evaluation"
    assert report["stages"][0]["task_count"] == 4
    assert report["stages"][0]["worker_count"] == 2
    assert report["stages"][0]["child_cpu_seconds"] >= 0.0
    assert report["stages"][0]["child_peak_rss_mib"] >= 0.0
    assert report["stages"][0]["available_logical_cpus"] >= 1
