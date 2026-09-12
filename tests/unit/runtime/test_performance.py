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
    assert report["metadata"] == {"source": "test"}
    assert report["stages"][0]["name"] == "statistical_evaluation"
    assert report["stages"][0]["task_count"] == 4
    assert report["stages"][0]["worker_count"] == 2
