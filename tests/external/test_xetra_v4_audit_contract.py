"""Opt-in contract checks for the external current-Xetra v4 audit."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.external


def test_current_xetra_full_audit_is_opt_in_and_records_complete_evidence() -> None:
    if os.environ.get("REGIME_RUN_XETRA_V4_AUDIT") != "1":
        pytest.skip("set REGIME_RUN_XETRA_V4_AUDIT=1 to run the current-Xetra audit")
    root = Path(__file__).parents[2]
    summary_path_text = os.environ.get("REGIME_EVALUATION_SUMMARY_PATH")
    if not summary_path_text:
        pytest.fail("REGIME_EVALUATION_SUMMARY_PATH must point to durable audit evidence")
    summary_path = Path(summary_path_text)
    completed = subprocess.run(
        [str(root / "scripts/run_xetra_v4_cron.sh")],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["production_eligible"] is True
    assert summary["outer_fold_count"] >= 1
    assert summary["valid_fold_rate"] == 1.0
    assert summary["evidence_hash"]
    assert Path(summary["math_audit_expectations"]).is_file()
    assert Path(summary["math_audit_report"]).is_file()
