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

    contract = summary["audit_contract"]
    assert contract["schema_version"] == 1
    assert contract["source_request"] == {
        "mode": "schema_discovery",
        "feature_names": [],
        "start": None,
        "end": None,
        "all_source_rows": True,
        "dynamic_catalog": True,
    }
    assert contract["audit_source_request"] == contract["source_request"]
    assert contract["valid_outer_fold_indices"] == summary["valid_outer_fold_indices"]
    assert contract["audit_outer_fold_indices"] == summary["audit_outer_fold_indices"]
    assert len(contract["audit_outer_fold_indices"]) == 3
    assert set(contract["identity_hashes"]) >= {
        "source_build_id",
        "source_data_sha256",
        "source_catalog_hash",
        "materialized_feature_data_sha256",
        "dataset_snapshot_key",
        "snapshot_sha256",
        "profile_hash",
        "outer_plan_hash",
        "repository_commit_sha",
        "uv_lock_sha256",
    }
    assert len(contract["identity_hashes"]["repository_commit_sha"]) == 40
    for key, value in contract["identity_hashes"].items():
        if key not in {"repository_commit_sha", "source_build_id"}:
            assert len(value) == 64

    expectations = json.loads(Path(summary["math_audit_expectations"]).read_text(encoding="utf-8"))
    assert expectations["schema_version"] == 3
    assert expectations["audit_contract"] == contract
    assert expectations["valid_outer_fold_indices"] == summary["valid_outer_fold_indices"]

    report = json.loads(Path(summary["math_audit_report"]).read_text(encoding="utf-8"))
    assert report["status"] == "verified"
    assert report["audit_contract_verified"] is True
    assert report["audited_outer_fold_indices"] == summary["audit_outer_fold_indices"]

    resource_path = Path(contract["resource_evidence"]["performance_report_path"])
    assert resource_path.is_file()
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    assert resource["schema_version"] == 2
    assert resource["status"] == "COMPLETE"
    assert resource["available_logical_cpus"] >= 1
    assert resource["stages"]
