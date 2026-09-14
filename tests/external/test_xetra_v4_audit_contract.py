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
    valid_outer_fold_indices = summary["valid_outer_fold_indices"]
    assert summary["outer_fold_count"] == len(valid_outer_fold_indices)
    assert len(valid_outer_fold_indices) >= 3
    assert summary["valid_fold_rate"] == 1.0
    assert summary["evidence_hash"]
    assert Path(summary["math_audit_expectations"]).is_file()
    assert Path(summary["math_audit_report"]).is_file()

    contract = summary["audit_contract"]
    assert contract["schema_version"] == 2
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
    assert contract["pca"]["mandatory"] is True
    assert contract["pca"]["component_count"] == 8
    assert contract["pca"]["generated_feature_names"] == [
        f"pca_pc_{index:03d}" for index in range(1, 9)
    ]
    search_bounds = contract["search_bounds"]
    per_fold = search_bounds["per_fold"]
    assert [item["outer_fold_index"] for item in per_fold] == valid_outer_fold_indices
    declared_final_ids = search_bounds["declared_bounds"]["candidate_families"][
        "final_candidate_ids"
    ]
    assert len(declared_final_ids) == 12
    for item in per_fold:
        assert item["final_candidate_ids"] == declared_final_ids
        assert item["cluster_count_candidates"] == list(
            range(2, min(12, item["eligible_feature_count"] - 1) + 1)
        )
        assert item["prefix_length_candidates"] == list(
            range(2, min(8, item["ranked_feature_count"]) + 1)
        )
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
    declared_by_fold = {item["outer_fold_index"]: item for item in per_fold}
    for dossier in expectations["fold_audits"]:
        declared = declared_by_fold[dossier["outer_fold_index"]]
        assert dossier["feature_order"] == declared["eligible_feature_names"]
        assert [item["feature"] for item in dossier["feature_scores"]] == dossier["feature_order"]
        assert [item["cluster_count"] for item in dossier["silhouette_clusters"]] == declared[
            "cluster_count_candidates"
        ]
        assert [item["prefix_length"] for item in dossier["prefix_nmi"]] == declared[
            "prefix_length_candidates"
        ]
        assert {item["scope"] for item in dossier["likelihoods"]} == {"TRAIN", "OOS"}

    report = json.loads(Path(summary["math_audit_report"]).read_text(encoding="utf-8"))
    assert report["status"] == "verified"
    assert report["audit_contract_verified"] is True
    assert report["audited_outer_fold_indices"] == summary["audit_outer_fold_indices"]
    assert report["audited_outer_fold_count"] == len(summary["audit_outer_fold_indices"])
    assert report["outer_agreement_count"] == len(valid_outer_fold_indices)
    for key in (
        "distance_max_abs_error",
        "silhouette_max_abs_error",
        "feature_score_max_abs_error",
        "soft_nmi_max_abs_error",
        "outer_soft_nmi_max_abs_error",
        "gaussian_likelihood_max_abs_error",
    ):
        assert report[key] <= 1.0e-10

    resource_path = Path(contract["resource_evidence"]["performance_report_path"])
    assert resource_path.is_file()
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    assert resource["schema_version"] == 2
    assert resource["status"] == "COMPLETE"
    assert resource["available_logical_cpus"] >= 1
    assert resource["stages"]
