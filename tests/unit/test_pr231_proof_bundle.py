from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def _module() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "verify_pr231_subproof_bundle.py"
    spec = importlib.util.spec_from_file_location("verify_pr231_proof_bundle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load proof-bundle verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_bundle(root: Path, *, repository_sha: str = "a" * 40) -> None:
    golden = "b" * 64
    proof_values: dict[str, dict[str, object]] = {
        "pipeline-math": {"golden_snapshot_hash": golden},
        "tracking-and-plots": {"plot_manifest_path": str(root / "plots.json")},
        "independent-process-and-labels": {
            "canonical_result_bytes_equal": True,
            "canonical_evidence_bytes_equal": True,
        },
        "future-mutation-isolation": {
            "final_fold_changed": True,
            "earlier_fold_result_bytes_equal": True,
            "earlier_selection_bytes_equal": True,
            "earlier_evidence_bytes_equal": True,
        },
    }
    (root / "plots.json").write_text("{}\n", encoding="utf-8")
    for phase, values in proof_values.items():
        proof_path = root / f"{phase}-proof.json"
        proof_path.write_text(json.dumps({"phase": phase, **values}) + "\n", encoding="utf-8")
        metadata = {
            "phase": phase,
            "repository_sha": repository_sha,
            "golden_snapshot_hash": golden,
            "exit_code": 0,
            "computation_proof_path": str(proof_path),
        }
        (root / f"pr231-{phase}.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")


def test_verify_accepts_one_consistent_bundle(tmp_path: Path) -> None:
    module = _module()
    _write_bundle(tmp_path)
    report = module.verify(tmp_path)
    assert report["status"] == "verified"
    assert report["errors"] == []


def test_verify_rejects_mixed_commits_and_failed_phase(tmp_path: Path) -> None:
    module = _module()
    _write_bundle(tmp_path)
    failed = json.loads((tmp_path / "pr231-pipeline-math.json").read_text(encoding="utf-8"))
    failed["exit_code"] = 1
    failed["repository_sha"] = "c" * 40
    (tmp_path / "pr231-pipeline-math.json").write_text(json.dumps(failed) + "\n", encoding="utf-8")
    report = module.verify(tmp_path)
    assert report["status"] == "failed"
    assert any("one repository SHA" in error for error in report["errors"])
    assert any("did not pass: pipeline-math" in error for error in report["errors"])
