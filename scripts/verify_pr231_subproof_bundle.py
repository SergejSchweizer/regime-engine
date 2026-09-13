#!/usr/bin/env python3
"""Verify that all local PR-231 sub-proof sidecars form one proof bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_PHASES = (
    "pipeline-math",
    "tracking-and-plots",
    "independent-process-and-labels",
    "future-mutation-isolation",
)


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON sidecar must contain an object: {path}")
    return value


def verify(output_dir: str | Path) -> dict[str, object]:
    """Return a machine-readable verification report without running computation."""

    root = Path(output_dir).resolve()
    errors: list[str] = []
    metadata: dict[str, dict[str, Any]] = {}
    proofs: dict[str, dict[str, Any]] = {}
    for phase in _PHASES:
        metadata_path = root / f"pr231-{phase}.json"
        if not metadata_path.is_file():
            errors.append(f"missing metadata sidecar: {metadata_path.name}")
            continue
        try:
            item = _read_mapping(metadata_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid metadata sidecar {metadata_path.name}: {exc}")
            continue
        metadata[phase] = item
        proof_value = item.get("computation_proof_path")
        if not isinstance(proof_value, str):
            errors.append(f"missing computation proof path for {phase}")
            continue
        proof_path = Path(proof_value)
        if not proof_path.is_file():
            errors.append(f"missing computation proof sidecar for {phase}: {proof_path}")
            continue
        try:
            proof = _read_mapping(proof_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid computation proof for {phase}: {exc}")
            continue
        proofs[phase] = proof

    repository_shas = {item.get("repository_sha") for item in metadata.values()}
    golden_hashes = {item.get("golden_snapshot_hash") for item in metadata.values()}
    if len(repository_shas) != 1:
        errors.append("sub-proof metadata does not share one repository SHA")
    if len(golden_hashes) != 1:
        errors.append("sub-proof metadata does not share one golden snapshot hash")
    for phase, item in metadata.items():
        if item.get("phase") != phase:
            errors.append(f"metadata phase mismatch for {phase}")
        if item.get("exit_code") != 0:
            errors.append(f"sub-proof did not pass: {phase}")
    for phase, proof in proofs.items():
        if proof.get("phase") != phase:
            errors.append(f"proof phase mismatch for {phase}")
    pipeline = proofs.get("pipeline-math", {})
    if pipeline.get("golden_snapshot_hash") != next(iter(golden_hashes), None):
        errors.append("pipeline-math proof does not match the metadata golden hash")
    tracking = proofs.get("tracking-and-plots", {})
    plot_manifest = tracking.get("plot_manifest_path")
    if not isinstance(plot_manifest, str) or not Path(plot_manifest).is_file():
        errors.append("tracking-and-plots proof has no plot manifest")
    independent = proofs.get("independent-process-and-labels", {})
    for field in ("canonical_result_bytes_equal", "canonical_evidence_bytes_equal"):
        if independent.get(field) is not True:
            errors.append(f"independent-process-and-labels failed: {field}")
    mutation = proofs.get("future-mutation-isolation", {})
    for field in (
        "final_fold_changed",
        "earlier_fold_result_bytes_equal",
        "earlier_selection_bytes_equal",
        "earlier_evidence_bytes_equal",
    ):
        if mutation.get(field) is not True:
            errors.append(f"future-mutation-isolation failed: {field}")

    return {
        "workflow": "local",
        "status": "verified" if not errors else "failed",
        "output_dir": str(root),
        "repository_sha": next(iter(repository_shas), None),
        "golden_snapshot_hash": next(iter(golden_hashes), None),
        "phases": list(_PHASES),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = verify(args.output_dir)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if report["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
