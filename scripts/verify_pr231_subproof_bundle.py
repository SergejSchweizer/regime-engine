#!/usr/bin/env python3
"""Verify that all local PR-231 sub-proof sidecars form one proof bundle."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_PHASES = (
    "pipeline-math",
    "tracking-and-plots",
    "independent-process-and-labels",
    "future-mutation-isolation",
)
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON sidecar must contain an object: {path}")
    return value


def _is_digest(value: object, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _proof_hash(
    proof: dict[str, Any],
    field: str,
    phase: str,
    errors: list[str],
) -> str | None:
    value = proof.get(field)
    if not _is_digest(value, _SHA256_RE):
        errors.append(f"proof {phase} has missing or invalid {field}")
        return None
    return value


def _resolve_bundle_path(
    root: Path,
    value: str,
    *,
    label: str,
    errors: list[str],
    relative_to: Path | None = None,
) -> Path | None:
    """Resolve a sidecar path and require it to remain inside the bundle."""

    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (relative_to or root) / candidate
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        errors.append(f"{label} cannot be resolved: {value} ({exc})")
        return None
    try:
        resolved.relative_to(root)
    except ValueError:
        errors.append(f"{label} is outside the proof bundle: {value}")
        return None
    return resolved


def verify(output_dir: str | Path) -> dict[str, object]:
    """Return a machine-readable verification report without running computation."""

    root = Path(output_dir).resolve()
    errors: list[str] = []
    metadata: dict[str, dict[str, Any]] = {}
    proofs: dict[str, dict[str, Any]] = {}
    for phase in _PHASES:
        metadata_name = f"pr231-{phase}.json"
        metadata_path = _resolve_bundle_path(
            root,
            metadata_name,
            label=f"metadata for {phase}",
            errors=errors,
        )
        if metadata_path is None or not metadata_path.is_file():
            errors.append(f"missing metadata sidecar: {metadata_name}")
            continue
        try:
            item = _read_mapping(metadata_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid metadata sidecar {metadata_path.name}: {exc}")
            continue
        metadata[phase] = item
        if item.get("workflow") != "local":
            errors.append(f"metadata for {phase} is not a local proof")
        proof_value = item.get("computation_proof_path")
        if not isinstance(proof_value, str):
            errors.append(f"missing computation proof path for {phase}")
            continue
        proof_path = _resolve_bundle_path(
            root,
            proof_value,
            label=f"computation proof for {phase}",
            errors=errors,
        )
        if proof_path is None or not proof_path.is_file():
            errors.append(f"missing computation proof sidecar for {phase}: {proof_path}")
            continue
        try:
            proof = _read_mapping(proof_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid computation proof for {phase}: {exc}")
            continue
        proofs[phase] = proof

    repository_shas = {
        value for item in metadata.values() if isinstance(value := item.get("repository_sha"), str)
    }
    golden_hashes = {
        value
        for item in metadata.values()
        if isinstance(value := item.get("golden_snapshot_hash"), str)
    }
    if len(repository_shas) != 1:
        errors.append("sub-proof metadata does not share one repository SHA")
    if len(golden_hashes) != 1:
        errors.append("sub-proof metadata does not share one golden snapshot hash")
    for phase, item in metadata.items():
        if item.get("phase") != phase:
            errors.append(f"metadata phase mismatch for {phase}")
        if not _is_digest(item.get("repository_sha"), _GIT_SHA_RE):
            errors.append(f"metadata for {phase} has missing or invalid repository_sha")
        if not _is_digest(item.get("golden_snapshot_hash"), _SHA256_RE):
            errors.append(f"metadata for {phase} has missing or invalid golden_snapshot_hash")
        if type(item.get("exit_code")) is not int or item.get("exit_code") != 0:
            errors.append(f"sub-proof did not pass: {phase}")
    for phase, proof in proofs.items():
        if proof.get("phase") != phase:
            errors.append(f"proof phase mismatch for {phase}")
    pipeline = proofs.get("pipeline-math", {})
    if pipeline.get("golden_snapshot_hash") != next(iter(golden_hashes), None):
        errors.append("pipeline-math proof does not match the metadata golden hash")
    pipeline_result_hash = _proof_hash(pipeline, "result_hash", "pipeline-math", errors)
    pipeline_evidence_hash = _proof_hash(pipeline, "evidence_hash", "pipeline-math", errors)
    likelihood_count = pipeline.get("selected_likelihood_records_recomputed")
    if type(likelihood_count) is not int or likelihood_count < 2:
        errors.append(
            "pipeline-math has fewer than two independently recomputed likelihood records"
        )
    tracking = proofs.get("tracking-and-plots", {})
    tracking_result_hash = _proof_hash(tracking, "result_hash", "tracking-and-plots", errors)
    tracking_evidence_hash = _proof_hash(tracking, "evidence_hash", "tracking-and-plots", errors)
    plot_manifest = tracking.get("plot_manifest_path")
    plot_manifest_path = (
        _resolve_bundle_path(
            root,
            plot_manifest,
            label="plot manifest",
            errors=errors,
        )
        if isinstance(plot_manifest, str)
        else None
    )
    if plot_manifest_path is None or not plot_manifest_path.is_file():
        errors.append("tracking-and-plots proof has no plot manifest")
    else:
        try:
            manifest = _read_mapping(plot_manifest_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid tracking-and-plots manifest: {exc}")
        else:
            entries = manifest.get("entries")
            if not isinstance(entries, list) or not entries:
                errors.append("tracking-and-plots manifest has no entries")
            else:
                for index, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        errors.append(f"tracking-and-plots manifest entry {index} is not an object")
                        continue
                    png_value = entry.get("png_path")
                    png_path = (
                        _resolve_bundle_path(
                            root,
                            png_value,
                            label=f"plot artifact {index}",
                            errors=errors,
                            relative_to=plot_manifest_path.parent,
                        )
                        if isinstance(png_value, str)
                        else None
                    )
                    if png_path is None or not png_path.is_file():
                        errors.append(f"missing plot artifact {index}: {png_value}")
    independent = proofs.get("independent-process-and-labels", {})
    _proof_hash(independent, "snapshot_sha256", "independent-process-and-labels", errors)
    independent_result_hash = _proof_hash(
        independent,
        "result_sha256",
        "independent-process-and-labels",
        errors,
    )
    independent_evidence_hash = _proof_hash(
        independent,
        "evidence_sha256",
        "independent-process-and-labels",
        errors,
    )
    for field in ("canonical_result_bytes_equal", "canonical_evidence_bytes_equal"):
        if independent.get(field) is not True:
            errors.append(f"independent-process-and-labels failed: {field}")
    mutation = proofs.get("future-mutation-isolation", {})
    mutation_result_hash = _proof_hash(
        mutation,
        "baseline_result_hash",
        "future-mutation-isolation",
        errors,
    )
    mutation_evidence_hash = _proof_hash(
        mutation,
        "baseline_evidence_hash",
        "future-mutation-isolation",
        errors,
    )
    mutated_result_hash = _proof_hash(
        mutation,
        "mutated_result_hash",
        "future-mutation-isolation",
        errors,
    )
    mutation_row_index = mutation.get("mutation_row_index")
    if type(mutation_row_index) is not int or mutation_row_index < 0:
        errors.append("future-mutation-isolation has an invalid mutation_row_index")
    mutation_feature = mutation.get("mutation_feature")
    if not isinstance(mutation_feature, str) or not mutation_feature.strip():
        errors.append("future-mutation-isolation has no mutation_feature")
    for field in (
        "final_fold_changed",
        "earlier_fold_result_bytes_equal",
        "earlier_selection_bytes_equal",
        "earlier_evidence_bytes_equal",
    ):
        if mutation.get(field) is not True:
            errors.append(f"future-mutation-isolation failed: {field}")

    if pipeline_result_hash is not None:
        for phase, value in (
            ("tracking-and-plots", tracking_result_hash),
            ("independent-process-and-labels", independent_result_hash),
            ("future-mutation-isolation", mutation_result_hash),
        ):
            if value is not None and value != pipeline_result_hash:
                errors.append(f"{phase} result hash does not match pipeline-math")
    if pipeline_evidence_hash is not None:
        for phase, value in (
            ("tracking-and-plots", tracking_evidence_hash),
            ("independent-process-and-labels", independent_evidence_hash),
            ("future-mutation-isolation", mutation_evidence_hash),
        ):
            if value is not None and value != pipeline_evidence_hash:
                errors.append(f"{phase} evidence hash does not match pipeline-math")
    if (
        mutation_result_hash is not None
        and mutated_result_hash is not None
        and mutated_result_hash == mutation_result_hash
    ):
        errors.append("future-mutation-isolation mutated result hash did not change")

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
