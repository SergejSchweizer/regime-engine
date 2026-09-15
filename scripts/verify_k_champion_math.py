#!/usr/bin/env python3
"""Independent mathematical oracle for K-specific champion selection.

The verifier intentionally uses only the Python standard library.  It does
not import production selection, agreement, scoring, or likelihood helpers.
It consumes a JSON dossier produced by a K-specific evaluation and emits a
small, hash-bound QA report.  The functions are also useful for hand-built
fixtures in ``tests/qa/test_k_champion_math.py``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Hashable, Mapping, Sequence
from hashlib import sha256
from math import fsum, isfinite, log
from pathlib import Path

LEGAL_K = (2, 3, 4, 5)
MODEL_FAMILIES = ("gaussian_hmm", "gmm_hmm", "student_t_hmm")
MIN_PREFIX_LENGTH = 2
MAX_PREFIX_LENGTH = 8
VALID_FOLD_RATE_GATE = 0.80
MIN_VALID_FOLD_COUNT = 3
PREFIX_NMI_TIE_TOLERANCE = 1.0e-12
RANKING_ABS_TOLERANCE = 1.0e-12


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty trimmed string")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{field} must be finite")
    return float(value)


def _unit(value: object, field: str) -> float:
    number = _finite(value, field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be in [0,1]")
    return number


def _sha256(value: object, field: str) -> str:
    text = _text(value, field)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return text


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _probability_row(row: Sequence[object], field: str) -> tuple[float, ...]:
    if not row:
        raise ValueError(f"{field} must not be empty")
    values = tuple(_finite(value, field) for value in row)
    if any(value < 0.0 for value in values):
        raise ValueError(f"{field} must be non-negative")
    total = fsum(values)
    if abs(total - 1.0) > 1.0e-10:
        raise ValueError(f"{field} must be normalized")
    return values


def independent_soft_regime_nmi(
    timestamps: Sequence[Hashable],
    probabilities: Sequence[Sequence[object]],
    teacher_timestamps: Sequence[Hashable],
    teacher_probabilities: Sequence[Sequence[object]],
) -> float:
    """Recompute soft NMI from the exact common timestamp set.

    The joint distribution is the mean outer product of the two posterior
    probability vectors.  Row/column state relabeling only permutes this
    distribution and therefore leaves the result unchanged.
    """

    if len(timestamps) != len(probabilities) or len(teacher_timestamps) != len(
        teacher_probabilities
    ):
        raise ValueError("timestamps and probability rows must align")
    left = dict(zip(timestamps, probabilities, strict=True))
    right = dict(zip(teacher_timestamps, teacher_probabilities, strict=True))
    if len(left) != len(timestamps) or len(right) != len(teacher_timestamps):
        raise ValueError("timestamps must be unique")
    shared = sorted(set(left).intersection(right), key=repr)
    if not shared:
        raise ValueError("soft-NMI requires non-empty shared timestamps")
    left_rows = tuple(_probability_row(left[item], "left probabilities") for item in shared)
    right_rows = tuple(_probability_row(right[item], "right probabilities") for item in shared)
    left_count = len(left_rows[0])
    right_count = len(right_rows[0])
    if any(len(row) != left_count for row in left_rows) or any(
        len(row) != right_count for row in right_rows
    ):
        raise ValueError("probability state dimensions must be constant")
    joint = [
        [
            fsum(
                left_row[i] * right_row[j]
                for left_row, right_row in zip(left_rows, right_rows, strict=True)
            )
            / len(shared)
            for j in range(right_count)
        ]
        for i in range(left_count)
    ]
    left_marginal = tuple(fsum(row) for row in joint)
    right_marginal = tuple(fsum(joint[i][j] for i in range(left_count)) for j in range(right_count))
    mutual_information = 0.0
    for i, row in enumerate(joint):
        for j, value in enumerate(row):
            if value > 0.0:
                denominator = left_marginal[i] * right_marginal[j]
                if denominator <= 0.0:
                    raise ValueError("soft-NMI joint distribution has invalid marginals")
                mutual_information += value * log(value / denominator)
    left_entropy = -fsum(value * log(value) for value in left_marginal if value > 0.0)
    right_entropy = -fsum(value * log(value) for value in right_marginal if value > 0.0)
    if left_entropy <= 0.0 or right_entropy <= 0.0:
        raise ValueError("soft-NMI requires non-degenerate marginal entropy")
    result = 2.0 * mutual_information / (left_entropy + right_entropy)
    if not isfinite(result) or not 0.0 <= result <= 1.0 + 1.0e-10:
        raise ValueError("soft-NMI result is outside [0,1]")
    return min(1.0, max(0.0, result))


def independent_support(
    timestamps: Sequence[Hashable],
    teacher_timestamps: Sequence[Hashable],
) -> tuple[int, float]:
    """Return common support count and coverage of the teacher timeline."""

    left = tuple(timestamps)
    right = tuple(teacher_timestamps)
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise ValueError("support timestamps must be unique")
    if not right:
        raise ValueError("teacher support cannot be empty")
    count = len(set(left).intersection(right))
    return count, count / len(right)


def permitted_prefix_lengths(ranked_feature_count: int) -> tuple[int, ...]:
    """Return every nested prefix length permitted by the v4 contract.

    This is intentionally duplicated as a small mathematical contract rather
    than importing the production prefix-search module.  The upper bound is
    the smaller of the ranked feature count and the fixed v4 cap.
    """

    if (
        isinstance(ranked_feature_count, bool)
        or not isinstance(ranked_feature_count, int)
        or ranked_feature_count < MIN_PREFIX_LENGTH
    ):
        raise ValueError("ranked_feature_count must be at least two")
    upper_bound = min(ranked_feature_count, MAX_PREFIX_LENGTH)
    return tuple(range(MIN_PREFIX_LENGTH, upper_bound + 1))


def independent_prefix_selection(
    prefixes: Sequence[Mapping[str, object]],
    *,
    ranked_feature_count: int | None = None,
) -> Mapping[str, object]:
    """Select a K-specific prefix using NMI, support, then smaller prefix."""

    if not prefixes:
        raise ValueError("prefix selection requires candidates")
    normalized: list[dict[str, object]] = []
    k_values: set[int] = set()
    lengths: set[int] = set()
    for raw in prefixes:
        if not isinstance(raw, Mapping):
            raise ValueError("prefix candidates must be mappings")
        k = raw.get("state_count")
        if isinstance(k, bool) or not isinstance(k, int) or k not in LEGAL_K:
            raise ValueError("prefix state_count must be K=2,3,4 or 5")
        length = raw.get("prefix_length")
        if (
            isinstance(length, bool)
            or not isinstance(length, int)
            or not MIN_PREFIX_LENGTH <= length <= MAX_PREFIX_LENGTH
        ):
            raise ValueError("prefix_length must be in the permitted range 2..8")
        k_values.add(k)
        lengths.add(length)
        valid = raw.get("valid")
        if not isinstance(valid, bool):
            raise ValueError("prefix valid must be boolean")
        item = dict(raw)
        item["state_count"] = k
        item["prefix_length"] = length
        if valid:
            item["soft_regime_nmi"] = _unit(raw.get("soft_regime_nmi"), "soft_regime_nmi")
            support = raw.get("shared_timestamp_count")
            if isinstance(support, bool) or not isinstance(support, int) or support < 0:
                raise ValueError("shared_timestamp_count must be a non-negative integer")
            item["shared_timestamp_count"] = support
        normalized.append(item)
    if len(k_values) != 1:
        raise ValueError("prefix selection must not compare different K values")
    if len(lengths) != len(normalized):
        raise ValueError("prefix lengths must be unique")
    if ranked_feature_count is None:
        expected_lengths = tuple(range(MIN_PREFIX_LENGTH, max(lengths) + 1))
    else:
        expected_lengths = permitted_prefix_lengths(ranked_feature_count)
    if tuple(sorted(lengths)) != expected_lengths:
        raise ValueError("prefix candidates must cover every permitted prefix length")
    eligible = tuple(item for item in normalized if item["valid"] is True)
    if not eligible:
        raise ValueError("no eligible K-specific prefix")
    maximum_nmi = max(float(item["soft_regime_nmi"]) for item in eligible)
    nmi_ties = tuple(
        item
        for item in eligible
        if float(item["soft_regime_nmi"]) >= maximum_nmi - PREFIX_NMI_TIE_TOLERANCE
    )
    maximum_support = max(int(item["shared_timestamp_count"]) for item in nmi_ties)
    support_ties = tuple(
        item for item in nmi_ties if int(item["shared_timestamp_count"]) == maximum_support
    )
    winner = min(support_ties, key=lambda item: int(item["prefix_length"]))
    return {
        "state_count": int(winner["state_count"]),
        "prefix_length": int(winner["prefix_length"]),
        "soft_regime_nmi": float(winner["soft_regime_nmi"]),
        "shared_timestamp_count": int(winner["shared_timestamp_count"]),
        "candidate_count": len(normalized),
    }


def independent_valid_fold_aggregation(
    folds: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Aggregate valid-fold NMI/support without dropping invalid folds."""

    if not folds:
        raise ValueError("fold aggregation requires planned folds")
    valid_values: list[float] = []
    support_values: list[float] = []
    for fold in folds:
        if not isinstance(fold, Mapping) or not isinstance(fold.get("valid"), bool):
            raise ValueError("fold validity must be boolean")
        if fold["valid"]:
            valid_values.append(_unit(fold.get("soft_regime_nmi"), "soft_regime_nmi"))
            support_values.append(_unit(fold.get("support"), "support"))
        elif any(key in fold for key in ("soft_regime_nmi", "support")):
            raise ValueError("invalid folds cannot carry partial aggregate values")
    valid_count = len(valid_values)
    rate = valid_count / len(folds)
    return {
        "planned_fold_count": len(folds),
        "valid_fold_count": valid_count,
        "valid_fold_rate": rate,
        "soft_regime_nmi_mean": fsum(valid_values) / valid_count if valid_values else None,
        "soft_regime_nmi_worst": min(valid_values) if valid_values else None,
        "support_mean": fsum(support_values) / len(support_values) if support_values else None,
        "eligible": rate >= VALID_FOLD_RATE_GATE and valid_count >= MIN_VALID_FOLD_COUNT,
    }


def _anchored_partition(
    candidates: Sequence[Mapping[str, object]], field: str, higher_is_better: bool
) -> tuple[tuple[Mapping[str, object], ...], ...]:
    remaining = list(candidates)
    partitions: list[tuple[Mapping[str, object], ...]] = []
    while remaining:
        values = [_finite(item.get(field), field) for item in remaining]
        anchor = max(values) if higher_is_better else min(values)
        tied = tuple(
            item
            for item in remaining
            if (
                _finite(item.get(field), field) >= anchor - RANKING_ABS_TOLERANCE
                if higher_is_better
                else _finite(item.get(field), field) <= anchor + RANKING_ABS_TOLERANCE
            )
        )
        partitions.append(tied)
        tied_ids = {item["candidate_id"] for item in tied}
        remaining = [item for item in remaining if item["candidate_id"] not in tied_ids]
    return tuple(partitions)


def independent_same_vector_family_ranking(
    candidates: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Independently rank exactly three families within one K/feature vector."""

    if len(candidates) != len(MODEL_FAMILIES):
        raise ValueError("same-vector family ranking requires exactly three candidates")
    if any(not isinstance(item, Mapping) for item in candidates):
        raise ValueError("same-vector family candidates must be mappings")
    normalized = tuple(dict(item) for item in candidates)
    families = tuple(_text(item.get("model_family"), "model_family") for item in normalized)
    if set(families) != set(MODEL_FAMILIES):
        raise ValueError("family set must be Gaussian, GMM and Student-t")
    ids = tuple(_text(item.get("candidate_id"), "candidate_id") for item in normalized)
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    k_values = {item.get("state_count") for item in normalized}
    if (
        len(k_values) != 1
        or isinstance(next(iter(k_values)), bool)
        or not isinstance(next(iter(k_values)), int)
        or next(iter(k_values)) not in LEGAL_K
    ):
        raise ValueError("same-vector family candidates must share legal K")
    lineage = tuple(
        (
            _sha256(item.get("feature_order_hash"), "feature_order_hash"),
            _text(item.get("source_build_id"), "source_build_id"),
            _sha256(item.get("evaluation_plan_hash"), "evaluation_plan_hash"),
            tuple(item.get("fold_ids", ()))
            if isinstance(item.get("fold_ids", ()), Sequence)
            and not isinstance(item.get("fold_ids", ()), (str, bytes))
            else (),
        )
        for item in normalized
    )
    if any(
        not fold_ids
        or len(set(fold_ids)) != len(fold_ids)
        or any(not isinstance(fold_id, str) or not fold_id for fold_id in fold_ids)
        for _, _, _, fold_ids in lineage
    ):
        raise ValueError("same-vector family candidates require unique fold IDs")
    if len(set(lineage)) != 1:
        raise ValueError("same-vector family candidates must share exact provenance")
    accepted: list[Mapping[str, object]] = []
    rejected: dict[str, str] = {}
    for item, candidate_id in zip(normalized, ids, strict=True):
        rate = _unit(item.get("valid_fold_rate"), "valid_fold_rate")
        count = item.get("valid_fold_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("valid_fold_count must be a non-negative integer")
        metrics = ("oos_mean", "oos_std", "oos_worst", "bic_mean", "aic_mean")
        if rate < VALID_FOLD_RATE_GATE:
            rejected[candidate_id] = "valid-fold rate below 0.80"
        elif count < MIN_VALID_FOLD_COUNT:
            rejected[candidate_id] = "fewer than three valid folds"
        else:
            try:
                metric_values = tuple(_finite(item.get(field), field) for field in metrics)
            except ValueError:
                rejected[candidate_id] = "missing or nonfinite ranking metric"
            else:
                if any(not isfinite(value) for value in metric_values):
                    rejected[candidate_id] = "missing or nonfinite ranking metric"
                else:
                    accepted.append(item)
    if not accepted:
        raise ValueError("no same-vector family candidate passes hard gates")
    groups: tuple[tuple[Mapping[str, object], ...], ...] = (tuple(accepted),)
    for field, higher in (
        ("oos_mean", True),
        ("oos_std", False),
        ("oos_worst", True),
        ("bic_mean", False),
        ("aic_mean", False),
    ):
        groups = tuple(
            subgroup for group in groups for subgroup in _anchored_partition(group, field, higher)
        )
    ranked = tuple(
        item
        for group in groups
        for item in sorted(
            group,
            key=lambda item: (int(item["state_count"]), str(item["candidate_id"])),
        )
    )
    return {
        "winner_candidate_id": str(ranked[0]["candidate_id"]),
        "ranked_candidate_ids": tuple(str(item["candidate_id"]) for item in ranked),
        "rejected": rejected,
    }


def reject_cross_dimension_metric_comparison(
    metric_key: str,
    feature_order_hashes: Sequence[str],
    *,
    state_counts: Sequence[int] | None = None,
) -> None:
    """Reject raw vector likelihood/criterion comparisons across K dimensions."""

    metric = _text(metric_key, "metric_key").lower()
    hashes = tuple(_sha256(value, "feature_order_hash") for value in feature_order_hashes)
    if not hashes:
        raise ValueError("feature_order_hashes must not be empty")
    if state_counts is not None:
        normalized_k = tuple(state_counts)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value not in LEGAL_K
            for value in normalized_k
        ):
            raise ValueError("state_counts must contain only K=2,3,4 or 5")
        if len(normalized_k) != len(hashes):
            raise ValueError("state_counts and feature_order_hashes must align")
    else:
        normalized_k = ()
    raw_metric = any(token in metric for token in ("loglik", "pll", "aic", "bic", "hqc"))
    different_dimensions = len(set(hashes)) > 1 or len(set(normalized_k)) > 1
    if raw_metric and different_dimensions:
        raise ValueError(f"{metric_key} cannot be compared across K-specific feature vectors")


def verify_k_slot_provenance(slots: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """Verify exact slot/fold lineage links and canonical slot identities."""

    if not slots or any(not isinstance(item, Mapping) for item in slots):
        raise ValueError("provenance dossier requires mapping slots")
    if (
        tuple(
            sorted(
                item.get("state_count")
                for item in slots
                if isinstance(item.get("state_count"), int)
                and not isinstance(item.get("state_count"), bool)
            )
        )
        != LEGAL_K
    ):
        raise ValueError("provenance dossier must contain exactly K=2,3,4,5")
    identities: list[dict[str, object]] = []
    seen_slots: set[str] = set()
    for slot in slots:
        k = slot.get("state_count")
        if isinstance(k, bool) or not isinstance(k, int) or k not in LEGAL_K:
            raise ValueError("slot state_count is invalid")
        slot_id = _text(slot.get("slot_id"), "slot_id")
        if slot_id != f"champion-k{k}" or slot_id in seen_slots:
            raise ValueError("slot identity is not canonical or is duplicated")
        seen_slots.add(slot_id)
        feature_hash = _sha256(slot.get("feature_order_hash"), "feature_order_hash")
        source_build = _text(slot.get("source_build_id"), "source_build_id")
        source_hash = _sha256(slot.get("source_data_sha256"), "source_data_sha256")
        policy = _text(slot.get("policy_version"), "policy_version")
        folds = slot.get("folds")
        if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)) or not folds:
            raise ValueError("every K slot must contain fold provenance")
        fold_ids: set[str] = set()
        for fold in folds:
            if not isinstance(fold, Mapping):
                raise ValueError("fold provenance must be a mapping")
            if fold.get("state_count") != k:
                raise ValueError("fold K does not match slot K")
            if fold.get("slot_id") != slot_id:
                raise ValueError("fold slot_id does not match slot")
            if _sha256(fold.get("feature_order_hash"), "fold feature_order_hash") != feature_hash:
                raise ValueError("fold feature hash does not match slot")
            if _text(fold.get("source_build_id"), "fold source_build_id") != source_build:
                raise ValueError("fold source build does not match slot")
            if _sha256(fold.get("source_data_sha256"), "fold source_data_sha256") != source_hash:
                raise ValueError("fold source hash does not match slot")
            fold_id = _text(fold.get("fold_id"), "fold_id")
            if fold_id in fold_ids:
                raise ValueError("fold IDs must be unique per K slot")
            fold_ids.add(fold_id)
        if not fold_ids:
            raise ValueError("every K slot must contain at least one fold ID")
        identities.append(
            {
                "slot_id": slot_id,
                "state_count": k,
                "feature_order_hash": feature_hash,
                "source_build_id": source_build,
                "source_data_sha256": source_hash,
                "policy_version": policy,
                "fold_count": len(folds),
            }
        )
    return {"slots": tuple(sorted(identities, key=lambda item: int(item["state_count"])))}


def audit_dossier(
    dossier: Mapping[str, object],
    *,
    command: Sequence[str] = (),
) -> dict[str, object]:
    """Run the independent checks represented by a JSON dossier."""

    if not isinstance(dossier, Mapping):
        raise ValueError("K math dossier must be a JSON object")
    slots = dossier.get("slots")
    if not isinstance(slots, Sequence) or isinstance(slots, (str, bytes)):
        raise ValueError("K math dossier requires slots")
    provenance = verify_k_slot_provenance(slots)
    return {
        "status": "verified",
        "oracle": "verify_k_champion_math.py",
        "oracle_version": "k_champion_math.v1",
        "python_version": platform.python_version(),
        "command": tuple(command),
        "exit_code": 0,
        "seed": dossier.get("seed"),
        "input_canonical_sha256": canonical_hash(dossier),
        "provenance": provenance,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dossier", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    dossier = json.loads(args.dossier.read_text(encoding="utf-8"))
    command = tuple(sys.argv) if argv is None else (Path(__file__).name, *tuple(map(str, argv)))
    report = audit_dossier(dossier, command=command)
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.report is None:
        sys.stdout.write(encoded)
    else:
        args.report.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
