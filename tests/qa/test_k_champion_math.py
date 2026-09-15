from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def _oracle() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "verify_k_champion_math.py"
    spec = importlib.util.spec_from_file_location("verify_k_champion_math", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load independent K champion math oracle")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _hash(character: str) -> str:
    return character * 64


def _provenance_slots() -> list[dict[str, object]]:
    return [
        {
            "slot_id": f"champion-k{k}",
            "state_count": k,
            "feature_order_hash": _hash(chr(ord("a") + k)),
            "source_build_id": "build-2026-09-15",
            "source_data_sha256": _hash("f"),
            "policy_version": "k_champion_policy.v1",
            "folds": [
                {
                    "slot_id": f"champion-k{k}",
                    "state_count": k,
                    "fold_id": "outer-001",
                    "feature_order_hash": _hash(chr(ord("a") + k)),
                    "source_build_id": "build-2026-09-15",
                    "source_data_sha256": _hash("f"),
                }
            ],
        }
        for k in (2, 3, 4, 5)
    ]


def test_independent_soft_nmi_is_exact_and_state_label_invariant() -> None:
    oracle = _oracle()
    timestamps = ("t1", "t2", "t3", "t4")
    left = ((1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0))
    right = ((1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0))
    assert oracle.independent_soft_regime_nmi(timestamps, left, timestamps, right) == pytest.approx(
        1.0
    )
    relabeled = tuple((row[1], row[0]) for row in right)
    assert oracle.independent_soft_regime_nmi(
        timestamps, left, timestamps, relabeled
    ) == pytest.approx(1.0)
    assert oracle.independent_support(("t1", "t2"), timestamps) == (2, 0.5)


@pytest.mark.parametrize("state_count", (2, 3, 4, 5))
def test_every_k_prefix_tie_rule_is_nmi_support_then_smallest_prefix(state_count: int) -> None:
    oracle = _oracle()
    selected = oracle.independent_prefix_selection(
        (
            {
                "state_count": state_count,
                "prefix_length": 2,
                "valid": True,
                "soft_regime_nmi": 0.8,
                "shared_timestamp_count": 90,
            },
            {
                "state_count": state_count,
                "prefix_length": 3,
                "valid": True,
                "soft_regime_nmi": 0.8,
                "shared_timestamp_count": 95,
            },
            {
                "state_count": state_count,
                "prefix_length": 4,
                "valid": True,
                "soft_regime_nmi": 0.7,
                "shared_timestamp_count": 100,
            },
        )
    )
    assert selected["prefix_length"] == 3


def test_valid_fold_aggregation_keeps_invalid_folds_in_denominator() -> None:
    oracle = _oracle()
    result = oracle.independent_valid_fold_aggregation(
        (
            {"valid": True, "soft_regime_nmi": 0.5, "support": 0.9},
            {"valid": True, "soft_regime_nmi": 0.7, "support": 1.0},
            {"valid": False},
        )
    )
    assert result["valid_fold_count"] == 2
    assert result["valid_fold_rate"] == pytest.approx(2 / 3)
    assert result["soft_regime_nmi_mean"] == pytest.approx(0.6)
    assert result["eligible"] is False
    with pytest.raises(ValueError, match="partial"):
        oracle.independent_valid_fold_aggregation(({"valid": False, "support": 0.5},))


def _family(candidate_id: str, family: str, **values: object) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "model_family": family,
        "state_count": 4,
        "feature_order_hash": _hash("a"),
        "source_build_id": "build-2026-09-15",
        "evaluation_plan_hash": _hash("b"),
        "fold_ids": ("outer-001", "outer-002", "outer-003"),
        "valid_fold_rate": 1.0,
        "valid_fold_count": 3,
        "oos_mean": -1.0,
        "oos_std": 0.2,
        "oos_worst": -1.2,
        "bic_mean": 10.0,
        "aic_mean": 9.0,
        **values,
    }


def test_same_vector_ranking_covers_three_families_and_deterministic_ties() -> None:
    oracle = _oracle()
    result = oracle.independent_same_vector_family_ranking(
        (
            _family("student", "student_t_hmm", oos_mean=-1.0),
            _family("gmm", "gmm_hmm", oos_mean=-1.0),
            _family("gaussian", "gaussian_hmm", oos_mean=-1.0),
        )
    )
    assert result["ranked_candidate_ids"] == ("gaussian", "gmm", "student")
    assert result["winner_candidate_id"] == "gaussian"


def test_oracle_rejects_cross_dimension_raw_metrics_and_bad_provenance() -> None:
    oracle = _oracle()
    with pytest.raises(ValueError, match="cannot be compared"):
        oracle.reject_cross_dimension_metric_comparison("aic", (_hash("a"), _hash("b")))
    with pytest.raises(ValueError, match="exactly K"):
        oracle.verify_k_slot_provenance(_provenance_slots()[:-1])
    mutated = _provenance_slots()
    mutated[2]["folds"][0]["source_data_sha256"] = _hash("e")  # type: ignore[index]
    with pytest.raises(ValueError, match="source hash"):
        oracle.verify_k_slot_provenance(mutated)


def test_dossier_report_hash_changes_when_one_primitive_changes(tmp_path: Path) -> None:
    oracle = _oracle()
    dossier = {"seed": 42, "slots": _provenance_slots()}
    report = oracle.audit_dossier(dossier)
    assert report["status"] == "verified"
    mutated = json.loads(json.dumps(dossier))
    mutated["slots"][0]["source_build_id"] = "build-mutated"
    with pytest.raises(ValueError, match="source build"):
        oracle.audit_dossier(mutated)
    path = tmp_path / "dossier.json"
    path.write_text(json.dumps(dossier), encoding="utf-8")
    assert oracle.canonical_hash(dossier) == report["input_canonical_sha256"]
