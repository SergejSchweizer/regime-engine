from __future__ import annotations

import importlib.util
import itertools
import json
import math
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


LEGAL_K = (2, 3, 4, 5)
PERMITTED_PREFIX_LENGTHS = tuple(range(2, 9))
FAMILIES = ("gaussian_hmm", "gmm_hmm", "student_t_hmm")


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
        for k in LEGAL_K
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


def test_soft_nmi_matches_a_hand_calculated_binary_channel() -> None:
    oracle = _oracle()
    timestamps = ("t1", "t2", "t3", "t4")
    left = ((1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0))
    right = ((0.75, 0.25), (0.75, 0.25), (0.25, 0.75), (0.25, 0.75))
    binary_entropy = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    expected = 1.0 - binary_entropy / math.log(2.0)
    assert oracle.independent_soft_regime_nmi(timestamps, left, timestamps, right) == pytest.approx(
        expected
    )
    assert oracle.independent_soft_regime_nmi(
        timestamps, left, timestamps, tuple((row[1], row[0]) for row in right)
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("left", "right", "message"),
    (
        (((1.0, 0.0),), ((1.0, 0.0),), "non-degenerate"),
        (((1.0, 0.0),), ((0.5, 0.5),), "non-degenerate"),
        (((1.1, -0.1),), ((1.0, 0.0),), "non-negative"),
        (((0.5, 0.5),), ((0.8, 0.3),), "normalized"),
    ),
)
def test_soft_nmi_rejects_degenerate_or_invalid_probability_rows(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
    message: str,
) -> None:
    oracle = _oracle()
    with pytest.raises(ValueError, match=message):
        oracle.independent_soft_regime_nmi(("t1",), left, ("t1",), right)


def test_support_rejects_duplicate_and_empty_teacher_timelines() -> None:
    oracle = _oracle()
    with pytest.raises(ValueError, match="unique"):
        oracle.independent_support(("t1", "t1"), ("t1",))
    with pytest.raises(ValueError, match="empty"):
        oracle.independent_support(("t1",), ())


@pytest.mark.parametrize("state_count", LEGAL_K)
def test_every_k_and_every_permitted_prefix_is_covered(state_count: int) -> None:
    oracle = _oracle()
    assert oracle.permitted_prefix_lengths(2) == (2,)
    assert oracle.permitted_prefix_lengths(8) == PERMITTED_PREFIX_LENGTHS
    selected = oracle.independent_prefix_selection(
        tuple(
            {
                "state_count": state_count,
                "prefix_length": length,
                "valid": True,
                "soft_regime_nmi": 0.8 if length in (3, 5) else 0.7,
                "shared_timestamp_count": 100 if length == 5 else 95,
            }
            for length in reversed(PERMITTED_PREFIX_LENGTHS)
        ),
        ranked_feature_count=8,
    )
    assert selected["prefix_length"] == 5
    assert selected["candidate_count"] == len(PERMITTED_PREFIX_LENGTHS)


def test_prefix_ties_use_nmi_then_support_then_smallest_length() -> None:
    oracle = _oracle()

    def candidate(length: int, nmi: float, support: int) -> dict[str, object]:
        return {
            "state_count": 2,
            "prefix_length": length,
            "valid": True,
            "soft_regime_nmi": nmi,
            "shared_timestamp_count": support,
        }

    prefixes = tuple(
        candidate(length, 0.8 if length in (3, 5) else 0.7, 95)
        for length in PERMITTED_PREFIX_LENGTHS
    )
    assert (
        oracle.independent_prefix_selection(prefixes, ranked_feature_count=8)["prefix_length"] == 3
    )
    higher_support = tuple(
        candidate(length, 0.8 if length in (3, 5) else 0.7, 100 if length == 5 else 95)
        for length in PERMITTED_PREFIX_LENGTHS
    )
    assert (
        oracle.independent_prefix_selection(higher_support, ranked_feature_count=8)["prefix_length"]
        == 5
    )


def test_prefix_selection_requires_complete_nested_prefixes_and_one_k() -> None:
    oracle = _oracle()

    def candidate(length: int, state_count: int = 2) -> dict[str, object]:
        return {
            "state_count": state_count,
            "prefix_length": length,
            "valid": True,
            "soft_regime_nmi": 0.5,
            "shared_timestamp_count": 10,
        }

    with pytest.raises(ValueError, match="every permitted"):
        oracle.independent_prefix_selection(
            tuple(candidate(length) for length in (2, 4)), ranked_feature_count=4
        )
    with pytest.raises(ValueError, match="different K"):
        oracle.independent_prefix_selection(
            (candidate(2, 2), candidate(3, 3)), ranked_feature_count=3
        )
    with pytest.raises(ValueError, match="permitted range"):
        oracle.independent_prefix_selection((candidate(1),), ranked_feature_count=2)
    invalid_support = tuple(candidate(length) for length in PERMITTED_PREFIX_LENGTHS)
    invalid_support[0]["shared_timestamp_count"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        oracle.independent_prefix_selection(invalid_support, ranked_feature_count=8)
    with pytest.raises(ValueError, match="at least two"):
        oracle.permitted_prefix_lengths(1)


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
    with pytest.raises(ValueError, match="boolean"):
        oracle.independent_valid_fold_aggregation(({"valid": 1},))


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


@pytest.mark.parametrize("state_count", LEGAL_K)
@pytest.mark.parametrize(
    ("field", "values", "expected"),
    (
        ("oos_mean", {"gaussian": -0.5, "gmm": -0.2, "student": -0.5}, "gmm"),
        ("oos_std", {"gaussian": 0.1, "gmm": 0.2, "student": 0.1}, "gaussian"),
        ("oos_worst", {"gaussian": -0.5, "gmm": -0.2, "student": -0.5}, "gmm"),
        ("bic_mean", {"gaussian": 10.0, "gmm": 8.0, "student": 8.0}, "gmm"),
        ("aic_mean", {"gaussian": 10.0, "gmm": 8.0, "student": 8.0}, "gmm"),
    ),
)
def test_same_vector_family_ranking_covers_every_k_and_each_tie_stage(
    state_count: int,
    field: str,
    values: dict[str, float],
    expected: str,
) -> None:
    oracle = _oracle()
    candidates = []
    for candidate_id, family in zip(("student", "gmm", "gaussian"), FAMILIES, strict=True):
        item = _family(candidate_id, family, state_count=state_count)
        item[field] = values[candidate_id]
        candidates.append(item)
    result = oracle.independent_same_vector_family_ranking(tuple(reversed(candidates)))
    assert result["winner_candidate_id"] == expected


def test_same_vector_family_ranking_is_completion_order_invariant_and_uses_id_tie_break() -> None:
    oracle = _oracle()
    candidates = tuple(
        _family(candidate_id, family, state_count=5)
        for candidate_id, family in zip(("zeta", "alpha", "mu"), FAMILIES, strict=True)
    )
    outputs = {
        oracle.independent_same_vector_family_ranking(order)["ranked_candidate_ids"]
        for order in itertools.permutations(candidates)
    }
    assert outputs == {("alpha", "mu", "zeta")}


def test_family_numeric_tolerance_is_a_tie_before_the_next_rule() -> None:
    oracle = _oracle()
    candidates = (
        _family("gaussian", "gaussian_hmm", oos_mean=-1.0, oos_std=0.2),
        _family("gmm", "gmm_hmm", oos_mean=-1.0 + 5.0e-13, oos_std=0.3),
        _family("student", "student_t_hmm", oos_mean=-1.0, oos_std=0.2),
    )
    result = oracle.independent_same_vector_family_ranking(candidates)
    assert result["winner_candidate_id"] == "gaussian"


def test_same_vector_family_ranking_rejects_ineligible_and_adversarial_candidates() -> None:
    oracle = _oracle()
    ineligible = tuple(
        _family(name, family, valid_fold_count=2)
        for name, family in zip(("gaussian", "gmm", "student"), FAMILIES, strict=True)
    )
    with pytest.raises(ValueError, match="no same-vector"):
        oracle.independent_same_vector_family_ranking(ineligible)
    result = oracle.independent_same_vector_family_ranking(
        (
            _family("gaussian", "gaussian_hmm", oos_mean=math.nan),
            _family("gmm", "gmm_hmm"),
            _family("student", "student_t_hmm"),
        )
    )
    assert result["rejected"] == {"gaussian": "missing or nonfinite ranking metric"}
    with pytest.raises(ValueError, match="no same-vector"):
        oracle.independent_same_vector_family_ranking(
            tuple(
                _family(name, family, oos_mean=math.nan)
                for name, family in zip(("gaussian", "gmm", "student"), FAMILIES, strict=True)
            )
        )
    with pytest.raises(ValueError, match="exact provenance"):
        oracle.independent_same_vector_family_ranking(
            (
                _family("gaussian", "gaussian_hmm"),
                _family("gmm", "gmm_hmm", feature_order_hash=_hash("d")),
                _family("student", "student_t_hmm"),
            )
        )
    with pytest.raises(ValueError, match="unique fold IDs"):
        oracle.independent_same_vector_family_ranking(
            (
                _family("gaussian", "gaussian_hmm", fold_ids=("outer-001", "outer-001")),
                _family("gmm", "gmm_hmm"),
                _family("student", "student_t_hmm"),
            )
        )


def test_oracle_rejects_cross_dimension_raw_metrics_and_bad_provenance() -> None:
    oracle = _oracle()
    for metric in ("pll", "loglik", "aic", "bic", "hqc"):
        with pytest.raises(ValueError, match="cannot be compared"):
            oracle.reject_cross_dimension_metric_comparison(
                metric, (_hash("a"), _hash("b")), state_counts=(2, 3)
            )
    with pytest.raises(ValueError, match="cannot be compared"):
        oracle.reject_cross_dimension_metric_comparison(
            "aic", (_hash("a"), _hash("a")), state_counts=(2, 5)
        )
    oracle.reject_cross_dimension_metric_comparison(
        "cross_k_score", (_hash("a"), _hash("b")), state_counts=(2, 5)
    )
    with pytest.raises(ValueError, match="align"):
        oracle.reject_cross_dimension_metric_comparison(
            "bic", (_hash("a"), _hash("b")), state_counts=(2,)
        )
    with pytest.raises(ValueError, match="must not be empty"):
        oracle.reject_cross_dimension_metric_comparison("aic", ())
    with pytest.raises(ValueError, match="exactly K"):
        oracle.verify_k_slot_provenance(_provenance_slots()[:-1])
    mutated = _provenance_slots()
    mutated[2]["folds"][0]["source_data_sha256"] = _hash("e")  # type: ignore[index]
    with pytest.raises(ValueError, match="source hash"):
        oracle.verify_k_slot_provenance(mutated)


def test_provenance_checks_every_slot_fold_link_and_rejects_adversarial_mutations() -> None:
    oracle = _oracle()
    report = oracle.verify_k_slot_provenance(_provenance_slots())
    assert tuple(item["state_count"] for item in report["slots"]) == LEGAL_K
    mutations = (
        (("folds", 0, "state_count"), 99, "fold K"),
        (("folds", 0, "slot_id"), "wrong-slot", "fold slot_id"),
        (("folds", 0, "feature_order_hash"), _hash("z"), "fold feature"),
        (("folds", 0, "source_build_id"), "other-build", "source build"),
    )
    for path, value, message in mutations:
        dossier = _provenance_slots()
        dossier[0][path[0]][path[1]][path[2]] = value  # type: ignore[index]
        with pytest.raises(ValueError, match=message):
            oracle.verify_k_slot_provenance(dossier)
    duplicate_fold = _provenance_slots()
    duplicate_fold[0]["folds"].append(dict(duplicate_fold[0]["folds"][0]))  # type: ignore[index]
    with pytest.raises(ValueError, match="fold IDs"):
        oracle.verify_k_slot_provenance(duplicate_fold)


def test_dossier_report_records_command_version_seed_hash_and_exit_code(tmp_path: Path) -> None:
    oracle = _oracle()
    dossier = {"seed": 42, "slots": _provenance_slots()}
    report = oracle.audit_dossier(dossier, command=("verify_k_champion_math.py", "dossier.json"))
    assert report["status"] == "verified"
    assert report["oracle_version"] == "k_champion_math.v1"
    assert report["python_version"] == ".".join(str(value) for value in sys.version_info[:3])
    assert report["command"] == ("verify_k_champion_math.py", "dossier.json")
    assert report["exit_code"] == 0
    assert report["seed"] == 42
    mutated = json.loads(json.dumps(dossier))
    mutated["slots"][0]["source_build_id"] = "build-mutated"
    with pytest.raises(ValueError, match="source build"):
        oracle.audit_dossier(mutated)
    path = tmp_path / "dossier.json"
    path.write_text(json.dumps(dossier), encoding="utf-8")
    assert oracle.canonical_hash(dossier) == report["input_canonical_sha256"]
    assert oracle.canonical_hash({**dossier, "seed": 43}) != report["input_canonical_sha256"]
    valid_mutations = (
        ("feature_order_hash", _hash("d")),
        ("source_build_id", "build-2026-09-16"),
        ("source_data_sha256", _hash("e")),
    )
    for field, value in valid_mutations:
        changed = json.loads(json.dumps(dossier))
        changed["slots"][0][field] = value
        changed["slots"][0]["folds"][0][field] = value
        changed_report = oracle.audit_dossier(changed)
        assert changed_report["input_canonical_sha256"] != report["input_canonical_sha256"]
    policy_changed = json.loads(json.dumps(dossier))
    policy_changed["slots"][0]["policy_version"] = "k_champion_policy.v2"
    assert (
        oracle.audit_dossier(policy_changed)["input_canonical_sha256"]
        != report["input_canonical_sha256"]
    )
    fold_changed = json.loads(json.dumps(dossier))
    fold_changed["slots"][0]["folds"][0]["fold_id"] = "outer-002"
    assert (
        oracle.audit_dossier(fold_changed)["input_canonical_sha256"]
        != report["input_canonical_sha256"]
    )


def test_independent_oracle_does_not_import_production_selection_code() -> None:
    source = (Path(__file__).parents[2] / "scripts" / "verify_k_champion_math.py").read_text(
        encoding="utf-8"
    )
    assert "from market_regime_engine" not in source
    assert "import market_regime_engine" not in source
