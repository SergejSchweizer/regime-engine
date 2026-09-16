from __future__ import annotations

import json
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta

import pytest

from market_regime_engine.contracts.core import (
    K_CHAMPION_ALIASES,
    K_CHAMPION_POLICY_ID,
    K_CHAMPION_POLICY_VERSION,
    K_CHAMPION_SLOT_IDS,
)
from market_regime_engine.evaluations.k_champion_contract import (
    K_SLOT_COMPARISON_DOMAIN,
    K_SLOT_PROMOTION_SCORE_VERSION,
    LEGAL_MODEL_FAMILIES,
    KChampionPromotionCandidate,
    KChampionPromotionEvidence,
    KChampionSelection,
    feature_order_hash,
    rank_k_slot_candidates,
)
from market_regime_engine.evaluations.k_feature_selection import (
    KFeatureSelectionPayload,
    run_k_feature_selection,
)
from market_regime_engine.mlflow_support.k_champion_contract import (
    KChampionPromotionInstruction,
    build_promotion_instruction,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def selection(
    *, slot_id: str = "k2", feature_order: tuple[str, ...] = ("f0", "f1")
) -> KFeatureSelectionPayload:
    state_count = int(slot_id[1:])
    return KChampionSelection(
        slot_id=slot_id,
        state_count=state_count,
        model_family="gaussian_hmm",
        candidate_identity=f"gaussian_hmm_k{state_count}_full",
        feature_order=feature_order,
        feature_order_hash=feature_order_hash(feature_order),
        source_snapshot_id="snapshot-1",
        profile_id="xetra",
        profile_config_version=4,
        policy_id="k_champion_portfolio",
        policy_version="k_champion_portfolio.v1",
        validation_cutoff=BASE,
        deployment_cutoff=BASE + timedelta(days=1),
        comparison_domain_id="k_slot_promotion.v1",
        promotion_score_version="k_slot_promotion.v1",
        reference_teacher_id=f"teacher-{state_count}",
        artifact_hash="a" * 64,
    )


def candidate(
    *, feature_order: tuple[str, ...] = ("f0", "f1"), nmi: float = 0.7
) -> KChampionPromotionCandidate:
    item = selection(feature_order=feature_order)
    return KChampionPromotionCandidate(
        selection=item,
        evidence=KChampionPromotionEvidence(
            slot_id="k2",
            source_snapshot_id="snapshot-1",
            profile_id="xetra",
            profile_config_version=4,
            policy_version="k_champion_portfolio.v1",
            validation_cutoff=BASE,
            reference_teacher_id="teacher-2",
            feature_order_hash=item.feature_order_hash,
            planned_fold_count=4,
            valid_fold_count=4,
            latest_complete_fold_valid=True,
            mean_soft_regime_nmi=nmi,
            worst_fold_soft_regime_nmi=nmi,
            common_support=0.9,
            stability=0.8,
        ),
    )


def test_contract_hashes_and_promotion_are_slot_local() -> None:
    decision = rank_k_slot_candidates(
        (
            candidate(feature_order=("f0", "f1"), nmi=0.7),
            candidate(feature_order=("f0", "f2"), nmi=0.8),
        )
    )
    assert decision.winner is not None
    assert decision.winner.selection.feature_order == ("f0", "f2")
    instruction = build_promotion_instruction(decision, exact_model_version="7")
    assert instruction.alias == "champion-k2"
    assert instruction.exact_model_version == "7"


def test_cross_k_and_ineligible_promotion_fail_closed() -> None:
    with pytest.raises(ValueError, match="cross-K"):
        rank_k_slot_candidates(
            (
                candidate(),
                KChampionPromotionCandidate(
                    selection=selection(slot_id="k3"),
                    evidence=KChampionPromotionEvidence(
                        slot_id="k3",
                        source_snapshot_id="snapshot-1",
                        profile_id="xetra",
                        profile_config_version=4,
                        policy_version="k_champion_portfolio.v1",
                        validation_cutoff=BASE,
                        reference_teacher_id="teacher-3",
                        feature_order_hash=feature_order_hash(("f0", "f1")),
                        planned_fold_count=4,
                        valid_fold_count=4,
                        latest_complete_fold_valid=True,
                        mean_soft_regime_nmi=0.8,
                        worst_fold_soft_regime_nmi=0.8,
                        common_support=0.9,
                        stability=0.8,
                    ),
                ),
            )
        )
    invalid = candidate()
    invalid = KChampionPromotionCandidate(
        selection=invalid.selection,
        evidence=KChampionPromotionEvidence(
            slot_id="k2",
            source_snapshot_id="snapshot-1",
            profile_id="xetra",
            profile_config_version=4,
            policy_version="k_champion_portfolio.v1",
            validation_cutoff=BASE,
            reference_teacher_id="teacher-2",
            feature_order_hash=invalid.selection.feature_order_hash,
            planned_fold_count=4,
            valid_fold_count=2,
            latest_complete_fold_valid=False,
            mean_soft_regime_nmi=None,
            worst_fold_soft_regime_nmi=None,
            common_support=None,
            stability=None,
            rejection_reasons=("fewer than three valid folds",),
        ),
    )
    decision = rank_k_slot_candidates((invalid,))
    assert decision.winner is None
    with pytest.raises(ValueError, match="ineligible"):
        build_promotion_instruction(decision, exact_model_version="1")


def _train_selector(
    train_rows,
    *,
    state_count: int,
    source_snapshot_id: str,
    validation_cutoff: datetime,
) -> KChampionSelection:
    del train_rows
    result = selection(slot_id=f"k{state_count}")
    return KFeatureSelectionPayload(
        selection=KChampionSelection(
            slot_id=result.slot_id,
            state_count=result.state_count,
            model_family=result.model_family,
            candidate_identity=result.candidate_identity,
            feature_order=result.feature_order,
            feature_order_hash=result.feature_order_hash,
            source_snapshot_id=source_snapshot_id,
            profile_id=result.profile_id,
            profile_config_version=result.profile_config_version,
            policy_id=result.policy_id,
            policy_version=result.policy_version,
            validation_cutoff=validation_cutoff,
            deployment_cutoff=validation_cutoff + timedelta(days=1),
            comparison_domain_id=result.comparison_domain_id,
            promotion_score_version=result.promotion_score_version,
            reference_teacher_id=result.reference_teacher_id,
            artifact_hash=result.artifact_hash,
        ),
        discovery_hash="b" * 64,
        teacher_identity=f"teacher-{state_count}",
        prefix_evidence_hash="c" * 64,
    )


def test_train_only_selector_runs_one_independent_process_task_per_k() -> None:
    import pandas as pd

    results = run_k_feature_selection(
        pd.DataFrame({"f0": (1.0, 2.0), "f1": (2.0, 1.0)}),
        source_snapshot_id="snapshot-1",
        validation_cutoff=BASE,
        selector=_train_selector,
        max_workers=2,
    )
    assert tuple(item.state_count for item in results) == (2, 3, 4, 5)
    assert all(item.eligible for item in results)

    serial = run_k_feature_selection(
        pd.DataFrame({"f0": (1.0, 2.0), "f1": (2.0, 1.0)}),
        source_snapshot_id="snapshot-1",
        validation_cutoff=BASE,
        selector=_train_selector,
        max_workers=1,
    )
    assert serial == results
    assert tuple(item.selection_hash for item in serial) == tuple(
        item.selection_hash for item in results
    )


def test_selection_canonical_round_trip_has_an_exact_decision_only_schema() -> None:
    item = selection()
    payload = json.loads(item.canonical_json)

    assert set(payload) == {field.name for field in fields(KChampionSelection)}
    assert not {
        "mlflow_run_id",
        "created_at",
        "updated_at",
        "temporary_path",
        "raw_pll",
        "aic",
        "bic",
    }.intersection(payload)
    assert payload["validation_cutoff"].endswith("Z")
    assert payload["deployment_cutoff"].endswith("Z")
    assert KChampionSelection.from_canonical_json(item.canonical_json) == item
    assert KChampionSelection.from_canonical_json(item.canonical_json).canonical_json == (
        item.canonical_json
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"slot_id": "k1", "state_count": 1}, "slot_id"),
        ({"state_count": 3}, "state_count"),
        ({"profile_config_version": "4"}, "profile_config_version"),
        ({"feature_order": []}, "feature_order"),
        ({"feature_order": ["f0", "f0"]}, "feature_order"),
        ({"feature_order_hash": ""}, "feature_order_hash"),
        ({"artifact_hash": "not-a-hash"}, "artifact_hash"),
        ({"comparison_domain_id": "same_feature_vector.v1"}, "promotion domain"),
        ({"policy_version": "k_champion_portfolio.v2"}, "policy version"),
        ({"promotion_score_version": "k_slot_promotion.v2"}, "score version"),
    ],
)
def test_selection_deserialization_rejects_malformed_identity_fields(
    mutation: dict[str, object], message: str
) -> None:
    payload = json.loads(selection().canonical_json)
    payload.update(mutation)

    with pytest.raises(ValueError, match=message):
        KChampionSelection.from_canonical_json(json.dumps(payload))


def test_selection_deserialization_rejects_missing_unknown_and_operational_fields() -> None:
    payload = json.loads(selection().canonical_json)
    del payload["artifact_hash"]
    with pytest.raises(ValueError, match="unknown or missing"):
        KChampionSelection.from_canonical_json(json.dumps(payload))

    payload = json.loads(selection().canonical_json)
    payload["mlflow_run_id"] = "operational-only"
    with pytest.raises(ValueError, match="unknown or missing"):
        KChampionSelection.from_canonical_json(json.dumps(payload))

    payload = json.loads(selection().canonical_json)
    payload["validation_cutoff"] = BASE.isoformat()
    with pytest.raises(ValueError, match="UTC Z"):
        KChampionSelection.from_canonical_json(json.dumps(payload))


def test_policy_slots_aliases_and_model_families_are_exact_and_single_source() -> None:
    assert K_CHAMPION_POLICY_ID == "k_champion_portfolio"
    assert K_CHAMPION_POLICY_VERSION == "k_champion_portfolio.v1"
    assert K_CHAMPION_SLOT_IDS == ("k2", "k3", "k4", "k5")
    assert K_CHAMPION_ALIASES == (
        "champion-k2",
        "champion-k3",
        "champion-k4",
        "champion-k5",
    )
    assert LEGAL_MODEL_FAMILIES == ("gaussian_hmm", "gmm_hmm", "student_t_hmm")

    for family in LEGAL_MODEL_FAMILIES:
        assert replace(selection(), model_family=family).model_family == family
    with pytest.raises(ValueError, match="model_family"):
        replace(selection(), model_family="unsupported_hmm")


def test_every_decision_identity_mutation_changes_hash_or_fails_closed() -> None:
    item = selection()
    valid_mutations = (
        replace(
            item,
            slot_id="k3",
            state_count=3,
            candidate_identity="gaussian_hmm_k3_full",
            reference_teacher_id="teacher-3",
        ),
        replace(
            item,
            feature_order=("f1", "f0"),
            feature_order_hash=feature_order_hash(("f1", "f0")),
        ),
        replace(item, source_snapshot_id="snapshot-2"),
        replace(item, artifact_hash="b" * 64),
    )
    assert len({item.selection_hash, *(value.selection_hash for value in valid_mutations)}) == 5
    assert len({item.idempotency_key, *(value.idempotency_key for value in valid_mutations)}) == 5

    with pytest.raises(ValueError, match="policy version"):
        replace(item, policy_version="k_champion_portfolio.v2")
    with pytest.raises(ValueError, match="score version"):
        replace(item, promotion_score_version="k_slot_promotion.v2")


def test_promotion_ranking_uses_only_the_versioned_dimension_independent_tuple() -> None:
    lower_rate = candidate(feature_order=("f0", "f1"), nmi=0.99)
    lower_rate = replace(
        lower_rate,
        evidence=replace(
            lower_rate.evidence,
            planned_fold_count=5,
            valid_fold_count=4,
        ),
    )
    full_rate = candidate(feature_order=("f0", "f2"), nmi=0.10)
    decision = rank_k_slot_candidates((lower_rate, full_rate))
    assert decision.winner == full_rate

    mean_winner = candidate(feature_order=("f0", "f3"), nmi=0.80)
    mean_loser = candidate(feature_order=("f0", "f4"), nmi=0.70)
    assert rank_k_slot_candidates((mean_loser, mean_winner)).winner == mean_winner

    tied_left = candidate(feature_order=("f0", "f5"), nmi=0.75)
    tied_right = candidate(feature_order=("f0", "f6"), nmi=0.75)
    forward = rank_k_slot_candidates((tied_left, tied_right)).winner
    reverse = rank_k_slot_candidates((tied_right, tied_left)).winner
    assert forward is not None and reverse is not None
    assert forward.selection.selection_hash == reverse.selection.selection_hash


def test_alias_slot_mismatch_and_legacy_champion_instruction_are_impossible() -> None:
    item = selection()
    with pytest.raises(ValueError, match="does not match"):
        KChampionPromotionInstruction(
            model_name="regime-xetra",
            slot_id="k2",
            alias="champion-k3",
            exact_model_version="1",
            expected_current_version=None,
            selection_hash=item.selection_hash,
            idempotency_key=item.idempotency_key,
            reason="invalid alias",
        )
    with pytest.raises(ValueError, match="does not match"):
        KChampionPromotionInstruction(
            model_name="regime-xetra",
            slot_id="k2",
            alias="champion",
            exact_model_version="1",
            expected_current_version=None,
            selection_hash=item.selection_hash,
            idempotency_key=item.idempotency_key,
            reason="legacy mutation",
        )

    with pytest.raises(ValueError, match="selection_hash"):
        KChampionPromotionInstruction(
            model_name="regime-xetra",
            slot_id="k2",
            alias="champion-k2",
            exact_model_version="1",
            expected_current_version=None,
            selection_hash="not-a-hash",
            idempotency_key=item.idempotency_key,
            reason="malformed audit identity",
        )


def test_selection_and_promotion_versions_are_bound_to_the_only_supported_contract() -> None:
    item = selection()
    assert item.policy_id == K_CHAMPION_POLICY_ID
    assert item.policy_version == K_CHAMPION_POLICY_VERSION
    assert item.comparison_domain_id == K_SLOT_COMPARISON_DOMAIN
    assert item.promotion_score_version == K_SLOT_PROMOTION_SCORE_VERSION
