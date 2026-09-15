from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from market_regime_engine.evaluations.k_champion_contract import (
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
from market_regime_engine.mlflow_support.k_champion_contract import build_promotion_instruction

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
