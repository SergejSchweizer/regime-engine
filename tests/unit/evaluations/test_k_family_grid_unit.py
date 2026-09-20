from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

import market_regime_engine.evaluations.k_family_grid as module
from market_regime_engine.evaluation.walk_forward import (
    WalkForwardEvaluation,
    WalkForwardFoldResult,
)
from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.evaluations.k_family_grid import (
    K_FAMILY_ORDER,
    KFamilyGridRequest,
    evaluate_k_family_grid,
    evaluate_k_family_grids,
    rank_k_family_candidates,
)
from market_regime_engine.profiles.loader import load_profile

HASH = "a" * 64


def _inputs() -> tuple[pd.DataFrame, object, object]:
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    timestamps = tuple(
        datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=index) for index in range(1323)
    )
    return (
        pd.DataFrame({"timestamp_m1": timestamps, "f0": 0.0, "f1": 1.0}),
        profile,
        plan_walk_forward(timestamps, profile.walk_forward),
    )


def _invalid_runner(frame, plan, profile, candidate, adapter):
    del frame, adapter
    fold = plan.folds[0]
    invalid = WalkForwardFoldResult(
        fold_id=fold.fold_id,
        fold_index=fold.fold_index,
        valid=False,
        failure_reason="unit fixture",
        train_source_observation_count=fold.train_source_observations,
        test_source_observation_count=fold.test_source_observations,
        train_model_observation_count=0,
        test_model_observation_count=0,
        skipped_train_incomplete_count=fold.train_source_observations,
        skipped_test_incomplete_count=fold.test_source_observations,
    )
    return WalkForwardEvaluation(
        profile_id=profile.profile_id,
        profile_config_version=profile.profile_config_version,
        candidate_id=candidate.candidate_id,
        state_count=candidate.state_count,
        source_build_id=candidate.source_build_id,
        feature_order=candidate.feature_order,
        feature_selection_definition_hash=candidate.feature_selection_definition_hash,
        feature_selection_execution_hash=candidate.feature_selection_execution_hash,
        evaluation_plan_hash=plan.plan_hash,
        evaluation_cutoff=plan.evaluation_cutoff,
        folds=(invalid,),
    )


def _request(state_count: int) -> KFamilyGridRequest:
    return KFamilyGridRequest(
        state_count=state_count,
        feature_order=("f0", "f1"),
        feature_selection_definition_hash=HASH,
        feature_selection_execution_hash="b" * 64,
    )


def test_k_family_request_rejects_invalid_k_hashes_and_duplicate_features() -> None:
    with pytest.raises(ValueError, match="only K=2,3,4,5"):
        _request(6)
    with pytest.raises(ValueError, match="non-empty and unique"):
        KFamilyGridRequest(2, ("f0", "f0"), HASH, "b" * 64)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        KFamilyGridRequest(2, ("f0",), "A" * 64, "b" * 64)


def test_fixed_k_grid_unit_runner_preserves_order_and_invalid_evidence() -> None:
    rows, profile, plan = _inputs()
    result = evaluate_k_family_grid(
        rows,
        state_count=2,
        feature_order=("f0", "f1"),
        original_feature_universe=("f0", "f1", "pca_pc_001"),
        profile=profile,
        plan=plan,
        source_build_id="unit-build",
        feature_selection_definition_hash=HASH,
        feature_selection_execution_hash="b" * 64,
        runner=_invalid_runner,
        max_workers=1,
        pca_raw_feature_order=("f0", "f1"),
    )
    assert result.selection is None
    assert result.no_selection_reason
    assert tuple(item.candidate_id for item in result.evaluations) == tuple(
        ("gmm_hmm_k2_m2_full" if family == "gmm_hmm" else f"{family}_k2_full")
        for family in K_FAMILY_ORDER
    )
    assert all(item.valid_fold_count == 0 for item in result.aggregates)


def test_fixed_k_grid_rejects_bad_runner_and_pca_universe() -> None:
    rows, profile, plan = _inputs()
    kwargs = dict(
        source_rows=rows,
        state_count=2,
        feature_order=("f0", "f1"),
        original_feature_universe=("f0", "f1", "pca_pc_001"),
        profile=profile,
        plan=plan,
        source_build_id="unit-build",
        feature_selection_definition_hash=HASH,
        feature_selection_execution_hash="b" * 64,
        max_workers=1,
    )
    with pytest.raises(ValueError, match="only K=2,3,4,5"):
        evaluate_k_family_grid(**{**kwargs, "state_count": 1})
    with pytest.raises(ValueError, match="PCA"):
        evaluate_k_family_grid(**{**kwargs, "pca_raw_feature_order": ("missing",)})
    with pytest.raises(TypeError, match="callable"):
        evaluate_k_family_grid(**{**kwargs, "runner": object()})


def test_k_family_portfolio_requires_exact_k_set_and_is_canonical() -> None:
    rows, profile, plan = _inputs()
    kwargs = dict(
        source_rows=rows,
        original_feature_universe=("f0", "f1", "pca_pc_001"),
        profile=profile,
        plan=plan,
        source_build_id="unit-build",
        runner=_invalid_runner,
        pca_raw_feature_order=("f0", "f1"),
        max_workers=1,
    )
    with pytest.raises(ValueError, match="exactly K=2,3,4,5"):
        evaluate_k_family_grids(**{**kwargs, "requests": (_request(2),)})
    results = evaluate_k_family_grids(
        **{**kwargs, "requests": tuple(reversed(tuple(_request(k) for k in (2, 3, 4, 5))))}
    )
    assert tuple(result.state_count for result in results) == (2, 3, 4, 5)
    with pytest.raises(ValueError, match="unique K"):
        evaluate_k_family_grids(
            **{
                **kwargs,
                "requests": (_request(2), _request(2), _request(3), _request(4)),
            }
        )


def test_k_family_ranker_rejects_cross_k_and_bad_family_order() -> None:
    rows, profile, plan = _inputs()
    result = evaluate_k_family_grid(
        rows,
        state_count=2,
        feature_order=("f0", "f1"),
        original_feature_universe=("f0", "f1", "pca_pc_001"),
        profile=profile,
        plan=plan,
        source_build_id="unit-build",
        feature_selection_definition_hash=HASH,
        feature_selection_execution_hash="b" * 64,
        runner=_invalid_runner,
        max_workers=1,
        pca_raw_feature_order=("f0", "f1"),
    )
    with pytest.raises(ValueError, match="exact ordered"):
        rank_k_family_candidates(tuple(reversed(result.evaluations)), result.aggregates)


def test_fixed_k_grid_preserves_a_valid_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    rows, profile, plan = _inputs()
    expected = (
        "gaussian_hmm_k2_full",
        "gmm_hmm_k2_m2_full",
        "student_t_hmm_k2_full",
    )

    def fake_rank(evaluations, aggregates):
        return SimpleNamespace(
            champion_state_count=2,
            champion_candidate_id=expected[0],
            ranked_candidate_ids=expected,
            evidence=aggregates,
        )

    monkeypatch.setattr(module, "rank_k_family_candidates", fake_rank)
    result = evaluate_k_family_grid(
        rows,
        state_count=2,
        feature_order=("f0", "f1"),
        original_feature_universe=("f0", "f1", "pca_pc_001"),
        profile=profile,
        plan=plan,
        source_build_id="unit-build",
        feature_selection_definition_hash=HASH,
        feature_selection_execution_hash="b" * 64,
        runner=_invalid_runner,
        max_workers=1,
        pca_raw_feature_order=("f0", "f1"),
    )
    assert result.selection is not None
    assert result.no_selection_reason is None
    assert result.evidence_hash
