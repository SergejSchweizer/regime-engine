from __future__ import annotations

from datetime import UTC, datetime, timedelta
from time import sleep

import numpy as np
import pandas as pd
import pytest

import market_regime_engine.evaluation.walk_forward as walk_forward
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
from market_regime_engine.models.protocols import FitResult
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.training.multistart import (
    MULTISTART_SEEDS,
    MultistartResult,
    StartDiagnostic,
)

pytestmark = pytest.mark.integration


def _source_rows() -> pd.DataFrame:
    row_count = 1323  # one complete 1260/63 outer fold
    start = datetime(2020, 1, 1, tzinfo=UTC)
    index = np.arange(row_count, dtype=np.float64)
    regime = np.random.default_rng(2031).integers(0, 3, size=row_count)
    first_axis = np.asarray((-1.5, 0.0, 1.5))[regime]
    second_axis = np.asarray((1.0, -1.0, 0.5))[regime]
    noise = np.random.default_rng(2032)
    return pd.DataFrame(
        {
            "timestamp_m1": tuple(start + timedelta(days=int(value)) for value in index),
            "f0": first_axis + 0.10 * noise.normal(size=row_count),
            "f1": second_axis + 0.10 * noise.normal(size=row_count),
            **{
                f"pca_pc_{component:03d}": np.sin(index / (17.0 + component))
                for component in range(1, 9)
            },
        }
    )


def _one_real_fit_multistart(train_rows, *, state_count, adapter_factory, **_kwargs):
    """Use one real fit while retaining the production multistart contract."""

    result: FitResult = adapter_factory().fit(train_rows, state_count, MULTISTART_SEEDS[0])
    diagnostics = tuple(
        StartDiagnostic(
            seed=seed,
            success=True,
            converged=True,
            iterations=result.iterations,
            train_log_likelihood=result.train_log_likelihood,
            artifact=result.artifact,
            failure_reason=None,
        )
        for seed in MULTISTART_SEEDS
    )
    return MultistartResult(
        state_count=state_count,
        winner=result,
        diagnostics=diagnostics,
    )


def _invalid_family_runner(frame, candidate_plan, candidate_profile, candidate, adapter):
    """Pickle-safe runner with deliberately reversed family completion latency."""

    del frame, adapter
    sleep({"gaussian_hmm": 0.03, "gmm_hmm": 0.02, "student_t_hmm": 0.01}[candidate.model_family])
    fold = candidate_plan.folds[0]
    invalid_fold = WalkForwardFoldResult(
        fold_id=fold.fold_id,
        fold_index=fold.fold_index,
        valid=False,
        failure_reason=f"fixture rejected {candidate.model_family}",
        train_source_observation_count=fold.train_source_observations,
        test_source_observation_count=fold.test_source_observations,
        train_model_observation_count=0,
        test_model_observation_count=0,
        skipped_train_incomplete_count=fold.train_source_observations,
        skipped_test_incomplete_count=fold.test_source_observations,
    )
    return WalkForwardEvaluation(
        profile_id=candidate_profile.profile_id,
        profile_config_version=candidate_profile.profile_config_version,
        candidate_id=candidate.candidate_id,
        state_count=candidate.state_count,
        source_build_id=candidate.source_build_id,
        feature_order=candidate.feature_order,
        feature_selection_definition_hash=candidate.feature_selection_definition_hash,
        feature_selection_execution_hash=candidate.feature_selection_execution_hash,
        evaluation_plan_hash=candidate_plan.plan_hash,
        evaluation_cutoff=candidate_plan.evaluation_cutoff,
        folds=(invalid_fold,),
    )


@pytest.mark.parametrize("state_count", (2, 3, 4, 5))
def test_k_family_grid_runs_real_gaussian_gmm_and_student_t_for_each_k(
    monkeypatch: pytest.MonkeyPatch, state_count: int
) -> None:
    monkeypatch.setattr(walk_forward, "run_multistart", _one_real_fit_multistart)
    rows = _source_rows()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    universe = tuple(column for column in rows.columns if column != "timestamp_m1")

    result = evaluate_k_family_grid(
        rows,
        state_count=state_count,
        feature_order=("f0", "f1"),
        original_feature_universe=universe,
        profile=profile,
        plan=plan,
        source_build_id="synthetic-k-family-build",
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        max_workers=3,
        pca_raw_feature_order=("f0", "f1"),
    )
    serial = evaluate_k_family_grid(
        rows,
        state_count=state_count,
        feature_order=("f0", "f1"),
        original_feature_universe=universe,
        profile=profile,
        plan=plan,
        source_build_id="synthetic-k-family-build",
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        max_workers=1,
        pca_raw_feature_order=("f0", "f1"),
    )

    expected_ids = tuple(
        (
            f"gmm_hmm_k{state_count}_m2_full"
            if family == "gmm_hmm"
            else f"{family}_k{state_count}_full"
        )
        for family in K_FAMILY_ORDER
    )
    assert tuple(item.candidate_id for item in result.evaluations) == expected_ids
    assert tuple(item.candidate_id for item in result.aggregates) == expected_ids
    assert all(item.state_count == state_count for item in result.evaluations)
    assert result.selection is not None or result.no_selection_reason
    assert result == serial
    assert result.evidence_hash == serial.evidence_hash
    assert all(item.feature_order == ("f0", "f1") for item in result.evaluations)
    assert all(item.evaluation_plan_hash == plan.plan_hash for item in result.evaluations)
    assert all(item.feature_selection_definition_hash == "a" * 64 for item in result.evaluations)
    assert all(item.feature_selection_execution_hash == "b" * 64 for item in result.evaluations)


def test_k_family_grid_retains_precise_invalid_evidence_for_every_family() -> None:
    rows = _source_rows()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    universe = tuple(column for column in rows.columns if column != "timestamp_m1")

    result = evaluate_k_family_grid(
        rows,
        state_count=2,
        feature_order=("f0", "f1"),
        original_feature_universe=universe,
        profile=profile,
        plan=plan,
        source_build_id="synthetic-k-family-build",
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        runner=_invalid_family_runner,
        max_workers=1,
        pca_raw_feature_order=("f0", "f1"),
    )

    assert result.selection is None
    assert result.no_selection_reason
    assert all(aggregate.valid_fold_count == 0 for aggregate in result.aggregates)
    assert all(
        evaluation.folds[0].failure_reason == f"fixture rejected {family}"
        for evaluation, family in zip(result.evaluations, K_FAMILY_ORDER, strict=True)
    )


def test_all_k_family_grids_are_process_independent_and_canonically_assembled() -> None:
    rows = _source_rows()
    profile = load_profile("configs/profiles/xetra_v4.yaml")
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    universe = tuple(column for column in rows.columns if column != "timestamp_m1")
    requests = tuple(
        KFamilyGridRequest(
            state_count=state_count,
            feature_order=("f0", "f1") if state_count % 2 == 0 else ("f1", "f0"),
            feature_selection_definition_hash=f"{state_count}" * 64,
            feature_selection_execution_hash=f"{state_count + 4}" * 64,
        )
        for state_count in (5, 4, 3, 2)
    )
    kwargs = dict(
        source_rows=rows,
        requests=requests,
        original_feature_universe=universe,
        profile=profile,
        plan=plan,
        source_build_id="synthetic-k-family-build",
        runner=_invalid_family_runner,
        pca_raw_feature_order=("f0", "f1"),
    )

    parallel = evaluate_k_family_grids(**kwargs, max_workers=4)
    constrained = evaluate_k_family_grids(**kwargs, max_workers=2)
    serial = evaluate_k_family_grids(**kwargs, max_workers=1)

    assert tuple(result.state_count for result in parallel) == (2, 3, 4, 5)
    assert parallel == serial
    assert constrained == serial
    assert tuple(result.evidence_hash for result in parallel) == tuple(
        result.evidence_hash for result in serial
    )
    assert tuple(result.feature_order for result in parallel) == (
        ("f0", "f1"),
        ("f1", "f0"),
        ("f0", "f1"),
        ("f1", "f0"),
    )

    k2, k3, *_ = parallel
    with pytest.raises(ValueError, match="exact ordered Gaussian/GMM/Student-t"):
        rank_k_family_candidates(
            tuple(reversed(k2.evaluations)),
            k2.aggregates,
        )
    with pytest.raises(ValueError, match="exact ordered Gaussian/GMM/Student-t"):
        rank_k_family_candidates(k2.evaluations[:-1], k2.aggregates[:-1])
    with pytest.raises(ValueError, match="exact ordered Gaussian/GMM/Student-t"):
        rank_k_family_candidates(
            (*k2.evaluations, k2.evaluations[-1]),
            (*k2.aggregates, k2.aggregates[-1]),
        )
    with pytest.raises(ValueError, match="cross-K comparison"):
        rank_k_family_candidates(
            (k2.evaluations[0], *k3.evaluations[1:]),
            (k2.aggregates[0], *k3.aggregates[1:]),
        )
