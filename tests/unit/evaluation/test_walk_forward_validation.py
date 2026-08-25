from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.evaluation.walk_forward import (
    WalkForwardEvaluation,
    WalkForwardFoldResult,
    _aligned_probabilities,
    _complete_case,
    _finite_optional,
    _fold_source_frames,
    _require_utc,
    _validate_source_rows,
    run_walk_forward_candidate,
)
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan, plan_walk_forward
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile
from market_regime_engine.states.alignment import align_first_fold

PROFILE_CONFIG = Path("configs/profiles/xetra_v1.yaml")


def source_rows(row_count: int = 1323) -> pd.DataFrame:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    return pd.DataFrame(
        {
            "timestamp_m1": tuple(start + timedelta(days=index) for index in range(row_count)),
            "f0": np.arange(row_count, dtype=np.float64),
            "f1": np.arange(row_count, dtype=np.float64) + 1.0,
        }
    )


def candidate() -> ResolvedCandidateProfile:
    return ResolvedCandidateProfile(
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        covariance_type="full",
        feature_order=("f0", "f1"),
        feature_dimension=2,
        source_build_id="build-1",
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        original_feature_universe=tuple(f"f{index}" for index in range(48)),
        preliminary_medoids=tuple(f"f{index}" for index in range(8)),
    )


def artifact() -> GaussianHMMArtifact:
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=("f0", "f1"),
        start_probabilities=(0.5, 0.5),
        transition_matrix=((0.8, 0.2), (0.2, 0.8)),
        means=((-1.0, -1.0), (1.0, 1.0)),
        full_covariances=(
            ((1.0, 0.1), (0.1, 1.0)),
            ((1.0, -0.1), (-0.1, 1.0)),
        ),
    )


def invalid_fold(index: int = 1) -> WalkForwardFoldResult:
    return WalkForwardFoldResult(
        fold_id=f"fold_{index:03d}",
        fold_index=index,
        valid=False,
        failure_reason="diagnostic failure",
        train_source_observation_count=1260,
        test_source_observation_count=63,
        train_model_observation_count=1260,
        test_model_observation_count=63,
        skipped_train_incomplete_count=0,
        skipped_test_incomplete_count=0,
    )


def evaluation(folds: tuple[WalkForwardFoldResult, ...] | None = None) -> WalkForwardEvaluation:
    if folds is None:
        folds = (invalid_fold(),)
    return WalkForwardEvaluation(
        profile_id="xetra",
        profile_config_version=1,
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        source_build_id="build-1",
        feature_order=("f0", "f1"),
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        evaluation_plan_hash="c" * 64,
        evaluation_cutoff=datetime(2026, 1, 1, tzinfo=UTC),
        folds=folds,
    )


def test_scalar_validation_helpers_fail_closed() -> None:
    assert _require_utc(datetime(2026, 1, 1, tzinfo=UTC), "ts").tzinfo is UTC
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        _require_utc(datetime(2026, 1, 1), "ts")
    _finite_optional(None, "metric")
    _finite_optional(1.0, "metric")
    with pytest.raises(ValueError, match="must be finite"):
        _finite_optional(float("inf"), "metric")


def test_fold_result_identity_count_and_failure_invariants() -> None:
    baseline = invalid_fold()
    with pytest.raises(ValueError, match="identity"):
        replace(baseline, fold_id="wrong")
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(baseline, train_model_observation_count=-1)
    with pytest.raises(ValueError, match="TRAIN model/skipped"):
        replace(baseline, train_model_observation_count=1259)
    with pytest.raises(ValueError, match="TEST model/skipped"):
        replace(baseline, test_model_observation_count=62)
    with pytest.raises(ValueError, match="valid fold"):
        replace(baseline, failure_reason=None)
    with pytest.raises(ValueError, match="must be finite"):
        replace(baseline, train_log_likelihood=float("nan"))
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        replace(baseline, oos_timestamps=(datetime(2026, 1, 1),))


def test_evaluation_contract_failures_are_explicit() -> None:
    baseline = evaluation()
    assert baseline.valid_folds == ()
    assert baseline.valid_fold_rate == 0.0
    with pytest.raises(ValueError, match="xetra"):
        replace(baseline, profile_id="other")
    with pytest.raises(ValueError, match="K2/K3/K4/K5"):
        replace(baseline, state_count=6, candidate_id="gaussian_hmm_k6_full")
    with pytest.raises(ValueError, match="candidate identity"):
        replace(baseline, candidate_id="other")
    with pytest.raises(ValueError, match="at least one"):
        replace(baseline, folds=())
    with pytest.raises(ValueError, match="preserve complete planned fold order"):
        replace(baseline, folds=(invalid_fold(2),))
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        replace(baseline, evaluation_cutoff=datetime(2026, 1, 1))
    future = replace(
        invalid_fold(),
        oos_timestamps=(datetime(2026, 1, 2, tzinfo=UTC),),
    )
    with pytest.raises(ValueError, match="beyond evaluation cutoff"):
        replace(baseline, folds=(future,))


def test_source_validation_rejects_missing_nonutc_and_nonmonotonic_rows() -> None:
    rows = source_rows()
    assert len(_validate_source_rows(rows, ("f0", "f1"))) == len(rows)
    with pytest.raises(ValueError, match="timestamp_m1"):
        _validate_source_rows(rows.drop(columns=["timestamp_m1"]), ("f0", "f1"))
    with pytest.raises(ValueError, match="missing resolved features"):
        _validate_source_rows(rows.drop(columns=["f1"]), ("f0", "f1"))
    duplicate = rows.copy()
    duplicate.loc[1, "timestamp_m1"] = duplicate.loc[0, "timestamp_m1"]
    with pytest.raises(ValueError, match="strictly increasing"):
        _validate_source_rows(duplicate, ("f0", "f1"))
    naive = rows.copy()
    naive["timestamp_m1"] = naive["timestamp_m1"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        _validate_source_rows(naive, ("f0", "f1"))


def test_fold_source_frame_bounds_and_complete_case_validation() -> None:
    rows = source_rows()
    profile = load_profile(PROFILE_CONFIG)
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    fold = plan.folds[0]
    train, test = _fold_source_frames(rows, fold)
    assert len(train) == 1260
    assert len(test) == 63
    with pytest.raises(ValueError, match="exceed supplied"):
        _fold_source_frames(rows.iloc[:-1], fold)
    shifted = rows.copy()
    shifted.loc[1260, "timestamp_m1"] += timedelta(hours=1)
    with pytest.raises(ValueError, match="do not match evaluation plan"):
        _fold_source_frames(shifted, fold)

    matrix, timestamps, skipped = _complete_case(test, ("f0", "f1"))
    assert matrix.shape == (63, 2)
    assert len(timestamps) == 63
    assert skipped == 0
    nonnumeric = test.copy()
    nonnumeric.loc[1260, "f0"] = "bad"
    with pytest.raises(ValueError, match="must be numeric"):
        _complete_case(nonnumeric, ("f0", "f1"))
    nonfinite = test.copy()
    nonfinite.loc[1260, "f0"] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        _complete_case(nonfinite, ("f0", "f1"))


def test_alignment_probability_validation_rejects_nonfinite_result() -> None:
    alignment = align_first_fold(artifact())
    probabilities = np.asarray(((0.5, 0.5), (np.inf, 0.0)), dtype=np.float64)
    with pytest.raises(ValueError, match="aligned filtered probabilities must be finite"):
        _aligned_probabilities(probabilities, alignment)


def test_runner_rejects_wrong_profile_and_empty_plan_before_fitting() -> None:
    rows = source_rows()
    profile = load_profile(PROFILE_CONFIG)
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    wrong_profile = replace(profile, profile_id="other")
    with pytest.raises(ValueError, match="supports xetra"):
        run_walk_forward_candidate(
            rows,
            plan=plan,
            profile=wrong_profile,
            candidate=candidate(),
            adapter_factory=lambda: object(),  # type: ignore[arg-type,return-value]
        )
    with pytest.raises(ValueError, match="at least one complete fold"):
        run_walk_forward_candidate(
            rows,
            plan=WalkForwardPlan((), None, "a" * 64),
            profile=profile,
            candidate=candidate(),
            adapter_factory=lambda: object(),  # type: ignore[arg-type,return-value]
        )
