from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import market_regime_engine.feature_discovery.prefix_search as module
from market_regime_engine.evaluations.provisional_teacher import build_inner_walk_forward_plan
from market_regime_engine.feature_discovery.contracts import ProvisionalTeacherReference
from market_regime_engine.profiles.loader import load_profile

HASH = "a" * 64
FEATURES = ("f0", "f1", "f2", "f3")


def source_rows(row_count: int = 819) -> pd.DataFrame:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    index = np.arange(row_count, dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp_m1": tuple(start + timedelta(days=int(value)) for value in index),
            "f0": np.sin(index / 11.0),
            "f1": np.cos(index / 17.0),
            "f2": np.sin(index / 23.0) + index / 10_000.0,
            "f3": np.cos(index / 31.0) + index / 20_000.0,
        }
    )


def teacher(rows: pd.DataFrame) -> ProvisionalTeacherReference:
    plan = build_inner_walk_forward_plan(tuple(rows["timestamp_m1"]))
    timestamps = tuple(rows["timestamp_m1"].iloc[756:819])
    probabilities = tuple(
        ((1.0, 0.0) if index % 2 == 0 else (0.0, 1.0)) for index in range(len(timestamps))
    )
    return ProvisionalTeacherReference(
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        timestamps=timestamps,
        filtered_probabilities=probabilities,
        dominant_states=tuple(index % 2 for index in range(len(timestamps))),
        valid_inner_fold_ids=("fold_001",),
        source_build_id="build-1",
        inner_plan_hash=plan.plan_hash,
    )


def fake_runner(frame, plan, profile, candidate, candidate_adapter_factory):
    del profile, candidate_adapter_factory
    prefix_length = len(candidate.feature_order)
    timestamps = tuple(frame["timestamp_m1"].iloc[756:819])
    if prefix_length == 2 and candidate.state_count == 2:
        probabilities = tuple(
            ((1.0, 0.0) if index % 2 == 0 else (0.0, 1.0)) for index in range(len(timestamps))
        )
    else:
        probabilities = ((0.5, 0.5),) * len(timestamps)
    fold = SimpleNamespace(
        fold_id="fold_001",
        oos_timestamps=timestamps,
        oos_filtered_probabilities=probabilities,
        oos_predictive_log_likelihood_per_observation=float(prefix_length),
        bic=float(prefix_length),
        aic=float(prefix_length),
    )
    return SimpleNamespace(
        candidate_id=candidate.candidate_id,
        state_count=candidate.state_count,
        source_build_id=candidate.source_build_id,
        feature_order=candidate.feature_order,
        feature_selection_definition_hash=candidate.feature_selection_definition_hash,
        feature_selection_execution_hash=candidate.feature_selection_execution_hash,
        evaluation_plan_hash=plan.plan_hash,
        folds=(fold,),
        valid_folds=(fold,),
    )


def test_nested_prefixes_choose_by_teacher_nmi_not_cross_dimension_likelihood() -> None:
    rows = source_rows()
    result = module.search_ranked_prefixes(
        rows,
        ranked_features=FEATURES,
        teacher=teacher(rows),
        profile=load_profile("configs/profiles/xetra_v4.yaml"),
        runner=fake_runner,
        max_workers=2,
    )

    assert tuple(item.prefix_length for item in result.evaluations) == (2, 3, 4)
    assert result.selected_prefix_length == 2
    assert result.selected_candidate_id == "gaussian_hmm_k2_full"
    assert result.evaluations[0].soft_regime_nmi == pytest.approx(1.0)
    assert result.evaluations[1].soft_regime_nmi == pytest.approx(0.0)
    assert all(
        tuple(item.candidate_id for item in evaluation.candidate_evaluations)
        == tuple(f"gaussian_hmm_k{state_count}_full" for state_count in (2, 3, 4, 5))
        for evaluation in result.evaluations
    )


def test_prefix_evaluation_sink_exposes_selected_raw_candidate_without_persistence() -> None:
    rows = source_rows()
    captured: dict[tuple[int, str], object] = {}

    result = module.search_ranked_prefixes(
        rows,
        ranked_features=FEATURES,
        teacher=teacher(rows),
        profile=load_profile("configs/profiles/xetra_v4.yaml"),
        runner=fake_runner,
        evaluation_sink=lambda prefix_length, candidate_id, evaluation: captured.__setitem__(
            (prefix_length, candidate_id), evaluation
        ),
    )

    assert set(captured) == {
        (item.prefix_length, item.candidate_id) for item in result.evaluations if item.valid
    }
    assert all(hasattr(evaluation, "valid_folds") for evaluation in captured.values())


def test_prefix_clock_preflight_happens_before_any_candidate_runner() -> None:
    rows = source_rows()
    rows["f0"] = 1.0
    calls: list[str] = []

    def runner(*args):
        calls.append(args[3].candidate_id)
        raise AssertionError("runner must not execute after failed prefix preflight")

    with pytest.raises(ValueError, match="no eligible prefix"):
        module.search_ranked_prefixes(
            rows,
            ranked_features=FEATURES,
            teacher=teacher(rows),
            profile=load_profile("configs/profiles/xetra_v4.yaml"),
            runner=runner,
        )
    assert calls == []


def test_prefix_search_rejects_duplicate_or_too_short_rankings() -> None:
    rows = source_rows()
    common = {
        "source_rows": rows,
        "teacher": teacher(rows),
        "profile": load_profile("configs/profiles/xetra_v4.yaml"),
        "runner": fake_runner,
    }
    with pytest.raises(ValueError, match="at least two unique"):
        module.search_ranked_prefixes(ranked_features=("f0",), **common)
    with pytest.raises(ValueError, match="at least two unique"):
        module.search_ranked_prefixes(ranked_features=("f0", "f0", "f1"), **common)
