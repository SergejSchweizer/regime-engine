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
FEATURE_UNIVERSE = (*FEATURES, "pca_pc_001")


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
            "pca_pc_001": np.sin(index / 37.0),
        }
    )


def teacher(rows: pd.DataFrame, *, state_count: int = 2) -> ProvisionalTeacherReference:
    plan = build_inner_walk_forward_plan(tuple(rows["timestamp_m1"]))
    timestamps = tuple(rows["timestamp_m1"].iloc[756:819])
    probabilities = tuple(
        tuple(1.0 if state == index % state_count else 0.0 for state in range(state_count))
        for index in range(len(timestamps))
    )
    return ProvisionalTeacherReference(
        candidate_id=f"gaussian_hmm_k{state_count}_full",
        state_count=state_count,
        timestamps=timestamps,
        filtered_probabilities=probabilities,
        dominant_states=tuple(index % state_count for index in range(len(timestamps))),
        valid_inner_fold_ids=("fold_001",),
        source_build_id="build-1",
        inner_plan_hash=plan.plan_hash,
    )


def k3_runner(frame, plan, profile, candidate, candidate_adapter_factory):
    del profile, candidate_adapter_factory
    timestamps = tuple(frame["timestamp_m1"].iloc[756:819])
    probabilities = tuple(
        tuple(1.0 if state == index % 3 else 0.0 for state in range(3))
        for index in range(len(timestamps))
    )
    fold = SimpleNamespace(
        fold_id="fold_001",
        oos_timestamps=timestamps,
        oos_filtered_probabilities=probabilities,
        oos_predictive_log_likelihood_per_observation=float(len(candidate.feature_order)),
        bic=float(len(candidate.feature_order)),
        aic=float(len(candidate.feature_order)),
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
        pca_raw_feature_order=FEATURES,
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


def test_serial_prefix_search_evaluates_every_ranked_prefix() -> None:
    rows = source_rows()
    result = module.search_ranked_prefixes(
        rows,
        ranked_features=FEATURES,
        teacher=teacher(rows),
        profile=load_profile("configs/profiles/xetra_v4.yaml"),
        runner=fake_runner,
        pca_raw_feature_order=FEATURES,
        max_workers=1,
    )

    assert tuple(item.prefix_length for item in result.evaluations) == (2, 3, 4)
    assert all(len(item.candidate_evaluations) == 4 for item in result.evaluations)


def test_non_pickleable_parallel_prefix_runner_fails_closed(monkeypatch) -> None:
    rows = source_rows()
    monkeypatch.setattr(
        module,
        "cpu_worker_count",
        lambda requested, task_count=None: min(requested or 4, task_count or requested or 4),
    )

    def synchronized_runner(frame, plan, profile, candidate, candidate_adapter_factory):
        return fake_runner(frame, plan, profile, candidate, candidate_adapter_factory)

    with pytest.raises(RuntimeError, match="pickleable CPU runner"):
        module.search_ranked_prefixes(
            rows,
            ranked_features=FEATURES,
            teacher=teacher(rows),
            profile=load_profile("configs/profiles/xetra_v4.yaml"),
            runner=synchronized_runner,
            pca_raw_feature_order=FEATURES,
            max_workers=3,
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
        pca_raw_feature_order=FEATURES,
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
            pca_raw_feature_order=FEATURES,
            max_workers=1,
        )
    assert calls == []


def test_prefix_search_rejects_duplicate_or_too_short_rankings() -> None:
    rows = source_rows()
    common = {
        "source_rows": rows,
        "teacher": teacher(rows),
        "profile": load_profile("configs/profiles/xetra_v4.yaml"),
        "runner": fake_runner,
        "pca_raw_feature_order": FEATURES,
    }
    with pytest.raises(ValueError, match="at least two unique"):
        module.search_ranked_prefixes(ranked_features=("f0",), **common)
    with pytest.raises(ValueError, match="at least two unique"):
        module.search_ranked_prefixes(ranked_features=("f0", "f0", "f1"), **common)


def test_fixed_k_invalid_sink_evidence_keeps_the_requested_k_identity() -> None:
    rows = source_rows()

    def reject_prefix_sink(prefix_length, _candidate_id, _evaluation):
        if prefix_length == 3:
            raise ValueError("fixture sink rejection")

    result = module.search_ranked_prefixes(
        rows,
        ranked_features=FEATURES,
        teacher=teacher(rows, state_count=3),
        profile=load_profile("configs/profiles/xetra_v4.yaml"),
        runner=k3_runner,
        pca_raw_feature_order=FEATURES,
        state_counts=(3,),
        evaluation_sink=reject_prefix_sink,
    )

    invalid = next(item for item in result.evaluations if item.prefix_length == 3)
    assert not invalid.valid
    assert invalid.candidate_id == "gaussian_hmm_k3_full"


def test_prefix_ranking_uses_nmi_then_support_then_shortest_prefix() -> None:
    def make(length: int, nmi: float, support: int) -> SimpleNamespace:
        return SimpleNamespace(
            prefix_length=length,
            soft_regime_nmi=nmi,
            shared_timestamp_count=support,
            valid=True,
        )

    winner = module._rank_prefixes(
        (make(4, 0.8, 10), make(3, 0.8, 12), make(2, 0.8, 12), make(1, 0.8, 1))
    )
    assert winner.prefix_length == 2
    with pytest.raises(ValueError, match="no eligible prefix"):
        module._rank_prefixes((SimpleNamespace(**{**vars(make(2, 0.0, 0)), "valid": False}),))


def test_prefix_validation_and_support_fail_closed() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        module._validate_features(("f0", "f1"), ("f0", "f0"))
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        module._utc(datetime(2026, 1, 1), "timestamp")
    invalid_fold = SimpleNamespace(
        oos_timestamps=(datetime(2026, 1, 1, tzinfo=UTC),),
        oos_filtered_probabilities=(),
    )
    with pytest.raises(ValueError, match="must align"):
        module._evaluation_support(SimpleNamespace(valid_folds=(invalid_fold,)))
