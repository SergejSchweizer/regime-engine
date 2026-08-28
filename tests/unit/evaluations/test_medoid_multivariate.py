from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

import market_regime_engine.evaluations.medoid_multivariate as module
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardPlan, plan_walk_forward
from market_regime_engine.evaluations.contracts import EvaluationId, EvaluationLineage, FeatureSpec
from market_regime_engine.profiles.loader import load_profile

HASH = "a" * 64


def valid_plan(profile) -> WalkForwardPlan:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    timestamps = tuple(start + timedelta(days=index) for index in range(1323))
    return plan_walk_forward(timestamps, profile.walk_forward)


def test_multivariate_orchestrator_delegates_to_canonical_grid_and_selection(monkeypatch) -> None:
    profile = load_profile("configs/profiles/xetra_v3.yaml")
    plan = valid_plan(profile)
    resolved = SimpleNamespace(
        final_features=("feature",),
        source_build_id="build",
        feature_selection_definition_hash=HASH,
        feature_selection_execution_hash=HASH,
    )
    spec = FeatureSpec(EvaluationId.MEDOID_MULTIVARIATE, ("feature",))
    lineage = EvaluationLineage(
        EvaluationId.MEDOID_MULTIVARIATE, "build", plan.plan_hash, HASH, HASH, HASH
    )
    grid = SimpleNamespace()
    monkeypatch.setattr(module, "evaluate_candidate_grid", lambda *_, **__: grid)
    monkeypatch.setattr(
        module,
        "select_statistical_champion",
        lambda _: SimpleNamespace(champion_candidate_id="gaussian_hmm_k2_full"),
    )

    result = module.evaluate_medoid_multivariate(
        pd.DataFrame(),
        plan=plan,
        profile=profile,
        resolved_profile=resolved,
        feature_spec=spec,
        lineage=lineage,
    )
    assert result.candidate_grid is grid
    assert result.medoid_multivariate_statistical_champion == "gaussian_hmm_k2_full"
    assert result.no_champion_reason is None
    assert result.candidate_grid is grid
    assert result.medoid_multivariate_statistical_champion == "gaussian_hmm_k2_full"
    assert result.no_champion_reason is None


def test_multivariate_orchestrator_fails_closed_and_records_no_champion(monkeypatch) -> None:
    profile = load_profile("configs/profiles/xetra_v3.yaml")
    plan = valid_plan(profile)
    resolved = SimpleNamespace(
        final_features=("feature",),
        source_build_id="build",
        feature_selection_definition_hash=HASH,
        feature_selection_execution_hash=HASH,
    )
    spec = FeatureSpec(EvaluationId.MEDOID_MULTIVARIATE, ("feature",))
    lineage = EvaluationLineage(
        EvaluationId.MEDOID_MULTIVARIATE, "build", plan.plan_hash, HASH, HASH, HASH
    )
    with pytest.raises(ValueError, match="canonical Xetra v3"):
        module.evaluate_medoid_multivariate(
            pd.DataFrame(),
            plan=plan,
            profile=load_profile("configs/profiles/xetra_v2.yaml"),
            resolved_profile=resolved,
            feature_spec=spec,
            lineage=lineage,
        )
    with pytest.raises(ValueError, match="frozen resolved"):
        module.evaluate_medoid_multivariate(
            pd.DataFrame(),
            plan=plan,
            profile=profile,
            resolved_profile=resolved,
            feature_spec=FeatureSpec(EvaluationId.MEDOID_MULTIVARIATE, ("other",)),
            lineage=lineage,
        )
    monkeypatch.setattr(module, "evaluate_candidate_grid", lambda *_, **__: SimpleNamespace())
    monkeypatch.setattr(
        module,
        "select_statistical_champion",
        lambda _: (_ for _ in ()).throw(ValueError("no candidate")),
    )
    result = module.evaluate_medoid_multivariate(
        pd.DataFrame(),
        plan=plan,
        profile=profile,
        resolved_profile=resolved,
        feature_spec=spec,
        lineage=lineage,
    )
    assert result.medoid_multivariate_statistical_champion is None
    assert result.no_champion_reason == "no candidate"


def test_multivariate_contract_guards(monkeypatch) -> None:
    profile = load_profile("configs/profiles/xetra_v3.yaml")
    plan = valid_plan(profile)
    resolved = SimpleNamespace(
        final_features=("feature",),
        source_build_id="build",
        feature_selection_definition_hash=HASH,
        feature_selection_execution_hash=HASH,
    )
    spec = FeatureSpec(EvaluationId.MEDOID_MULTIVARIATE, ("feature",))
    lineage = EvaluationLineage(
        EvaluationId.MEDOID_MULTIVARIATE, "build", plan.plan_hash, HASH, HASH, HASH
    )
    with pytest.raises(ValueError, match="multivariate evaluation ID"):
        module.MedoidMultivariateEvaluation(
            FeatureSpec(EvaluationId.MEDOID_UNIVARIATE, tuple(f"f{index}" for index in range(8))),
            lineage,
            SimpleNamespace(),
            None,
            "no champion",
        )
    with pytest.raises(ValueError, match="exactly one champion"):
        module.MedoidMultivariateEvaluation(spec, lineage, SimpleNamespace(), None, None)
    with pytest.raises(ValueError, match="multivariate feature spec"):
        module.evaluate_medoid_multivariate(
            pd.DataFrame(),
            plan=plan,
            profile=profile,
            resolved_profile=resolved,
            feature_spec=FeatureSpec(
                EvaluationId.MEDOID_UNIVARIATE, tuple(f"f{index}" for index in range(8))
            ),
            lineage=lineage,
        )
    with pytest.raises(ValueError, match="frozen resolved"):
        module.evaluate_medoid_multivariate(
            pd.DataFrame(),
            plan=plan,
            profile=profile,
            resolved_profile=resolved,
            feature_spec=FeatureSpec(EvaluationId.MEDOID_MULTIVARIATE, ("other",)),
            lineage=lineage,
        )
    monkeypatch.setattr(module, "evaluate_candidate_grid", lambda *_, **__: SimpleNamespace())
    with pytest.raises(ValueError, match="lineage"):
        module.evaluate_medoid_multivariate(
            pd.DataFrame(),
            plan=plan,
            profile=profile,
            resolved_profile=resolved,
            feature_spec=spec,
            lineage=EvaluationLineage(
                EvaluationId.MEDOID_MULTIVARIATE, "other", plan.plan_hash, HASH, HASH, HASH
            ),
        )
