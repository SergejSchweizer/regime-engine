from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import market_regime_engine.evaluations.delta1_univariate as module
from market_regime_engine.evaluations.agreement import UnivariateAgreement
from market_regime_engine.evaluations.contracts import (
    DELTA1_FEATURES,
    EvaluationId,
    EvaluationLineage,
)

HASH = "a" * 64


def _arguments():
    clock = SimpleNamespace(
        evaluation_id=EvaluationId.DELTA1_UNIVARIATE, feature_order=DELTA1_FEATURES, clock_hash=HASH
    )
    lineage = EvaluationLineage(EvaluationId.DELTA1_UNIVARIATE, "build", HASH, HASH, HASH, HASH)
    multivariate = SimpleNamespace(
        medoid_multivariate_statistical_champion="gaussian_hmm_k2_full",
        candidate_grid=SimpleNamespace(
            evaluations=(SimpleNamespace(candidate_id="gaussian_hmm_k2_full"),)
        ),
    )
    return (
        clock,
        lineage,
        SimpleNamespace(plan_hash=HASH),
        SimpleNamespace(profile_id="xetra", profile_config_version=3),
        multivariate,
    )


def _grid(feature_name: str, winner: str | None = "gaussian_hmm_k2_full"):
    return SimpleNamespace(
        feature_name=feature_name,
        diagnostic_feature_model_winner=winner,
        candidate_grid=SimpleNamespace(
            evaluations=() if winner is None else (SimpleNamespace(candidate_id=winner),)
        ),
    )


def test_delta_orchestrator_runs_all_features_in_order(monkeypatch) -> None:
    clock, lineage, plan, profile, multivariate = _arguments()
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "evaluate_univariate_feature_grid",
        lambda _rows, **kwargs: (
            calls.append(kwargs["feature_name"]) or _grid(kwargs["feature_name"])
        ),
    )
    monkeypatch.setattr(
        module,
        "compare_univariate_to_multivariate",
        lambda name, *_: UnivariateAgreement(name, ("fold_001",), 1, 1.0, 1, 0.5, None, None, None),
    )
    result = module.evaluate_delta1_univariate(
        pd.DataFrame(),
        plan=plan,
        profile=profile,
        clock=clock,
        lineage=lineage,
        multivariate=multivariate,
    )
    assert set(calls) == set(DELTA1_FEATURES)
    assert result.delta1_univariate_evaluation_champion == min(DELTA1_FEATURES)


def test_delta_orchestrator_records_no_champion_and_rejects_bad_inputs(monkeypatch) -> None:
    clock, lineage, plan, profile, multivariate = _arguments()
    monkeypatch.setattr(
        module,
        "evaluate_univariate_feature_grid",
        lambda _rows, **kwargs: _grid(kwargs["feature_name"], None),
    )
    result = module.evaluate_delta1_univariate(
        pd.DataFrame(),
        plan=plan,
        profile=profile,
        clock=clock,
        lineage=lineage,
        multivariate=multivariate,
    )
    assert result.no_champion_reason == "no feature winner has eligible agreement evidence"
    with pytest.raises(ValueError, match="canonical Xetra"):
        module.evaluate_delta1_univariate(
            pd.DataFrame(),
            plan=plan,
            profile=SimpleNamespace(profile_id="xetra", profile_config_version=2),
            clock=clock,
            lineage=lineage,
            multivariate=multivariate,
        )
    with pytest.raises(ValueError, match="multivariate champion"):
        module.evaluate_delta1_univariate(
            pd.DataFrame(),
            plan=plan,
            profile=profile,
            clock=clock,
            lineage=lineage,
            multivariate=SimpleNamespace(medoid_multivariate_statistical_champion=None),
        )
    with pytest.raises(ValueError, match="clock"):
        module.evaluate_delta1_univariate(
            pd.DataFrame(),
            plan=plan,
            profile=profile,
            clock=SimpleNamespace(
                evaluation_id=EvaluationId.DELTA1_UNIVARIATE,
                feature_order=(),
                clock_hash=HASH,
            ),
            lineage=lineage,
            multivariate=multivariate,
        )
    with pytest.raises(ValueError, match="lineage"):
        module.evaluate_delta1_univariate(
            pd.DataFrame(),
            plan=SimpleNamespace(plan_hash="b" * 64),
            profile=profile,
            clock=clock,
            lineage=lineage,
            multivariate=multivariate,
        )
    with pytest.raises(ValueError, match="candidate grid"):
        module.evaluate_delta1_univariate(
            pd.DataFrame(),
            plan=plan,
            profile=profile,
            clock=clock,
            lineage=lineage,
            multivariate=SimpleNamespace(
                medoid_multivariate_statistical_champion="missing",
                candidate_grid=SimpleNamespace(evaluations=()),
            ),
        )
