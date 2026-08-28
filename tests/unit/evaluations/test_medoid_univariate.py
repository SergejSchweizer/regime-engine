from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import market_regime_engine.evaluations.medoid_univariate as module
from market_regime_engine.evaluations.agreement import UnivariateAgreement
from market_regime_engine.evaluations.contracts import (
    DELTA1_FEATURES,
    EvaluationId,
    EvaluationLineage,
    FeatureSpec,
)

HASH = "a" * 64
MEDOIDS = tuple(f"medoid_{index}" for index in range(8))


def _arguments():
    feature_spec = FeatureSpec(EvaluationId.MEDOID_UNIVARIATE, MEDOIDS)
    clock = SimpleNamespace(
        evaluation_id=EvaluationId.MEDOID_UNIVARIATE,
        feature_order=MEDOIDS,
        clock_hash=HASH,
    )
    lineage = EvaluationLineage(EvaluationId.MEDOID_UNIVARIATE, "build", HASH, HASH, HASH, HASH)
    plan = SimpleNamespace(plan_hash=HASH)
    profile = SimpleNamespace(profile_id="xetra", profile_config_version=3)
    multivariate_winner = SimpleNamespace(candidate_id="gaussian_hmm_k2_full")
    multivariate = SimpleNamespace(
        medoid_multivariate_statistical_champion="gaussian_hmm_k2_full",
        candidate_grid=SimpleNamespace(evaluations=(multivariate_winner,)),
    )
    return feature_spec, clock, lineage, plan, profile, multivariate


def _grid(feature_name: str, winner: str | None = "gaussian_hmm_k2_full"):
    evaluations = () if winner is None else (SimpleNamespace(candidate_id=winner),)
    return SimpleNamespace(
        feature_name=feature_name,
        diagnostic_feature_model_winner=winner,
        candidate_grid=SimpleNamespace(evaluations=evaluations),
    )


def _agreement(feature_name: str, nmi: float | None, timestamps: int, reason: str | None = None):
    return UnivariateAgreement(
        feature_name,
        5 * ("fold_001",),
        5,
        1.0,
        timestamps,
        nmi,
        None,
        None,
        reason,
    )


def test_medoid_orchestrator_runs_all_grids_in_canonical_order(monkeypatch) -> None:
    feature_spec, clock, lineage, plan, profile, multivariate = _arguments()
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
        lambda feature_name, *_: _agreement(feature_name, 0.5, 20),
    )

    result = module.evaluate_medoid_univariate(
        pd.DataFrame(),
        plan=plan,
        profile=profile,
        feature_spec=feature_spec,
        clock=clock,
        lineage=lineage,
        multivariate=multivariate,
    )

    assert set(calls) == set(MEDOIDS)
    assert tuple(grid.feature_name for grid in result.feature_grids) == MEDOIDS
    assert result.eligible_feature_names == MEDOIDS
    assert result.medoid_univariate_evaluation_champion == MEDOIDS[0]


def test_medoid_orchestrator_uses_only_nmi_support_and_feature_name(monkeypatch) -> None:
    feature_spec, clock, lineage, plan, profile, multivariate = _arguments()
    monkeypatch.setattr(
        module,
        "evaluate_univariate_feature_grid",
        lambda _rows, **kwargs: _grid(kwargs["feature_name"]),
    )
    agreements = {
        MEDOIDS[0]: _agreement(MEDOIDS[0], 0.7, 10),
        MEDOIDS[1]: _agreement(MEDOIDS[1], 0.7 + 5e-13, 15),
        MEDOIDS[2]: _agreement(MEDOIDS[2], 0.9, 99, "zero shared OOS timestamps"),
    }
    monkeypatch.setattr(
        module,
        "compare_univariate_to_multivariate",
        lambda feature_name, *_: agreements.get(feature_name, _agreement(feature_name, 0.1, 1)),
    )

    result = module.evaluate_medoid_univariate(
        pd.DataFrame(),
        plan=plan,
        profile=profile,
        feature_spec=feature_spec,
        clock=clock,
        lineage=lineage,
        multivariate=multivariate,
    )

    assert MEDOIDS[2] not in result.eligible_feature_names
    assert result.champion_tie_feature_names == (MEDOIDS[1],)
    assert result.medoid_univariate_evaluation_champion == MEDOIDS[1]


def test_medoid_orchestrator_records_no_champion_and_rejects_missing_multivariate(
    monkeypatch,
) -> None:
    feature_spec, clock, lineage, plan, profile, multivariate = _arguments()
    monkeypatch.setattr(
        module,
        "evaluate_univariate_feature_grid",
        lambda _rows, **kwargs: _grid(kwargs["feature_name"], None),
    )
    result = module.evaluate_medoid_univariate(
        pd.DataFrame(),
        plan=plan,
        profile=profile,
        feature_spec=feature_spec,
        clock=clock,
        lineage=lineage,
        multivariate=multivariate,
    )
    assert result.medoid_univariate_evaluation_champion is None
    assert result.no_champion_reason == "no feature winner has eligible agreement evidence"

    with pytest.raises(ValueError, match="multivariate champion"):
        module.evaluate_medoid_univariate(
            pd.DataFrame(),
            plan=plan,
            profile=profile,
            feature_spec=feature_spec,
            clock=clock,
            lineage=lineage,
            multivariate=SimpleNamespace(
                medoid_multivariate_statistical_champion=None,
                candidate_grid=SimpleNamespace(evaluations=()),
            ),
        )


def test_medoid_orchestrator_rejects_incompatible_inputs() -> None:
    feature_spec, clock, lineage, plan, profile, multivariate = _arguments()
    arguments = dict(
        plan=plan,
        profile=profile,
        feature_spec=feature_spec,
        clock=clock,
        lineage=lineage,
        multivariate=multivariate,
    )
    with pytest.raises(ValueError, match="canonical Xetra"):
        module.evaluate_medoid_univariate(
            pd.DataFrame(),
            **(
                arguments
                | {"profile": SimpleNamespace(profile_id="xetra", profile_config_version=2)}
            ),
        )
    with pytest.raises(ValueError, match="feature spec"):
        module.evaluate_medoid_univariate(
            pd.DataFrame(),
            **(
                arguments
                | {"feature_spec": FeatureSpec(EvaluationId.DELTA1_UNIVARIATE, DELTA1_FEATURES)}
            ),
        )
    with pytest.raises(ValueError, match="clock"):
        module.evaluate_medoid_univariate(
            pd.DataFrame(),
            **(
                arguments
                | {
                    "clock": SimpleNamespace(
                        evaluation_id=EvaluationId.MEDOID_UNIVARIATE,
                        feature_order=(),
                        clock_hash=HASH,
                    )
                }
            ),
        )
    with pytest.raises(ValueError, match="lineage"):
        module.evaluate_medoid_univariate(
            pd.DataFrame(), **(arguments | {"plan": SimpleNamespace(plan_hash="b" * 64)})
        )
    with pytest.raises(ValueError, match="candidate grid"):
        module.evaluate_medoid_univariate(
            pd.DataFrame(),
            **(
                arguments
                | {
                    "multivariate": SimpleNamespace(
                        medoid_multivariate_statistical_champion="missing",
                        candidate_grid=SimpleNamespace(evaluations=()),
                    )
                }
            ),
        )


def test_medoid_result_invariants_and_winner_resolution_fail_closed() -> None:
    feature_spec, clock, lineage, *_ = _arguments()
    grids = tuple(_grid(feature_name, None) for feature_name in MEDOIDS)
    result = module.MedoidUnivariateEvaluation(
        feature_spec, clock, lineage, grids, (), (), (), None, "no eligible agreement"
    )
    assert result.no_champion_reason == "no eligible agreement"
    with pytest.raises(ValueError, match="canonical medoid order"):
        module.MedoidUnivariateEvaluation(
            feature_spec, clock, lineage, tuple(reversed(grids)), (), (), (), None, "reason"
        )
    with pytest.raises(ValueError, match="exactly one champion"):
        module.MedoidUnivariateEvaluation(
            feature_spec, clock, lineage, grids, (), (), (), None, None
        )
    with pytest.raises(ValueError, match="missing feature winner"):
        module._winner_evaluation(_grid("feature", None))
    with pytest.raises(ValueError, match="not present"):
        module._winner_evaluation(
            SimpleNamespace(
                diagnostic_feature_model_winner="missing",
                candidate_grid=SimpleNamespace(evaluations=()),
            )
        )
