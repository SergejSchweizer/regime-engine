from __future__ import annotations

import re
from dataclasses import fields
from math import isfinite
from types import UnionType
from typing import Any, get_args, get_origin, get_type_hints

import pytest

from market_regime_engine.evaluation.walk_forward import (
    WalkForwardEvaluation,
    WalkForwardFoldResult,
)
from market_regime_engine.evaluations.k_score import (
    KScoreCandidate,
    KScoreFoldEvidence,
    rank_k_candidates,
)
from market_regime_engine.mlflow_support.metric_catalog import (
    _DYNAMIC_PATTERNS,
    METRIC_CATALOG,
    metric_definition,
    validate_metric_points,
)
from market_regime_engine.mlflow_support.model_metrics import model_metric_points
from market_regime_engine.mlflow_support.ports import MetricPoint
from tests.unit.mlflow_support.test_all_plotting_functions import _evaluation

_K_SCORE_COMPONENTS = (
    "valid_fold_rate",
    "forecast_score",
    "calibration_score",
    "stability_score",
    "robustness_score",
    "worst_fold_forecast_score",
    "mean_support_score",
    "complexity_penalty",
    "total_score",
    "eligible",
)
_K_SCORE_FOLD_IDS = ("outer_001", "outer_002", "outer_003", "outer_004")
_K_SCORE_FEATURE_HASH = "a" * 64
_K_SCORE_PLAN_HASH = "b" * 64


def _k_score_candidate(
    state_count: int = 2,
    *,
    invalid_latest: bool = False,
) -> KScoreCandidate:
    folds = tuple(
        KScoreFoldEvidence(
            fold_id=fold_id,
            valid=not (invalid_latest and fold_id == "outer_004"),
            target_log_score=None if invalid_latest and fold_id == "outer_004" else 0.4,
            baseline_target_log_score=None if invalid_latest and fold_id == "outer_004" else 0.0,
            calibration_error=None if invalid_latest and fold_id == "outer_004" else 0.1,
            stability_score=None if invalid_latest and fold_id == "outer_004" else 0.8,
            support_score=None if invalid_latest and fold_id == "outer_004" else 0.9,
            invalid_reason="fit failed" if invalid_latest and fold_id == "outer_004" else None,
        )
        for fold_id in _K_SCORE_FOLD_IDS
    )
    return KScoreCandidate(
        state_count=state_count,
        feature_order_hash=_K_SCORE_FEATURE_HASH,
        source_build_id="source-1",
        evaluation_plan_hash=_K_SCORE_PLAN_HASH,
        target_metric_id="next_return_log_score",
        target_horizon="one_step",
        folds=folds,
    )


def _contains_numeric(annotation: Any) -> bool:
    if annotation in {int, float}:
        return True
    origin = get_origin(annotation)
    if origin in {UnionType, tuple, list, set, frozenset}:
        return any(_contains_numeric(item) for item in get_args(annotation))
    return any(_contains_numeric(item) for item in get_args(annotation))


def _numeric_fields(model: type[Any]) -> frozenset[str]:
    hints = get_type_hints(model)
    return frozenset(item.name for item in fields(model) if _contains_numeric(hints[item.name]))


# This is the explicit schema boundary for numeric evaluation evidence.  A
# scope/identity number is not a Model Metric, and the row-accounting fields
# are intentionally evidence-only because they describe source retention,
# not a comparable fitted-model quantity.  Each evidence-only entry carries
# its rationale so a new numeric field cannot silently disappear from QA.
_NUMERIC_FIELD_CLASSIFICATION: dict[str, tuple[str, str]] = {
    "WalkForwardEvaluation.profile_config_version": (
        "identity-only",
        "profile version is bound in LoggedModel tags",
    ),
    "WalkForwardEvaluation.state_count": (
        "model-metric",
        "selected_K and model-state diagnostics",
    ),
    "WalkForwardFoldResult.fold_index": (
        "identity-only",
        "fold index is the canonical metric step",
    ),
    "WalkForwardFoldResult.train_source_observation_count": (
        "evidence-only",
        "source-row accounting is preserved in the fold evidence artifact",
    ),
    "WalkForwardFoldResult.test_source_observation_count": (
        "evidence-only",
        "source-row accounting is preserved in the fold evidence artifact",
    ),
    "WalkForwardFoldResult.train_model_observation_count": (
        "model-metric",
        "fit_quality_train_observation_count",
    ),
    "WalkForwardFoldResult.test_model_observation_count": (
        "evidence-only",
        "retained TEST-row accounting is preserved in the fold evidence artifact",
    ),
    "WalkForwardFoldResult.skipped_train_incomplete_count": (
        "evidence-only",
        "TRAIN completeness accounting is preserved in the fold evidence artifact",
    ),
    "WalkForwardFoldResult.skipped_test_incomplete_count": (
        "evidence-only",
        "TEST completeness accounting is preserved in the fold evidence artifact",
    ),
    "WalkForwardFoldResult.train_log_likelihood": (
        "model-metric",
        "train_loglik_total and train_loglik_per_obs",
    ),
    "WalkForwardFoldResult.oos_predictive_log_likelihood": (
        "model-metric",
        "oos_predictive_loglik_total and oos_predictive_loglik_per_obs",
    ),
    "WalkForwardFoldResult.oos_predictive_log_likelihood_per_observation": (
        "model-metric",
        "oos_predictive_loglik_per_obs",
    ),
    "WalkForwardFoldResult.aic": ("model-metric", "aic and aic_per_train_obs"),
    "WalkForwardFoldResult.bic": ("model-metric", "bic and bic_per_train_obs"),
    "WalkForwardFoldResult.multistart_success_rate": (
        "model-metric",
        "multistart_success_rate",
    ),
    "WalkForwardFoldResult.train_hard_occupancy": (
        "model-metric",
        "state occupancy diagnostics",
    ),
    "WalkForwardFoldResult.train_soft_occupancy": (
        "model-metric",
        "state occupancy diagnostics",
    ),
    "WalkForwardFoldResult.oos_hard_occupancy": (
        "model-metric",
        "state occupancy diagnostics",
    ),
    "WalkForwardFoldResult.oos_soft_occupancy": (
        "model-metric",
        "state occupancy diagnostics",
    ),
    "WalkForwardFoldResult.max_state_signature_drift": (
        "model-metric",
        "state-diagnostic stability metric",
    ),
    "WalkForwardFoldResult.mean_state_duration": (
        "model-metric",
        "state-diagnostic duration metric",
    ),
    "WalkForwardFoldResult.switches_per_year": (
        "model-metric",
        "state-diagnostic switching metric",
    ),
    "WalkForwardFoldResult.oos_entropy_mean": (
        "model-metric",
        "filtered entropy metric",
    ),
    "WalkForwardFoldResult.oos_confidence_mean": (
        "model-metric",
        "filtered confidence metric",
    ),
    "WalkForwardFoldResult.oos_filtered_probabilities": (
        "model-metric",
        "oos filtered probability history",
    ),
}


def _assert_numeric_schema_is_classified(
    numeric_schema: frozenset[str],
) -> None:
    expected = frozenset(_NUMERIC_FIELD_CLASSIFICATION)
    assert numeric_schema == expected, (
        "numeric evaluation schema classification drift: "
        f"missing={sorted(numeric_schema - expected)}, "
        f"stale={sorted(expected - numeric_schema)}"
    )
    for field_name, (classification, rationale) in _NUMERIC_FIELD_CLASSIFICATION.items():
        assert classification in {"identity-only", "model-metric", "evidence-only"}
        assert rationale.strip(), f"missing rationale for {field_name}"


def _definition_match_count(key: str) -> int:
    return int(key in METRIC_CATALOG) + sum(
        pattern.fullmatch(key) is not None for pattern in _DYNAMIC_PATTERNS
    )


def test_numeric_evaluation_schema_is_classified_exactly_once() -> None:
    # The parent schema contributes its fully qualified field name; the fold
    # schema is checked separately so a field cannot be hidden by a collision.
    numeric_schema = frozenset(
        {f"WalkForwardEvaluation.{name}" for name in _numeric_fields(WalkForwardEvaluation)}
        | {f"WalkForwardFoldResult.{name}" for name in _numeric_fields(WalkForwardFoldResult)}
    )
    _assert_numeric_schema_is_classified(numeric_schema)


def test_numeric_schema_mutation_fails_closed_when_field_is_unclassified() -> None:
    with pytest.raises(AssertionError, match="new_numeric_evidence"):
        _assert_numeric_schema_is_classified(
            frozenset((*_NUMERIC_FIELD_CLASSIFICATION, "new_numeric_evidence"))
        )


def test_synthetic_metric_dossier_maps_every_numeric_metric_once() -> None:
    points = model_metric_points(_evaluation(valid_fold_count=1))
    assert points
    assert len({(point.key, point.step) for point in points}) == len(points)
    for point in points:
        assert _definition_match_count(point.key) == 1, point.key
        definition = metric_definition(point.key)
        assert definition is not None
        assert definition.model_metrics_visible
        assert definition.description.strip()
        assert definition.source_field.strip()

    # Every concrete indexed key used by the synthetic dossier must be
    # classified by one explicit definition or one, and only one, family.
    indexed_keys = tuple(sorted({point.key for point in points}))
    assert all(re.fullmatch(r"[A-Za-z0-9_]+", key) for key in indexed_keys)


def test_cross_k_projection_registers_every_component_for_each_legal_k() -> None:
    ranking = rank_k_candidates(
        (_k_score_candidate(2), _k_score_candidate(3)),
        latest_fold_id="outer_004",
    )

    points = ranking.metric_points(timestamp_ms=123, step=7)
    expected_keys = {
        f"k_score_{component}_k{state_count}"
        for state_count in (2, 3)
        for component in _K_SCORE_COMPONENTS
    }

    assert {point.key for point in points} == expected_keys
    assert len({(point.key, point.step) for point in points}) == len(points)
    assert all(isfinite(point.value) for point in points)
    assert all(_definition_match_count(point.key) == 1 for point in points)
    assert all(metric_definition(point.key) is not None for point in points)
    validate_metric_points(points)


def test_cross_k_ineligible_projection_is_finite_and_omits_total_score() -> None:
    ranking = rank_k_candidates(
        (_k_score_candidate(invalid_latest=True),),
        latest_fold_id="outer_004",
    )

    points = ranking.metric_points(timestamp_ms=123)
    keys = {point.key for point in points}

    assert "k_score_total_score_k2" not in keys
    assert "k_score_eligible_k2" in keys
    assert next(point.value for point in points if point.key == "k_score_eligible_k2") == 0.0
    assert all(isfinite(point.value) for point in points)
    validate_metric_points(points)


def test_cross_k_catalog_rejects_nonfinite_unknown_and_duplicate_points() -> None:
    with pytest.raises(ValueError, match="finite"):
        MetricPoint("k_score_forecast_score_k2", float("nan"), 0, 123)

    with pytest.raises(KeyError, match="not registered"):
        validate_metric_points((MetricPoint("k_score_forecast_score_k6", 0.5, 0, 123),))

    point = MetricPoint("k_score_eligible_k2", 1.0, 0, 123)
    with pytest.raises(ValueError, match="duplicate metric point identity"):
        validate_metric_points((point, point))
