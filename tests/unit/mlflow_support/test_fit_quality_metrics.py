from __future__ import annotations

from math import log

import pytest

from market_regime_engine.mlflow_support.fit_quality_metrics import (
    build_fit_quality_evidence,
    validate_fit_quality_comparison,
)
from market_regime_engine.mlflow_support.model_metrics import model_metric_points
from tests.unit.mlflow_support.test_all_plotting_functions import _evaluation


def test_fit_quality_emits_exact_criteria_and_population_aggregates() -> None:
    evaluation = _evaluation(valid_fold_count=1)
    evidence = build_fit_quality_evidence(evaluation)
    points = {(point.key, point.step): point.value for point in evidence.metric_points}

    train_ll = -100.0
    parameter_count = points[("fit_quality_parameter_count", 0)]
    train_count = 1260.0
    expected_aic = 2.0 * parameter_count - 2.0 * train_ll
    expected_bic = parameter_count * log(train_count) - 2.0 * train_ll
    expected_hqc = 2.0 * parameter_count * log(log(train_count)) - 2.0 * train_ll

    assert points[("fit_quality_aic", 1)] == pytest.approx(expected_aic)
    assert points[("fit_quality_bic", 1)] == pytest.approx(expected_bic)
    assert points[("fit_quality_hqc", 1)] == pytest.approx(expected_hqc)
    assert points[("fit_quality_aic_mean", 0)] == pytest.approx(expected_aic)
    assert points[("fit_quality_aic_count", 0)] == 1.0
    assert points[("fit_quality_valid_fold_rate", 0)] == 0.5
    assert len(model_metric_points(evaluation)) > len(evidence.metric_points)


def test_fit_quality_comparison_rejects_different_feature_vectors() -> None:
    first = _evaluation(feature_order=("feature_a", "feature_b"))
    second = _evaluation(feature_order=("feature_a", "feature_c"))
    with pytest.raises(ValueError, match="feature vector"):
        validate_fit_quality_comparison({"a": first, "b": second})


def test_fit_quality_empty_valid_population_keeps_only_counts() -> None:
    evaluation = _evaluation(valid_fold_count=0)
    evidence = build_fit_quality_evidence(evaluation)
    keys = {point.key for point in evidence.metric_points}
    assert "fit_quality_valid_fold_count" in keys
    assert "fit_quality_aic" not in keys
