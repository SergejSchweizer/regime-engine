from __future__ import annotations

from types import MappingProxyType

import pytest

from market_regime_engine.evaluation_statistics.metric_extraction import (
    extract_metric_points,
    require_catalogued_numeric_evidence,
)
from market_regime_engine.mlflow_support.metric_catalog import METRIC_CATALOG


def test_catalog_is_immutable() -> None:
    assert isinstance(METRIC_CATALOG, MappingProxyType)
    assert all(definition.source_field for definition in METRIC_CATALOG.values())
    with pytest.raises(TypeError):
        METRIC_CATALOG["injected"] = object()  # type: ignore[index]


def test_extractor_preserves_scalar_and_history_without_rounding() -> None:
    points = extract_metric_points(
        {
            "valid_fold_count": 2,
            "oos_filtered_probability_state_0": (0.125, 0.375),
            "status": "FINISHED",
        },
        timestamp_ms=123,
    )
    assert [(point.key, point.step, point.value, point.timestamp_ms) for point in points] == [
        ("oos_filtered_probability_state_0", 0, 0.125, 123),
        ("oos_filtered_probability_state_0", 1, 0.375, 123),
        ("valid_fold_count", 0, 2.0, 123),
    ]


def test_unclassified_numeric_evidence_fails_closed() -> None:
    with pytest.raises(ValueError, match="not catalogued"):
        require_catalogued_numeric_evidence({"new_numeric_field": 1.0})
    with pytest.raises(KeyError, match="not registered"):
        extract_metric_points({"new_numeric_field": 1.0}, timestamp_ms=0)


def test_scalar_history_and_malformed_history_fail_closed() -> None:
    with pytest.raises(ValueError, match="scalar metric"):
        extract_metric_points({"valid_fold_count": (1, 2)}, timestamp_ms=0)
    with pytest.raises(ValueError, match="only numbers"):
        extract_metric_points({"oos_filtered_probability_state_0": (0.1, "bad")}, timestamp_ms=0)
