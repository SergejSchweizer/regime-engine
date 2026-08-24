from __future__ import annotations

from datetime import UTC, datetime

import pytest

from market_regime_engine.mlflow_support.ports import MetricPoint, ResolvedModelVersion
from market_regime_engine.mlflow_support.settings import (
    MLflowSettings,
    PRODUCTION_MLFLOW_URI,
)


def test_one_port_mlflow_settings_are_exact() -> None:
    settings = MLflowSettings()
    assert settings.tracking_uri == PRODUCTION_MLFLOW_URI == "http://10.10.1.3:5000"
    assert settings.registry_uri == settings.tracking_uri
    with pytest.raises(ValueError, match="tracking URI"):
        MLflowSettings(
            tracking_uri="http://10.10.1.3:5001",
            registry_uri="http://10.10.1.3:5001",
        )


def test_alias_resolution_contract_returns_exact_immutable_version() -> None:
    resolved = ResolvedModelVersion(
        model_name="regime-xetra",
        alias="champion",
        exact_version="17",
        resolved_at_utc=datetime(2026, 8, 24, tzinfo=UTC),
    )
    assert resolved.exact_version == "17"


def test_fold_metric_points_carry_explicit_step_and_timestamp() -> None:
    point = MetricPoint(
        key="fold_oos_predictive_loglik_per_obs",
        value=-3.25,
        step=4,
        timestamp_ms=1787558400000,
    )
    assert point.step == 4
    assert point.timestamp_ms == 1787558400000
    with pytest.raises(ValueError, match="negative"):
        MetricPoint(key="metric", value=1.0, step=-1, timestamp_ms=0)
