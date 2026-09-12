"""Immutable local statistics dossiers for evaluation MLflow runs."""

from market_regime_engine.evaluation_statistics.contracts import (
    GLOBAL_V4_EVALUATION_ID,
    GLOBAL_V4_SCHEMA_VERSION,
    GlobalV4Evidence,
    RunStatistics,
    RunType,
    Status,
)
from market_regime_engine.evaluation_statistics.metric_extraction import (
    extract_metric_points,
    require_catalogued_numeric_evidence,
)
from market_regime_engine.evaluation_statistics.writer import StatisticsWriter

__all__ = [
    "GLOBAL_V4_EVALUATION_ID",
    "GLOBAL_V4_SCHEMA_VERSION",
    "GlobalV4Evidence",
    "RunStatistics",
    "RunType",
    "StatisticsWriter",
    "Status",
    "extract_metric_points",
    "require_catalogued_numeric_evidence",
]
