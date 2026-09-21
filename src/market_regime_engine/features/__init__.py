"""Feature-source ports and adapters."""

from market_regime_engine.features.ports import (
    DynamicFeatureSource,
    FeatureRequest,
    FeatureRow,
    FeatureSnapshot,
    FeatureSource,
    SchemaWideFeatureSource,
    SourceMode,
    materialized_feature_data_hash,
)
from market_regime_engine.features.postgres_source import (
    MacroFeaturesPostgresSource,
    PostgresFeatureSource,
)

__all__ = [
    "DynamicFeatureSource",
    "FeatureRequest",
    "FeatureRow",
    "FeatureSnapshot",
    "FeatureSource",
    "MacroFeaturesPostgresSource",
    "PostgresFeatureSource",
    "SchemaWideFeatureSource",
    "SourceMode",
    "materialized_feature_data_hash",
]
