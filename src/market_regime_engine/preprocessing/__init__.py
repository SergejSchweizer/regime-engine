"""Train-only preprocessing primitives."""

from market_regime_engine.preprocessing.scaling import (
    StandardScalerArtifact,
    fit_standard_scaler,
)

__all__ = ["StandardScalerArtifact", "fit_standard_scaler"]
