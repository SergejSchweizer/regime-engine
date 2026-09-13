"""Train-only preprocessing primitives."""

from market_regime_engine.preprocessing.pca import PCAArtifact, fit_pca_transformer
from market_regime_engine.preprocessing.pca_policy import (
    PCA_SOURCE_UNIVERSE,
    PCAFitClock,
    PCAFitResult,
    fit_pca_inner_train,
)
from market_regime_engine.preprocessing.scaling import (
    StandardScalerArtifact,
    fit_standard_scaler,
)

__all__ = [
    "PCA_SOURCE_UNIVERSE",
    "PCAArtifact",
    "PCAFitClock",
    "PCAFitResult",
    "StandardScalerArtifact",
    "fit_pca_inner_train",
    "fit_pca_transformer",
    "fit_standard_scaler",
]
