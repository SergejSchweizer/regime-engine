"""Train-only preprocessing primitives."""

from market_regime_engine.preprocessing.pca import PCAArtifact, fit_pca_transformer
from market_regime_engine.preprocessing.pca_features import (
    PCAGeneratedFeatureSet,
    materialize_pca_generated_features,
)
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
from market_regime_engine.preprocessing.two_stage import (
    PCATwoStageScalerArtifact,
    fit_pca_hmm_scaler,
)

__all__ = [
    "PCA_SOURCE_UNIVERSE",
    "PCAArtifact",
    "PCAFitClock",
    "PCAFitResult",
    "PCAGeneratedFeatureSet",
    "PCATwoStageScalerArtifact",
    "StandardScalerArtifact",
    "fit_pca_hmm_scaler",
    "fit_pca_inner_train",
    "fit_pca_transformer",
    "fit_standard_scaler",
    "materialize_pca_generated_features",
]
