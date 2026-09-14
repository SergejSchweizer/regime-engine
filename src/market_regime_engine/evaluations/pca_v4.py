"""Route PCA-generated feature snapshots through the existing v4 selector."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluations.global_regime_v4 import (
    V4ConfigurationSelection,
    evaluate_global_regime_v4,
    select_v4_configuration,
)
from market_regime_engine.feature_discovery.contracts import AdaptiveEvaluationResult
from market_regime_engine.preprocessing.pca_features import PCAGeneratedFeatureSet
from market_regime_engine.profiles.config import ModelProfile

PCASelector = Callable[..., V4ConfigurationSelection]
PCAEvaluator = Callable[..., AdaptiveEvaluationResult]


def _as_v4_frame(generated: PCAGeneratedFeatureSet) -> pd.DataFrame:
    columns = generated.catalog.feature_names
    return pd.DataFrame(
        {
            "timestamp_m1": tuple(row.timestamp for row in generated.snapshot.rows),
            **{
                name: tuple(row.values[index] for row in generated.snapshot.rows)
                for index, name in enumerate(columns)
            },
        }
    )


def select_v4_configuration_with_pca(
    generated: PCAGeneratedFeatureSet,
    *,
    profile: ModelProfile,
    selector: PCASelector = select_v4_configuration,
    max_workers: int | None = None,
) -> V4ConfigurationSelection:
    """Run the canonical v4 policy over raw plus generated PCA features.

    This adapter owns no statistical selection logic.  It materializes the
    immutable generated snapshot into the evaluator's existing frame port and
    forwards the caller's process-worker budget unchanged.
    """

    if not isinstance(generated, PCAGeneratedFeatureSet):
        raise TypeError("PCA v4 selection requires a generated feature set")
    return selector(
        _as_v4_frame(generated),
        catalog=generated.catalog,
        profile=profile,
        source_build_id=generated.catalog.lineage.source_build_id,
        max_workers=max_workers,
        pca_raw_feature_order=generated.raw_feature_names,
        pca_variance_threshold=profile.pca.variance_threshold,
    )


def evaluate_global_regime_v4_with_pca(
    generated: PCAGeneratedFeatureSet,
    *,
    profile: ModelProfile,
    evaluator: PCAEvaluator = evaluate_global_regime_v4,
    max_workers: int | None = None,
) -> AdaptiveEvaluationResult:
    """Run the complete global policy on one immutable generated snapshot."""

    if not isinstance(generated, PCAGeneratedFeatureSet):
        raise TypeError("PCA v4 evaluation requires a generated feature set")
    return evaluator(
        _as_v4_frame(generated),
        catalog=generated.catalog,
        profile=profile,
        source_build_id=generated.catalog.lineage.source_build_id,
        max_workers=max_workers,
        pca_raw_feature_order=generated.raw_feature_names,
        pca_variance_threshold=profile.pca.variance_threshold,
    )


__all__ = [
    "evaluate_global_regime_v4_with_pca",
    "select_v4_configuration_with_pca",
]
