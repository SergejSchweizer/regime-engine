"""Full-history v4 deployment selection from one immutable source snapshot."""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import pandas as pd  # type: ignore[import-untyped]

from market_regime_engine.evaluations.global_regime_v4 import (
    V4ConfigurationSelection,
    select_v4_configuration,
)
from market_regime_engine.feature_discovery.contracts import (
    AdaptiveEvaluationResult,
    DeploymentSelection,
    FinalSelectedConfiguration,
)
from market_regime_engine.features.ports import FeatureCatalogSnapshot
from market_regime_engine.profiles.config import ModelProfile

ConfigurationSelector = Callable[..., V4ConfigurationSelection]


def select_deployment_configuration(
    source_rows: pd.DataFrame,
    *,
    catalog: FeatureCatalogSnapshot,
    profile: ModelProfile,
    validation: AdaptiveEvaluationResult,
    selector: ConfigurationSelector = select_v4_configuration,
    max_workers: int | None = None,
) -> DeploymentSelection:
    """Select the production configuration against all rows through source maximum.

    ``validation`` is the completed outer-policy result. Discovery is deliberately
    rerun through the shared ``select_v4_configuration`` entry point so deployment
    never copies the last outer fold's configuration or implements a second policy.
    """

    if not isinstance(source_rows, pd.DataFrame):
        raise TypeError("deployment selection requires a pandas DataFrame")
    if not isinstance(catalog, FeatureCatalogSnapshot):
        raise TypeError("deployment selection requires a feature catalog")
    if profile.profile_id != "xetra" or profile.profile_config_version != 4:
        raise ValueError("deployment selection requires the canonical Xetra v4 profile")
    if not isinstance(validation, AdaptiveEvaluationResult):
        raise TypeError("deployment selection requires a completed v4 adaptive evaluation")
    if not validation.production_eligible:
        raise ValueError("deployment selection requires a production-eligible v4 evaluation")
    if validation.source_build_id != catalog.lineage.source_build_id:
        raise ValueError("deployment source build differs from validation source build")
    if validation.catalog_hash != catalog.catalog_hash:
        raise ValueError("deployment catalog differs from validation catalog")
    if catalog.materialized_max_timestamp is None:
        raise ValueError("deployment selection requires a materialized source maximum")
    deployment_cutoff = catalog.materialized_max_timestamp
    if validation.validation_evaluation_cutoff >= deployment_cutoff:
        raise ValueError("deployment selection cutoff must be after validation cutoff")
    if "timestamp_m1" not in source_rows.columns:
        raise ValueError("deployment source rows require timestamp_m1")
    timestamps = tuple(source_rows["timestamp_m1"])
    if not timestamps or timestamps[-1] != deployment_cutoff:
        raise ValueError("deployment source rows must extend exactly through source maximum")
    if any(left >= right for left, right in pairwise(timestamps)):
        raise ValueError("deployment source timestamps must be strictly increasing and unique")

    selection = selector(
        source_rows,
        catalog=catalog,
        profile=profile,
        source_build_id=validation.source_build_id,
        max_workers=max_workers,
    )
    candidate = selection.final_candidate
    if selection.source_build_id != validation.source_build_id:
        raise ValueError("deployment selection returned a different source build")
    if selection.catalog_hash != catalog.catalog_hash:
        raise ValueError("deployment selection returned a different source catalog")
    configuration = FinalSelectedConfiguration(
        feature_order=candidate.feature_order,
        candidate_id=candidate.candidate_id,
        state_count=candidate.state_count,
        model_family=candidate.model_family,
        selected_prefix_length=len(candidate.feature_order),
        feature_discovery_hash=selection.feature_discovery_hash,
        source_build_id=validation.source_build_id,
        catalog_hash=selection.catalog_hash,
        selection_definition_hash=candidate.feature_selection_definition_hash,
        selection_execution_hash=candidate.feature_selection_execution_hash,
        state_identity_scope="model_version_local",
    )
    return DeploymentSelection(
        source_build_id=validation.source_build_id,
        source_catalog_hash=selection.catalog_hash,
        validation_evaluation_cutoff=validation.validation_evaluation_cutoff,
        deployment_selection_cutoff=deployment_cutoff,
        configuration=configuration,
        discovery_hash=selection.feature_discovery_hash,
    )


__all__ = ["select_deployment_configuration"]
