"""Execute the real Xetra v2 K2/K3/K4 lifecycle against the deployed MLflow server."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import pandas as pd
import psycopg
import yaml
from mlflow.tracking import MlflowClient

from market_regime_engine.evaluation.selection import select_statistical_champion
from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.feature_selection.contracts import FeatureBlock, FeatureSelectionPolicy
from market_regime_engine.feature_selection.freeze import freeze_first_train_features
from market_regime_engine.features.ports import FeatureRequest, SourceMode
from market_regime_engine.features.postgres_settings import FeaturePostgresSettings
from market_regime_engine.features.postgres_source import PostgresFeatureSource
from market_regime_engine.mlflow_support.model_package import save_production_package
from market_regime_engine.mlflow_support.registry import MlflowModelRegistry
from market_regime_engine.mlflow_support.tracking import (
    FileMlflowTrackingPort,
    track_walk_forward_evaluations,
)
from market_regime_engine.predictions.oos_publication import publish_walk_forward_oos
from market_regime_engine.predictions.store import PredictionStore
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import resolve_selected_feature_profile
from market_regime_engine.training.candidate_grid import evaluate_candidate_grid
from market_regime_engine.training.final_refit import final_production_refit


def _policy(root: Path) -> FeatureSelectionPolicy:
    raw = yaml.safe_load(
        (root / "configs/feature_selection/xetra_semantic_medoid_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(raw, dict):
        raise ValueError("v2 feature policy must be a mapping")
    return FeatureSelectionPolicy(
        policy_id=str(raw["policy_id"]),
        blocks=tuple(
            FeatureBlock(str(item["block_id"]), tuple(str(value) for value in item["features"]))
            for item in raw["blocks"]
        ),
        within_block_method=str(raw["within_block_method"]),
        cross_block_method=str(raw["cross_block_method"]),
        minimum_feature_coverage=float(raw["minimum_feature_coverage"]),
        minimum_nonzero_variance=float(raw["minimum_nonzero_variance"]),
        minimum_block_complete_observations=int(raw["minimum_block_complete_observations"]),
        maximum_cross_block_abs_spearman=float(raw["maximum_cross_block_abs_spearman"]),
        numeric_tie_abs_tolerance=float(raw["numeric_tie_abs_tolerance"]),
    )


def _aligned_eligible_window(rows: pd.DataFrame, policy: FeatureSelectionPolicy, size: int) -> int:
    """Choose an eligible origin whose final complete fold reaches the latest row."""

    for index in range(len(rows) - size + 1):
        if (len(rows) - index - size) % 63 != 0:
            continue
        window = rows.iloc[index : index + size]
        if all(
            window.loc[:, list(block.features)].notna().mean().max()
            >= policy.minimum_feature_coverage
            for block in policy.blocks
        ):
            return index
    raise ValueError("no aligned source window satisfies the v2 semantic-block coverage contract")


def main() -> None:
    root = Path(os.environ.get("REGIME_ENGINE_ROOT", Path(__file__).resolve().parents[1]))
    policy = _policy(root)
    profile = load_profile(root / "configs/profiles/xetra_v2.yaml")
    settings = FeaturePostgresSettings.from_env(os.environ)
    source = PostgresFeatureSource(
        lambda: psycopg.connect(**cast(Any, settings.connection_kwargs())),
        policy.feature_universe,
    )
    snapshot = source.read(
        FeatureRequest(
            policy.feature_universe, start=None, end=None, mode=SourceMode.FEATURE_SELECTION
        )
    )
    rows = pd.DataFrame([row.values for row in snapshot.rows], columns=policy.feature_universe)
    rows.insert(0, "timestamp_m1", [row.timestamp for row in snapshot.rows])
    rows = rows.iloc[
        _aligned_eligible_window(
            rows, policy, profile.walk_forward.minimum_train_source_observations
        ) :
    ].reset_index(drop=True)
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    selection = freeze_first_train_features(
        rows.iloc[: plan.folds[0].train_source_observations],
        policy,
        source_build_id=snapshot.lineage.source_build_id,
        data_sha256=snapshot.lineage.data_sha256,
        evaluation_plan_hash=plan.plan_hash,
    )
    resolved = resolve_selected_feature_profile(
        profile, policy, selection, source_build_id=snapshot.lineage.source_build_id
    )
    grid = evaluate_candidate_grid(rows, plan=plan, profile=profile, resolved_profile=resolved)
    champion = select_statistical_champion(grid)
    winner = next(
        item for item in grid.evaluations if item.candidate_id == champion.champion_candidate_id
    )
    candidate = next(
        item for item in resolved.candidates if item.candidate_id == champion.champion_candidate_id
    )
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    artifact_root = Path(os.environ["MLFLOW_ARTIFACT_ROOT"]) / "regime-engine"
    evidence = track_walk_forward_evaluations(
        FileMlflowTrackingPort(tracking_uri),
        source_lineage=snapshot.lineage,
        plan=plan,
        evaluations=grid.evaluations,
        statistical_selection_result=champion.champion_candidate_id,
        artifact_root=artifact_root / "evaluations" / snapshot.lineage.source_build_id,
    )
    production = final_production_refit(
        rows, lineage=snapshot.lineage, candidate=candidate, winning_evaluation=winner
    )
    package = save_production_package(
        production,
        artifact_root
        / "packages"
        / snapshot.lineage.source_build_id
        / evidence.parent_run_id
        / champion.champion_candidate_id,
    )
    feature_contract_hash = sha256(
        json.dumps(list(production.feature_order), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    oos = publish_walk_forward_oos(
        PredictionStore(artifact_root / "oos"),
        winner,
        snapshot.lineage,
        feature_contract_hash=feature_contract_hash,
        created_at_utc=datetime.now(UTC),
    )
    registry = MlflowModelRegistry(
        MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    )
    registry_client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    registry_client.log_artifacts(
        evidence.parent_run_id,
        str(package),
        artifact_path="production-package",
    )
    registered = registry.register_production_model(
        production,
        package,
        package_source_uri=f"runs:/{evidence.parent_run_id}/production-package",
    )
    try:
        previous = registry.resolve_alias("regime-xetra", "champion").exact_version
    except Exception:
        previous = None
    if not registry.compare_and_swap_alias(
        model_name="regime-xetra",
        alias="challenger",
        expected_current_version=None,
        new_version=registered.exact_version,
        reason="real Xetra v2 full evaluation completed",
    ):
        raise RuntimeError("challenger alias mutation failed")
    if not registry.compare_and_swap_alias(
        model_name="regime-xetra",
        alias="champion",
        expected_current_version=previous,
        new_version=registered.exact_version,
        reason="statistical champion promoted after real Xetra v2 evaluation",
    ):
        raise RuntimeError("champion alias mutation failed")
    print(
        json.dumps(
            {
                "candidate_ids": [item.candidate_id for item in grid.evaluations],
                "champion_candidate_id": champion.champion_candidate_id,
                "champion_version": registered.exact_version,
                "evaluation_run_id": evidence.parent_run_id,
                "oos_build_id": oos.build_id,
                "profile_config_version": 2,
                "source_build_id": snapshot.lineage.source_build_id,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
