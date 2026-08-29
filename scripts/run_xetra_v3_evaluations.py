"""Run the three independent Xetra v3 evaluation hierarchies."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import psycopg
import yaml
from mlflow.tracking import MlflowClient

from market_regime_engine.evaluation.walk_forward import run_walk_forward_candidate
from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.evaluation_statistics.writer import StatisticsWriter
from market_regime_engine.evaluations.checkpoints import EvaluationCheckpointStore
from market_regime_engine.evaluations.clocks import build_evaluation_clock
from market_regime_engine.evaluations.contracts import (
    EvaluationId,
    EvaluationLineage,
    delta1_feature_spec,
    medoid_feature_spec,
    multivariate_feature_spec,
)
from market_regime_engine.evaluations.delta1_univariate import evaluate_delta1_univariate
from market_regime_engine.evaluations.medoid_multivariate import evaluate_medoid_multivariate
from market_regime_engine.evaluations.medoid_univariate import evaluate_medoid_univariate
from market_regime_engine.feature_selection.contracts import FeatureBlock, FeatureSelectionPolicy
from market_regime_engine.feature_selection.freeze import freeze_first_train_features
from market_regime_engine.features.ports import FeatureRequest, SourceMode
from market_regime_engine.features.postgres_settings import FeaturePostgresSettings
from market_regime_engine.features.postgres_source import PostgresFeatureSource
from market_regime_engine.mlflow_support.evaluation_dedup import (
    record_completed_xetra_v3_evaluation,
    xetra_v3_evaluation_fingerprint,
)
from market_regime_engine.mlflow_support.evaluation_tracking import track_evaluation_result
from market_regime_engine.mlflow_support.tracking import FileMlflowTrackingPort
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import resolve_selected_feature_profile


def _policy(root: Path) -> FeatureSelectionPolicy:
    raw = yaml.safe_load(
        (root / "configs/feature_selection/xetra_semantic_medoid_v3.yaml").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(raw, dict):
        raise ValueError("v3 feature policy must be a mapping")
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


def _lineage(
    evaluation_id: EvaluationId,
    source_build_id: str,
    plan_hash: str,
    definition_hash: str,
    execution_hash: str,
    clock_hash: str,
) -> EvaluationLineage:
    return EvaluationLineage(
        evaluation_id, source_build_id, plan_hash, definition_hash, execution_hash, clock_hash
    )


def _common_feature_history(rows: pd.DataFrame, feature_names: tuple[str, ...]) -> pd.DataFrame:
    """Exclude history before every canonical feature has its first observation."""

    positions: list[int] = []
    for feature_name in feature_names:
        available = rows[feature_name].notna().to_numpy()
        first = next((index for index, value in enumerate(available) if value), None)
        if first is None:
            raise ValueError(f"canonical feature has no observations: {feature_name}")
        positions.append(first)
    return rows.iloc[max(positions) :].reset_index(drop=True)


def _git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def _checkpointed_runner(store: EvaluationCheckpointStore, evaluation_id: EvaluationId):
    def runner(source_rows: Any, plan: Any, profile: Any, candidate: Any, factory: Any) -> Any:
        return store.load_or_compute(
            evaluation_id=evaluation_id.value,
            feature_order=candidate.feature_order,
            candidate_id=candidate.candidate_id,
            compute=lambda: run_walk_forward_candidate(
                source_rows,
                plan=plan,
                profile=profile,
                candidate=candidate,
                adapter_factory=factory,
            ),
        )

    return runner


def main() -> None:
    root = Path(os.environ.get("REGIME_ENGINE_ROOT", Path(__file__).resolve().parents[1]))
    profile = load_profile(root / "configs/profiles/xetra_v3.yaml")
    policy = _policy(root)
    settings = FeaturePostgresSettings.from_env(os.environ)
    source = PostgresFeatureSource(
        lambda: psycopg.connect(**cast(Any, settings.connection_kwargs())),  # type: ignore[arg-type,return-value]
        policy.feature_universe,
    )
    snapshot = source.read(
        FeatureRequest(
            policy.feature_universe, start=None, end=None, mode=SourceMode.FEATURE_SELECTION
        )
    )
    rows = pd.DataFrame([row.values for row in snapshot.rows], columns=policy.feature_universe)
    rows.insert(0, "timestamp_m1", [row.timestamp for row in snapshot.rows])
    rows = _common_feature_history(rows, policy.feature_universe)
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
    medoid_spec = medoid_feature_spec(selection)
    multi_spec = multivariate_feature_spec(selection)
    delta_spec = delta1_feature_spec(
        tuple(feature for feature in policy.feature_universe if feature.endswith("_delta_1obs"))
    )
    multi_clock = build_evaluation_clock(rows, plan, multi_spec)
    medoid_clock = build_evaluation_clock(rows, plan, medoid_spec)
    delta_clock = build_evaluation_clock(rows, plan, delta_spec)
    definition_hash = selection.feature_selection_definition_hash
    execution_hash = selection.feature_selection_execution_hash
    git_commit = _git_commit(root)
    fingerprint = xetra_v3_evaluation_fingerprint(
        git_commit=git_commit, data_sha256=snapshot.lineage.data_sha256
    )
    os.environ["REGIME_EVALUATION_SCHEDULING_SEED"] = fingerprint
    checkpoints = EvaluationCheckpointStore(
        Path(
            os.environ.get(
                "REGIME_EVALUATION_CHECKPOINT_ROOT", "/volume2/docker/mlflow/evaluation-checkpoints"
            )
        ),
        fingerprint=fingerprint,
    )
    multivariate = evaluate_medoid_multivariate(
        rows,
        plan=plan,
        profile=profile,
        resolved_profile=resolved,
        feature_spec=multi_spec,
        lineage=_lineage(
            EvaluationId.MEDOID_MULTIVARIATE,
            snapshot.lineage.source_build_id,
            plan.plan_hash,
            definition_hash,
            execution_hash,
            multi_clock.clock_hash,
        ),
        runner=_checkpointed_runner(checkpoints, EvaluationId.MEDOID_MULTIVARIATE),
    )
    medoid = evaluate_medoid_univariate(
        rows,
        plan=plan,
        profile=profile,
        feature_spec=medoid_spec,
        clock=medoid_clock,
        lineage=_lineage(
            EvaluationId.MEDOID_UNIVARIATE,
            snapshot.lineage.source_build_id,
            plan.plan_hash,
            definition_hash,
            execution_hash,
            medoid_clock.clock_hash,
        ),
        multivariate=multivariate,
        runner=_checkpointed_runner(checkpoints, EvaluationId.MEDOID_UNIVARIATE),
    )
    delta = evaluate_delta1_univariate(
        rows,
        plan=plan,
        profile=profile,
        clock=delta_clock,
        lineage=_lineage(
            EvaluationId.DELTA1_UNIVARIATE,
            snapshot.lineage.source_build_id,
            plan.plan_hash,
            definition_hash,
            execution_hash,
            delta_clock.clock_hash,
        ),
        multivariate=multivariate,
        runner=_checkpointed_runner(checkpoints, EvaluationId.DELTA1_UNIVARIATE),
    )
    writer = StatisticsWriter(root)
    writer.preflight()
    port = FileMlflowTrackingPort(os.environ["MLFLOW_TRACKING_URI"])
    tracked = tuple(
        track_evaluation_result(port, writer, result=result)
        for result in (multivariate, medoid, delta)
    )
    marker_run_id = record_completed_xetra_v3_evaluation(
        MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"]),
        fingerprint=fingerprint,
        git_commit=git_commit,
        data_sha256=snapshot.lineage.data_sha256,
        parent_run_ids=tuple(item.parent_run_id for item in tracked),
    )
    print(
        json.dumps(
            {
                "source_build_id": snapshot.lineage.source_build_id,
                "parent_run_ids": [item.parent_run_id for item in tracked],
                "completion_marker_run_id": marker_run_id,
                "evaluation_fingerprint": fingerprint,
                "candidate_counts": [len(item.candidate_run_ids) for item in tracked],
                "clock_hashes": [
                    multi_clock.clock_hash,
                    medoid_clock.clock_hash,
                    delta_clock.clock_hash,
                ],
                "champions": [
                    multivariate.medoid_multivariate_statistical_champion,
                    medoid.medoid_univariate_evaluation_champion,
                    delta.delta1_univariate_evaluation_champion,
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
