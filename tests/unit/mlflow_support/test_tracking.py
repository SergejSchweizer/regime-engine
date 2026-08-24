from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from market_regime_engine.contracts import SourceLineage
from market_regime_engine.evaluation.walk_forward import (
    WalkForwardEvaluation,
    WalkForwardFoldResult,
    run_walk_forward_candidate,
)
from market_regime_engine.evaluation.walk_forward_splits import plan_walk_forward
from market_regime_engine.mlflow_support.plots import (
    candidate_covariance_scale,
    render_candidate_comparison,
    render_covariance_heatmap,
    render_fold_history,
    render_transition_heatmap,
)
from market_regime_engine.mlflow_support.ports import MetricPoint
from market_regime_engine.mlflow_support.tracking import track_walk_forward_evaluations
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import FilterResult, FitResult
from market_regime_engine.profiles.loader import load_profile
from market_regime_engine.profiles.resolution import ResolvedCandidateProfile

PROFILE_CONFIG = Path("configs/profiles/xetra_v1.yaml")


class FakeTrackingPort:
    def __init__(self) -> None:
        self.runs: list[tuple[str, str, str | None]] = []
        self.params: dict[str, dict[str, str]] = {}
        self.metrics: dict[str, list[MetricPoint]] = {}
        self.artifacts: list[tuple[str, str, str]] = []

    def start_run(self, *, run_name: str, parent_run_id: str | None = None) -> str:
        run_id = f"run-{len(self.runs) + 1}"
        self.runs.append((run_id, run_name, parent_run_id))
        return run_id

    def log_params(self, run_id: str, params: dict[str, str]) -> None:
        self.params.setdefault(run_id, {}).update(params)

    def log_metric_points(self, run_id: str, points: tuple[MetricPoint, ...]) -> None:
        self.metrics.setdefault(run_id, []).extend(points)

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str) -> None:
        self.artifacts.append((run_id, local_path, artifact_path))


def candidate() -> ResolvedCandidateProfile:
    return ResolvedCandidateProfile(
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        covariance_type="full",
        feature_order=("f0", "f1"),
        feature_dimension=2,
        source_build_id="build-1",
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        original_feature_universe=tuple(f"f{index}" for index in range(48)),
        preliminary_medoids=tuple(f"f{index}" for index in range(8)),
    )


def model_artifact() -> GaussianHMMArtifact:
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=("f0", "f1"),
        start_probabilities=(0.5, 0.5),
        transition_matrix=((0.8, 0.2), (0.2, 0.8)),
        means=((-1.0, -1.0), (1.0, 1.0)),
        full_covariances=(
            ((0.20, 0.02), (0.02, 0.20)),
            ((0.30, -0.04), (-0.04, 0.25)),
        ),
    )


class DeterministicAdapter:
    def __init__(self) -> None:
        self._artifact = model_artifact()

    def fit(self, train_rows: object, state_count: int, seed: int) -> FitResult:
        assert state_count == 2
        values = np.asarray(train_rows, dtype=np.float64)
        return FitResult(
            artifact=self._artifact,
            train_log_likelihood=-float(np.sum(values * values)) + seed * 1e-6,
            converged=True,
            iterations=5,
            seed=seed,
        )

    def extract(self) -> GaussianHMMArtifact:
        return self._artifact

    def reconstruct(self, artifact: GaussianHMMArtifact) -> None:
        self._artifact = artifact

    def causal_filter(
        self,
        rows: object,
        initial_filtered_probabilities: tuple[float, ...] | None = None,
    ) -> FilterResult:
        raise AssertionError("walk-forward runner uses backend-independent causal filtering")


def source_rows(row_count: int = 1323) -> pd.DataFrame:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    timestamps = tuple(start + timedelta(days=index) for index in range(row_count))
    signs = np.where(np.arange(row_count) % 2 == 0, -1.0, 1.0)
    return pd.DataFrame(
        {
            "timestamp_m1": timestamps,
            "f0": signs,
            "f1": signs + np.where(signs > 0.0, 0.05, -0.05),
        }
    )


def valid_evaluation() -> tuple[WalkForwardEvaluation, object]:
    rows = source_rows()
    profile = load_profile(PROFILE_CONFIG)
    plan = plan_walk_forward(tuple(rows["timestamp_m1"]), profile.walk_forward)
    evaluation = run_walk_forward_candidate(
        rows,
        plan=plan,
        profile=profile,
        candidate=candidate(),
        adapter_factory=DeterministicAdapter,
    )
    return evaluation, plan


def lineage() -> SourceLineage:
    rows = source_rows()
    return SourceLineage(
        source_dataset="regime_loader.regime_features_daily",
        source_build_id="build-1",
        data_sha256="c" * 64,
        schema_version=1,
        feature_version=1,
        source_table="regime_loader.regime_features_daily",
        synced_at_utc=datetime(2026, 8, 24, tzinfo=UTC),
        row_count=len(rows),
        min_timestamp=rows["timestamp_m1"].iloc[0],
        max_timestamp=rows["timestamp_m1"].iloc[-1],
    )


def test_tracking_writes_hierarchy_histories_parameters_heatmaps_and_manifest(tmp_path: Path) -> None:
    evaluation, plan = valid_evaluation()
    assert candidate_covariance_scale(evaluation) == pytest.approx(0.30)
    port = FakeTrackingPort()
    result = track_walk_forward_evaluations(
        port,
        source_lineage=lineage(),
        plan=plan,
        evaluations=(evaluation,),
        statistical_selection_result="pending_candidate_grid",
        artifact_root=tmp_path,
    )

    assert result.parent_run_id == "run-1"
    assert result.candidate_run_ids == (("gaussian_hmm_k2_full", "run-2"),)
    assert port.runs == [
        ("run-1", "evaluation-xetra-build-1", None),
        ("run-2", "gaussian_hmm_k2_full", "run-1"),
        ("run-3", "fold_001", "run-2"),
    ]
    assert port.params["run-1"]["candidate_count"] == "1"
    assert port.params["run-1"]["data_time_semantics"] == "current_vintage_observation_day"
    assert port.params["run-2"]["covariance_type"] == "full"
    assert port.params["run-2"]["multistart_seeds"] == "11,23,37,53,71,89,107,131"
    assert port.params["run-3"]["valid"] == "true"

    candidate_points = port.metrics["run-2"]
    keys = {point.key for point in candidate_points}
    assert "fold_oos_predictive_loglik_per_obs" in keys
    assert "fold_train_hard_occupancy_state_0" in keys
    assert "fold_self_transition_state_1" in keys
    assert "candidate_valid_fold_rate" in keys
    expected_timestamp = int(plan.folds[0].test_end.timestamp() * 1000)
    assert all(
        point.timestamp_ms == expected_timestamp
        for point in candidate_points
        if point.key.startswith("fold_")
    )

    timeline = pq.read_table(tmp_path / "gaussian_hmm_k2_full" / "fold_timeline.parquet")
    assert timeline.num_rows == 1
    assert timeline.column("fold_id").to_pylist() == ["fold_001"]
    metrics = pq.read_table(tmp_path / "gaussian_hmm_k2_full" / "fold_metrics.parquet")
    assert metrics.column("valid").to_pylist() == [True]

    parameter_path = tmp_path / "gaussian_hmm_k2_full" / "fold_001" / "aligned_parameters.json"
    parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
    assert parameters["persistent_state_ids"] == ["state_0", "state_1"]
    assert parameters["feature_order"] == ["f0", "f1"]
    assert parameters["full_covariances"][0][0][1] != 0.0

    manifest_path = tmp_path / "gaussian_hmm_k2_full" / "plot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plot_types = {item["plot_type"] for item in manifest}
    assert {"fold_history", "transition_heatmap", "full_covariance_heatmap"} <= plot_types
    history = next(item for item in manifest if item["plot_type"] == "fold_history")
    assert history["x_axis_field"] == "test_end"
    assert history["x_axis_label"] == "Test window end (UTC)"
    assert history["dpi"] == 180
    covariance_entries = [item for item in manifest if item["plot_type"] == "full_covariance_heatmap"]
    assert {tuple(item["scale_bounds"]) for item in covariance_entries} == {(-0.3, 0.3)}
    assert all(Path(item["png_path"]).exists() for item in manifest)
    assert all(Path(item["svg_path"]).exists() for item in manifest)

    parent_manifest = json.loads(Path(result.parent_manifest_path).read_text(encoding="utf-8"))
    assert parent_manifest[0]["plot_type"] == "candidate_comparison"
    assert parent_manifest[0]["legend_entries"] == ["gaussian_hmm_k2_full"]


def test_invalid_fold_is_kept_in_parquet_and_creates_plot_gaps_without_fold_artifacts(
    tmp_path: Path,
) -> None:
    evaluation, plan = valid_evaluation()
    invalid = WalkForwardFoldResult(
        fold_id="fold_001",
        fold_index=1,
        valid=False,
        failure_reason="retained TEST observations are below pinned minimum 42",
        train_source_observation_count=1260,
        test_source_observation_count=63,
        train_model_observation_count=1260,
        test_model_observation_count=40,
        skipped_train_incomplete_count=0,
        skipped_test_incomplete_count=23,
    )
    invalid_evaluation = replace(evaluation, folds=(invalid,))
    port = FakeTrackingPort()
    track_walk_forward_evaluations(
        port,
        source_lineage=lineage(),
        plan=plan,
        evaluations=(invalid_evaluation,),
        statistical_selection_result="no_valid_candidate_yet",
        artifact_root=tmp_path,
    )
    assert port.params["run-3"]["valid"] == "false"
    assert port.params["run-3"]["failure_reason"].startswith("retained TEST")
    assert not (tmp_path / "gaussian_hmm_k2_full" / "fold_001" / "aligned_parameters.json").exists()
    metrics = pq.read_table(tmp_path / "gaussian_hmm_k2_full" / "fold_metrics.parquet")
    assert metrics.column("valid").to_pylist() == [False]
    assert metrics.column("fold_oos_predictive_loglik_per_obs").to_pylist() == [None]
    assert {point.key for point in port.metrics["run-2"]} == {"candidate_valid_fold_rate"}


def test_plot_and_tracking_validation_fail_closed(tmp_path: Path) -> None:
    evaluation, plan = valid_evaluation()
    invalid = WalkForwardFoldResult(
        fold_id="fold_001",
        fold_index=1,
        valid=False,
        failure_reason="invalid",
        train_source_observation_count=1260,
        test_source_observation_count=63,
        train_model_observation_count=1260,
        test_model_observation_count=63,
        skipped_train_incomplete_count=0,
        skipped_test_incomplete_count=0,
    )
    with pytest.raises(ValueError, match="unsupported fold-history metric"):
        render_fold_history(evaluation, plan, "unknown_metric", tmp_path)
    with pytest.raises(ValueError, match="valid aligned fold"):
        render_transition_heatmap(evaluation, invalid, tmp_path)
    with pytest.raises(ValueError, match="valid aligned fold"):
        render_covariance_heatmap(evaluation, invalid, 0, 1.0, tmp_path)
    with pytest.raises(ValueError, match="outside candidate state range"):
        render_covariance_heatmap(evaluation, evaluation.folds[0], 3, 1.0, tmp_path)
    with pytest.raises(ValueError, match="finite and positive"):
        render_covariance_heatmap(evaluation, evaluation.folds[0], 0, 0.0, tmp_path)
    with pytest.raises(ValueError, match="at least one evaluation"):
        render_candidate_comparison((), plan, tmp_path)
    with pytest.raises(ValueError, match="non-empty trimmed"):
        track_walk_forward_evaluations(
            FakeTrackingPort(),
            source_lineage=lineage(),
            plan=plan,
            evaluations=(evaluation,),
            statistical_selection_result=" ",
            artifact_root=tmp_path,
        )
    with pytest.raises(ValueError, match="at least one candidate"):
        track_walk_forward_evaluations(
            FakeTrackingPort(),
            source_lineage=lineage(),
            plan=plan,
            evaluations=(),
            statistical_selection_result="pending",
            artifact_root=tmp_path,
        )
    wrong_source = replace(evaluation, source_build_id="other-build")
    with pytest.raises(ValueError, match="source build"):
        track_walk_forward_evaluations(
            FakeTrackingPort(),
            source_lineage=lineage(),
            plan=plan,
            evaluations=(wrong_source,),
            statistical_selection_result="pending",
            artifact_root=tmp_path,
        )
    duplicate = replace(evaluation)
    with pytest.raises(ValueError, match="unique"):
        track_walk_forward_evaluations(
            FakeTrackingPort(),
            source_lineage=lineage(),
            plan=plan,
            evaluations=(evaluation, duplicate),
            statistical_selection_result="pending",
            artifact_root=tmp_path,
        )
