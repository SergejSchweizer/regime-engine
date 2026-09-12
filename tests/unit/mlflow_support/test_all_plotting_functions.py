from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

import market_regime_engine.mlflow_support.model_metrics as model_metrics
import market_regime_engine.mlflow_support.tracking as tracking
from market_regime_engine.evaluation.walk_forward import (
    WalkForwardEvaluation,
    WalkForwardFoldResult,
)
from market_regime_engine.evaluation.walk_forward_splits import WalkForwardFold, WalkForwardPlan
from market_regime_engine.mlflow_support.plots import (
    candidate_covariance_scale,
    fold_history_metric_keys,
    render_candidate_comparison,
    render_candidate_oos_gap_heatmap,
    render_candidate_oos_summary,
    render_covariance_heatmap,
    render_em_convergence,
    render_em_convergence_comparison,
    render_fold_history,
    render_oos_state_timeline,
    render_state_feature_influence,
    render_state_occupancy_table,
    render_state_transition_history,
    render_transition_heatmap,
    summarize_em_convergence,
)
from market_regime_engine.mlflow_support.ports import MetricPoint
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import FitResult
from market_regime_engine.preprocessing.scaling import StandardScalerArtifact
from market_regime_engine.states.alignment import align_first_fold
from market_regime_engine.training.multistart import (
    MULTISTART_SEEDS,
    MultistartResult,
    StartDiagnostic,
)

START = datetime(2024, 1, 1, tzinfo=UTC)
HASH = "a" * 64


def _artifact(feature_order: tuple[str, ...]) -> GaussianHMMArtifact:
    dimension = len(feature_order)
    covariance = tuple(
        tuple(1.0 if row == column else 0.1 for column in range(dimension))
        for row in range(dimension)
    )
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=feature_order,
        start_probabilities=(0.6, 0.4),
        transition_matrix=((0.9, 0.1), (0.2, 0.8)),
        means=(tuple(0.0 for _ in feature_order), tuple(1.0 for _ in feature_order)),
        full_covariances=(covariance, covariance),
    )


def _multistart(artifact: GaussianHMMArtifact) -> MultistartResult:
    results = tuple(
        FitResult(
            artifact=artifact,
            train_log_likelihood=-100.0 - index,
            converged=True,
            iterations=3,
            seed=seed,
            em_log_likelihood_history=(-120.0 - index, -110.0 - index, -100.0 - index),
        )
        for index, seed in enumerate(MULTISTART_SEEDS)
    )
    diagnostics = tuple(
        StartDiagnostic(
            seed=result.seed,
            success=True,
            converged=True,
            iterations=result.iterations,
            train_log_likelihood=result.train_log_likelihood,
            artifact=result.artifact,
            failure_reason=None,
        )
        for result in results
    )
    return MultistartResult(state_count=2, winner=results[0], diagnostics=diagnostics)


def _plan() -> WalkForwardPlan:
    folds = tuple(
        WalkForwardFold(
            fold_index=index,
            fold_id=f"fold_{index:03d}",
            train_start=START + timedelta(days=(index - 1) * 10),
            train_end=START + timedelta(days=(index - 1) * 10 + 1),
            test_start=START + timedelta(days=(index - 1) * 10 + 2),
            test_end=START + timedelta(days=(index - 1) * 10 + 4),
            train_source_observations=1260 + (index - 1) * 63,
            test_source_observations=63,
        )
        for index in (1, 2)
    )
    return WalkForwardPlan(folds=folds, evaluation_cutoff=folds[-1].test_end, plan_hash=HASH)


def _evaluation(
    *,
    feature_order: tuple[str, ...] = ("feature_a", "feature_b"),
    candidate_id: str = "gaussian_hmm_k2_full",
    valid_fold_count: int = 1,
) -> WalkForwardEvaluation:
    plan = _plan()
    artifact = _artifact(feature_order)
    alignment = align_first_fold(artifact)
    scaler = StandardScalerArtifact(
        feature_order=feature_order,
        means=tuple(0.0 for _ in feature_order),
        variances=tuple(1.0 for _ in feature_order),
        scales=tuple(1.0 for _ in feature_order),
    )
    folds: list[WalkForwardFoldResult] = []
    for index, planned in enumerate(plan.folds, start=1):
        if index <= valid_fold_count:
            timestamps = tuple(planned.test_start + timedelta(hours=hour) for hour in range(3))
            probabilities = ((0.8, 0.2), (0.3, 0.7), (0.6, 0.4))
            folds.append(
                WalkForwardFoldResult(
                    fold_id=planned.fold_id,
                    fold_index=index,
                    valid=True,
                    failure_reason=None,
                    train_source_observation_count=planned.train_source_observations,
                    test_source_observation_count=planned.test_source_observations,
                    train_model_observation_count=planned.train_source_observations,
                    test_model_observation_count=3,
                    skipped_train_incomplete_count=0,
                    skipped_test_incomplete_count=60,
                    scaler_artifact=scaler,
                    multistart_result=_multistart(artifact),
                    model_artifact=artifact,
                    alignment=alignment,
                    train_log_likelihood=-100.0,
                    oos_predictive_log_likelihood=-3.0,
                    oos_predictive_log_likelihood_per_observation=-1.0,
                    aic=100.0,
                    bic=110.0,
                    multistart_success_rate=1.0,
                    train_hard_occupancy=(0.6, 0.4),
                    train_soft_occupancy=(0.55, 0.45),
                    oos_hard_occupancy=(2 / 3, 1 / 3),
                    oos_soft_occupancy=(0.57, 0.43),
                    max_state_signature_drift=0.0,
                    mean_state_duration=4.0,
                    switches_per_year=12.0,
                    oos_entropy_mean=0.5,
                    oos_confidence_mean=0.7,
                    oos_timestamps=timestamps,
                    oos_filtered_probabilities=probabilities,
                )
            )
        else:
            folds.append(
                WalkForwardFoldResult(
                    fold_id=planned.fold_id,
                    fold_index=index,
                    valid=False,
                    failure_reason="synthetic invalid fold",
                    train_source_observation_count=planned.train_source_observations,
                    test_source_observation_count=planned.test_source_observations,
                    train_model_observation_count=0,
                    test_model_observation_count=0,
                    skipped_train_incomplete_count=planned.train_source_observations,
                    skipped_test_incomplete_count=planned.test_source_observations,
                )
            )
    return WalkForwardEvaluation(
        profile_id="xetra",
        profile_config_version=4,
        candidate_id=candidate_id,
        state_count=2,
        source_build_id="synthetic-build",
        feature_order=feature_order,
        feature_selection_definition_hash=HASH,
        feature_selection_execution_hash=HASH,
        evaluation_plan_hash=HASH,
        evaluation_cutoff=plan.evaluation_cutoff,
        folds=tuple(folds),
        alignment_reference_scaler=scaler,
    )


def test_all_candidate_and_fold_plot_renderers(tmp_path: Path) -> None:
    evaluation = _evaluation()
    plan = _plan()

    assert set(fold_history_metric_keys()) == {
        "fold_train_loglik",
        "fold_oos_predictive_loglik",
        "fold_oos_predictive_loglik_per_obs",
        "fold_aic_per_train_obs",
        "fold_bic_per_train_obs",
        "fold_multistart_success_rate",
        "fold_min_train_hard_occupancy",
        "fold_min_train_soft_occupancy",
        "fold_max_state_signature_drift",
        "fold_mean_state_duration",
        "fold_switches_per_year",
        "fold_oos_entropy_mean",
        "fold_oos_confidence_mean",
    }
    for metric_key in fold_history_metric_keys():
        assert render_fold_history(evaluation, plan, metric_key, tmp_path).png_path.endswith(".png")

    fold = evaluation.folds[0]
    scale = candidate_covariance_scale(evaluation)
    render_transition_heatmap(evaluation, fold, tmp_path)
    render_state_occupancy_table(evaluation, plan, tmp_path)
    render_oos_state_timeline(evaluation, plan, tmp_path)
    render_state_transition_history(evaluation, plan, tmp_path)
    render_state_feature_influence(evaluation, tmp_path)
    for state_index in range(evaluation.state_count):
        render_covariance_heatmap(evaluation, fold, state_index, scale, tmp_path)

    second = _evaluation(candidate_id="gmm_hmm_k2_m2_full")
    evaluations = (evaluation, second)
    render_candidate_comparison(evaluations, plan, tmp_path, evaluation.candidate_id)
    render_candidate_oos_gap_heatmap(evaluations, plan, tmp_path, evaluation.candidate_id)
    render_candidate_oos_summary(evaluations, plan, tmp_path, evaluation.candidate_id)

    pngs = tuple(tmp_path.rglob("*.png"))
    assert len(pngs) >= 18
    assert all(path.stat().st_size > 0 for path in pngs)


def test_em_plot_renderers_cover_available_and_unavailable_traces(tmp_path: Path) -> None:
    available = _evaluation(feature_order=("feature_a",), valid_fold_count=2)
    unavailable = _evaluation(
        feature_order=("feature_a",),
        candidate_id="gmm_hmm_k2_m2_full",
        valid_fold_count=0,
    )
    summary = summarize_em_convergence(available)
    assert summary.available
    entry, rendered_summary = render_em_convergence(available, "feature_a", tmp_path)
    assert entry.available_candidate_ids == (available.candidate_id,)
    assert rendered_summary.iterations == (1, 2, 3)
    comparison, summaries = render_em_convergence_comparison(
        (available, unavailable), "feature_a", tmp_path
    )
    assert comparison.unavailable_candidate_ids == (unavailable.candidate_id,)
    assert len(summaries) == 2
    assert Path(entry.png_path).is_file()
    assert Path(entry.svg_path).is_file()
    assert Path(comparison.png_path).is_file()
    assert Path(comparison.svg_path).is_file()


def test_plot_renderers_reject_invalid_inputs(tmp_path: Path) -> None:
    evaluation = _evaluation()
    plan = _plan()
    with pytest.raises(ValueError, match="unsupported fold-history metric"):
        render_fold_history(evaluation, plan, "not-a-metric", tmp_path)


def test_model_metric_projection_covers_all_valid_fold_diagnostics() -> None:
    evaluation = _evaluation()
    points = model_metrics.model_metric_points(evaluation)
    keys = {point.key for point in points}

    assert {"train_loglik_total", "hqc", "aic", "bic"} <= keys
    assert "oos_filtered_probability_state_0" in keys
    assert "viterbi_state" in keys
    assert "transition_probability_state_0_to_state_0" in keys
    assert "covariance_min_eigenvalue" in keys

    gmm = _evaluation(candidate_id="gmm_hmm_k2_m2_full")
    student_t = _evaluation(candidate_id="student_t_hmm_k2_full")
    assert model_metrics.model_metric_points(gmm)
    assert model_metrics.model_metric_points(student_t)
    assert model_metrics.model_metric_points(
        evaluation,
        fold_timestamps=(datetime(2024, 1, 1, tzinfo=UTC),) * len(evaluation.folds),
    )

    with pytest.raises(ValueError, match="fold_timestamps"):
        model_metrics.model_metric_points(evaluation, fold_timestamps=(START,))
    with pytest.raises(ValueError, match="non-negative"):
        model_metrics.model_metric_points(
            evaluation,
            fold_timestamps=(datetime(1960, 1, 1, tzinfo=UTC),) * len(evaluation.folds),
        )


def test_walk_forward_tracking_projection_is_deterministic(tmp_path: Path) -> None:
    evaluation = _evaluation()
    plan = _plan()

    candidate_points = tracking._candidate_metric_points(evaluation, plan)
    aggregate_points = tracking._aggregate_metric_points(evaluation)
    assert candidate_points
    assert aggregate_points
    assert (
        tracking._candidate_model_tags(
            evaluation,
            source_build_id="build-1",
            plan=plan,
            dataset_snapshot_key="dataset-1",
            evaluation_run_key="evaluation-1",
        )["regime_engine.scope"]
        == "candidate"
    )
    assert len(tracking._timeline_rows(evaluation, plan)) == len(plan.folds)
    assert len(tracking._metric_rows(evaluation, plan)) == len(plan.folds)
    assert tracking._aligned_parameter_payload(evaluation, evaluation.folds[0])["candidate_id"] == (
        evaluation.candidate_id
    )

    class Port:
        def __init__(self) -> None:
            self.logged: list[tuple[str, tuple[object, ...]]] = []
            self.finalized: list[tuple[str, bool]] = []

        def create_logged_model(self, **kwargs: object) -> str:
            assert kwargs["name"] == "evaluation-1-gaussian_hmm_k2_full"
            return "model-1"

        def log_model_metric_points(self, model_id: str, points: tuple[object, ...]) -> None:
            self.logged.append((model_id, points))

        def log_model_artifacts(self, model_id: str, local_dir: str) -> None:
            assert model_id == "model-1"
            assert local_dir == str(tmp_path)

        def finalize_logged_model(self, model_id: str, *, failed: bool = False) -> None:
            self.finalized.append((model_id, failed))

    port = Port()
    assert (
        tracking._project_candidate_logged_model(
            port,  # type: ignore[arg-type]
            evaluation=evaluation,
            source_run_id="run-1",
            source_build_id="build-1",
            plan=plan,
            candidate_dir=tmp_path,
            dataset_snapshot_key="dataset-1",
            evaluation_run_key="evaluation-1",
        )
        == "model-1"
    )
    assert len(port.logged) == 1
    logged_points = port.logged[0][1]
    logged_keys = {cast(MetricPoint, point).key for point in logged_points}
    assert {
        "train_loglik_total",
        "oos_filtered_probability_state_0",
        "transition_probability_state_0_to_state_0",
        "covariance_min_eigenvalue",
    } <= logged_keys
    assert port.finalized == [("model-1", False)]
    with pytest.raises(ValueError, match="state index"):
        render_covariance_heatmap(evaluation, evaluation.folds[0], 2, 1.0, tmp_path)
    with pytest.raises(ValueError, match="at least one evaluation"):
        render_candidate_comparison((), plan, tmp_path)
    with pytest.raises(ValueError, match="at least one candidate"):
        render_em_convergence_comparison((), "feature_a", tmp_path)
