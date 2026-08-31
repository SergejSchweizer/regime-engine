from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from mlflow.tracking import MlflowClient

import market_regime_engine.mlflow_support.evaluation_tracking as tracking
from market_regime_engine.evaluation_statistics.writer import StatisticsWriter
from market_regime_engine.evaluations.contracts import DELTA1_FEATURES, EvaluationId
from market_regime_engine.evaluations.delta1_univariate import Delta1UnivariateEvaluation
from market_regime_engine.mlflow_support.tracking import FileMlflowTrackingPort
from market_regime_engine.profiles.resolution import expected_candidate_ids

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class _Lineage:
    source_build_id: str = "hermetic-build"


def _candidate(candidate_id: str, feature_name: str) -> SimpleNamespace:
    fold = SimpleNamespace(
        fold_id="fold_001",
        valid=True,
        train_model_observation_count=10,
        train_log_likelihood=-100.0,
        oos_predictive_log_likelihood_per_observation=-2.0,
        aic=40.0,
        bic=50.0,
        multistart_success_rate=0.875,
        multistart_result=SimpleNamespace(
            winner=SimpleNamespace(seed=11, iterations=2, em_log_likelihood_history=(-100.0, -90.0))
        ),
    )
    return SimpleNamespace(candidate_id=candidate_id, feature_order=(feature_name,), folds=(fold,))


def _result() -> Delta1UnivariateEvaluation:
    grids = tuple(
        SimpleNamespace(
            feature_name=feature_name,
            candidate_grid=SimpleNamespace(
                feature_order=(feature_name,),
                evaluations=tuple(
                    _candidate(candidate_id, feature_name)
                    for candidate_id in expected_candidate_ids(3)
                ),
            ),
        )
        for feature_name in DELTA1_FEATURES
    )
    result = object.__new__(Delta1UnivariateEvaluation)
    object.__setattr__(result, "lineage", _Lineage())
    object.__setattr__(result, "feature_grids", grids)
    object.__setattr__(result, "delta1_univariate_evaluation_champion", None)
    object.__setattr__(result, "no_champion_reason", "hermetic diagnostic fixture")
    return result


def _stub_renderers(monkeypatch: pytest.MonkeyPatch) -> None:
    def performance(_evaluation, _metric_key, _label, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path, (-1.0,)

    def oos(_grid, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    def em(evaluation, _feature_name, output_dir: Path):
        root = output_dir / evaluation.candidate_id
        root.mkdir(parents=True, exist_ok=True)
        png, svg = root / "em_convergence.png", root / "em_convergence.svg"
        png.write_bytes(b"png")
        svg.write_text("svg", encoding="utf-8")
        entry = SimpleNamespace(png_path=str(png), svg_path=str(svg), source_artifact_hash="em")
        summary = SimpleNamespace(
            available=True,
            unavailable_reason=None,
            iterations=(1, 2),
            median=(-10.0, -9.0),
            as_json_dict=lambda: {"iterations": [1, 2], "median": [-10.0, -9.0]},
        )
        return entry, summary

    def comparison(_evaluations, _feature_name, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        png, svg = (
            output_dir / "em_convergence_all_models.png",
            output_dir / "em_convergence_all_models.svg",
        )
        png.write_bytes(b"png")
        svg.write_text("svg", encoding="utf-8")
        return SimpleNamespace(png_path=str(png), svg_path=str(svg)), ()

    monkeypatch.setattr(tracking, "_render_performance_history", performance)
    monkeypatch.setattr(tracking, "_render_oos_comparison", oos)
    monkeypatch.setattr(tracking, "render_em_convergence", em)
    monkeypatch.setattr(tracking, "render_em_convergence_comparison", comparison)
    monkeypatch.setattr(tracking, "_candidate_evidence", lambda *_, **__: {})


def test_delta1_model_metrics_hierarchy_is_hermetic_and_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    _stub_renderers(monkeypatch)
    tracking_uri = (tmp_path / "mlruns").as_uri()
    port = FileMlflowTrackingPort(tracking_uri, experiment_name="delta1-metrics-proof")
    writer = StatisticsWriter(tmp_path / "statistics")

    tracked = tracking.track_evaluation_result(port, writer, result=_result())

    assert len(tracked.feature_run_ids) == len(DELTA1_FEATURES) == 13
    assert len(tracked.candidate_run_ids) == 13 * 12
    client = MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs([port._experiment_id])  # type: ignore[attr-defined]
    assert len(runs) == 1 + 13 + 156
    for feature_name, run_id in tracked.feature_run_ids:
        manifest = client.list_artifacts(run_id, "model_metrics")
        assert any(item.path == "model_metrics/manifest.json" for item in manifest)
        assert feature_name in DELTA1_FEATURES
    first_candidate_run = tracked.candidate_run_ids[0][1]
    history = client.get_metric_history(
        first_candidate_run,
        "model_metrics.em_convergence.train_loglik_per_obs_median",
    )
    assert [(point.step, point.value) for point in history] == [(1, -10.0), (2, -9.0)]
    assert all(
        EvaluationId.DELTA1_UNIVARIATE.value in path.as_posix()
        for path in (tmp_path / "statistics").rglob("statistics.json")
    )
