from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from market_regime_engine.mlflow_support.metric_catalog import (
    METRIC_CATALOG_VERSION,
    metric_definition,
)
from market_regime_engine.mlflow_support.model_metrics import model_metric_points
from market_regime_engine.mlflow_support.ports import MetricPoint
from market_regime_engine.mlflow_support.tracking import (
    FileMlflowTrackingPort,
    _aggregate_metric_points,
    _candidate_metric_points,
    _project_candidate_logged_model,
)
from tests.unit.mlflow_support.test_all_plotting_functions import _evaluation, _plan

pytestmark = pytest.mark.integration


_ROLES = (
    ("provisional_k2", "gaussian_hmm_k2_full", "provisional_teacher"),
    ("prefix_winner", "gmm_hmm_k2_m2_full", "prefix_statistical_winner"),
    ("final_grid", "student_t_hmm_k2_full", "final_grid"),
)


def _point_signature(points: tuple[MetricPoint, ...]) -> tuple[tuple[str, str, int, int], ...]:
    return tuple(
        sorted((point.key, point.value.hex(), point.step, point.timestamp_ms) for point in points)
    )


def _expected_points(candidate_id: str) -> tuple[MetricPoint, ...]:
    evaluation = _evaluation(candidate_id=candidate_id, valid_fold_count=1)
    return (
        _candidate_metric_points(evaluation, _plan())
        + _aggregate_metric_points(evaluation)
        + model_metric_points(
            evaluation,
            fold_timestamps=tuple(item.test_end for item in _plan().folds),
        )
    )


def _project_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    order: tuple[tuple[str, str, str], ...],
) -> dict[str, tuple[tuple[tuple[str, str, int, int], ...], dict[str, str]]]:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = (tmp_path / "mlruns").as_uri()
    port = FileMlflowTrackingPort(tracking_uri, experiment_name="projection-test")
    plan = _plan()
    projected: dict[str, tuple[tuple[tuple[str, str, int, int], ...], dict[str, str]]] = {}
    for role, candidate_id, scope in order:
        evaluation = _evaluation(candidate_id=candidate_id, valid_fold_count=1)
        source_run_id = port.start_run(run_name=f"source-{role}")
        candidate_dir = tmp_path / "artifacts" / role
        candidate_dir.mkdir(parents=True)
        (candidate_dir / "immutable-evidence.json").write_text(
            f'{{"role":"{role}"}}\n',
            encoding="utf-8",
        )
        model_id = _project_candidate_logged_model(
            port,
            evaluation=evaluation,
            source_run_id=source_run_id,
            source_build_id=evaluation.source_build_id,
            plan=plan,
            candidate_dir=candidate_dir,
            dataset_snapshot_key="dataset:synthetic:catalog-v1",
            evaluation_run_key="evaluation:synthetic:run-1",
            scope=scope,
            model_name=f"evaluation:synthetic:run-1:{role}",
        )
        port.end_run(source_run_id)
        model = port._client.get_logged_model(model_id)
        expected = _expected_points(candidate_id)
        actual = tuple(
            MetricPoint(
                key=key,
                value=float(item.value),
                step=int(item.step),
                timestamp_ms=int(item.timestamp),
            )
            for key in sorted({point.key for point in expected})
            for item in port._client.get_metric_history(model.source_run_id, key)
        )
        projected[role] = (
            _point_signature(actual),
            {
                key: str(model.tags[key])
                for key in (
                    "regime_engine.metric_catalog_version",
                    "regime_engine.dataset_snapshot_key",
                    "regime_engine.evaluation_run_key",
                    "regime_engine.scope",
                    "regime_engine.candidate_id",
                    "regime_engine.logical_model_key",
                )
            },
        )
        assert _point_signature(actual) == _point_signature(expected)
        assert all(metric_definition(point.key) is not None for point in actual)
        assert len({(point.key, point.step) for point in actual}) == len(actual)
        assert model.status == "READY"
    return projected


def test_file_store_enumerates_complete_provisional_prefix_and_final_projections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projected = _project_in_order(tmp_path, monkeypatch, _ROLES)
    tracking_uri = (tmp_path / "mlruns").as_uri()
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("projection-test")
    assert experiment is not None
    models = client.search_logged_models([experiment.experiment_id])
    assert {model.name for model in models} == {
        f"evaluation_x3a_synthetic_x3a_run-1_x3a_{role}" for role, _, _ in _ROLES
    }
    assert {tags["regime_engine.scope"] for _, tags in projected.values()} == {
        scope for _, _, scope in _ROLES
    }
    assert all(
        tags["regime_engine.metric_catalog_version"] == str(METRIC_CATALOG_VERSION)
        and tags["regime_engine.dataset_snapshot_key"] == "dataset:synthetic:catalog-v1"
        and tags["regime_engine.evaluation_run_key"] == "evaluation:synthetic:run-1"
        for _, tags in projected.values()
    )


def test_logged_model_payload_is_independent_of_candidate_completion_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forward = _project_in_order(tmp_path / "forward", monkeypatch, _ROLES)
    reverse = _project_in_order(tmp_path / "reverse", monkeypatch, tuple(reversed(_ROLES)))
    assert forward == reverse
