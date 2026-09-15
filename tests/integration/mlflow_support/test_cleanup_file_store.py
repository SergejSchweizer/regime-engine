from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from mlflow.tracking import MlflowClient

from market_regime_engine.mlflow_support.cleanup import (
    collect_evaluation_manifest,
    execute_evaluation_cleanup,
)
from market_regime_engine.mlflow_support.tracking import FileMlflowTrackingPort

pytestmark = pytest.mark.integration


class FailOnceAfterRunDelete:
    """Fail after one deterministic FileStore deletion, then delegate normally."""

    def __init__(self, client: MlflowClient, failing_run_id: str) -> None:
        self._client = client
        self._failing_run_id = failing_run_id
        self.failed = False

    def delete_logged_model(self, model_id: str) -> None:
        self._client.delete_logged_model(model_id)

    def delete_run(self, run_id: str) -> None:
        self._client.delete_run(run_id)
        if run_id == self._failing_run_id and not self.failed:
            self.failed = True
            raise RuntimeError("deterministic FileStore deletion failure")

    def get_experiment_by_name(self, name: str) -> Any:
        return self._client.get_experiment_by_name(name)

    def search_runs(self, experiment_ids: list[str]) -> list[Any]:
        return self._client.search_runs(experiment_ids)

    def search_logged_models(self, experiment_ids: list[str]) -> list[Any]:
        return self._client.search_logged_models(experiment_ids)


def _populate_file_store(tmp_path: Path) -> tuple[MlflowClient, MlflowClient, str]:
    tracking_uri = (tmp_path / "mlruns").as_uri()
    target_port = FileMlflowTrackingPort(
        tracking_uri,
        experiment_name="macro-regime-evaluation",
    )
    parent_run = target_port.start_run(run_name="legacy-parent")
    nested_run = target_port.start_run(run_name="legacy-nested", parent_run_id=parent_run)
    model_id = target_port.create_logged_model(
        name="legacy-parent-nested-candidate",
        source_run_id=nested_run,
        model_type="legacy-candidate",
        tags={"regime_engine.logical_model_key": "legacy-parent-nested-candidate"},
    )
    target_port.finalize_logged_model(model_id)
    target_port.end_run(nested_run)
    target_port.end_run(parent_run)

    unrelated_port = FileMlflowTrackingPort(
        tracking_uri,
        experiment_name="unrelated-project",
    )
    unrelated_run = unrelated_port.start_run(run_name="unrelated-run")
    unrelated_model = unrelated_port.create_logged_model(
        name="unrelated-candidate",
        source_run_id=unrelated_run,
        model_type="unrelated-candidate",
        tags={"project": "unrelated"},
    )
    unrelated_port.finalize_logged_model(unrelated_model)
    unrelated_port.end_run(unrelated_run)
    return (
        MlflowClient(tracking_uri=tracking_uri),
        MlflowClient(tracking_uri=tracking_uri),
        tracking_uri,
    )


def test_file_store_cleanup_is_exact_and_idempotent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    client, unrelated_client, tracking_uri = _populate_file_store(tmp_path)
    target = collect_evaluation_manifest(client, tracking_uri=tracking_uri)
    unrelated = unrelated_client.get_experiment_by_name("unrelated-project")
    assert unrelated is not None
    unrelated_runs = unrelated_client.search_runs([unrelated.experiment_id])
    unrelated_models = unrelated_client.search_logged_models([unrelated.experiment_id])
    unrelated_run_ids = {item.info.run_id for item in unrelated_runs}
    unrelated_model_ids = {item.model_id for item in unrelated_models}
    assert len(target.run_ids) == 2
    assert len(target.logged_model_ids) == 1
    assert len(unrelated_runs) == 1
    assert len(unrelated_models) == 1

    first = execute_evaluation_cleanup(
        client,
        target,
        expected_tracking_uri=tracking_uri,
    )
    assert first["status"] == "verified"
    assert collect_evaluation_manifest(client, tracking_uri=tracking_uri).run_ids == ()
    assert collect_evaluation_manifest(client, tracking_uri=tracking_uri).logged_model_ids == ()

    second = execute_evaluation_cleanup(
        client,
        target,
        expected_tracking_uri=tracking_uri,
    )
    assert second["status"] == "verified"
    assert second["surviving_run_ids"] == []
    assert second["surviving_logged_model_ids"] == []

    unrelated_after = unrelated_client.get_experiment_by_name("unrelated-project")
    assert unrelated_after is not None
    assert {
        item.info.run_id for item in unrelated_client.search_runs([unrelated_after.experiment_id])
    } == unrelated_run_ids
    assert {
        item.model_id
        for item in unrelated_client.search_logged_models([unrelated_after.experiment_id])
    } == unrelated_model_ids


def test_file_store_cleanup_restarts_from_the_same_manifest_after_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    client, unrelated_client, tracking_uri = _populate_file_store(tmp_path)
    manifest = collect_evaluation_manifest(client, tracking_uri=tracking_uri)
    unrelated = unrelated_client.get_experiment_by_name("unrelated-project")
    assert unrelated is not None
    unrelated_run_ids = {
        item.info.run_id for item in unrelated_client.search_runs([unrelated.experiment_id])
    }
    unrelated_model_ids = {
        item.model_id for item in unrelated_client.search_logged_models([unrelated.experiment_id])
    }
    failing = FailOnceAfterRunDelete(client, manifest.run_ids[0])

    with pytest.raises(RuntimeError, match="deterministic FileStore deletion failure"):
        execute_evaluation_cleanup(
            failing,
            manifest,
            expected_tracking_uri=tracking_uri,
        )
    assert failing.failed
    assert (
        manifest.run_ids[0]
        not in collect_evaluation_manifest(
            client,
            tracking_uri=tracking_uri,
        ).run_ids
    )

    resumed = execute_evaluation_cleanup(
        client,
        manifest,
        expected_tracking_uri=tracking_uri,
    )
    assert resumed["status"] == "verified"
    assert resumed["run_ids"] == list(manifest.run_ids)
    assert resumed["logged_model_ids"] == list(manifest.logged_model_ids)

    unrelated_after = unrelated_client.get_experiment_by_name("unrelated-project")
    assert unrelated_after is not None
    assert {
        item.info.run_id for item in unrelated_client.search_runs([unrelated_after.experiment_id])
    } == unrelated_run_ids
    assert {
        item.model_id
        for item in unrelated_client.search_logged_models([unrelated_after.experiment_id])
    } == unrelated_model_ids
