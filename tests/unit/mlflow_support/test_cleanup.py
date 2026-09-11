from __future__ import annotations

from types import SimpleNamespace

import pytest

from market_regime_engine.mlflow_support.cleanup import (
    EvaluationCleanupManifest,
    collect_evaluation_manifest,
    execute_evaluation_cleanup,
    execute_retired_registered_model_cleanup,
)


class EvaluationClient:
    def __init__(self) -> None:
        self.experiment = SimpleNamespace(experiment_id="7")
        self.runs = {"run-b", "run-a"}
        self.models = {"model-b", "model-a"}

    def get_experiment_by_name(self, name: str) -> object:
        assert name == "regime-engine-evaluation"
        return self.experiment

    def search_runs(self, experiment_ids: list[str]) -> list[object]:
        assert experiment_ids == ["7"]
        return [SimpleNamespace(info=SimpleNamespace(run_id=item)) for item in self.runs]

    def search_logged_models(self, experiment_ids: list[str]) -> list[object]:
        assert experiment_ids == ["7"]
        return [SimpleNamespace(model_id=item) for item in self.models]

    def delete_logged_model(self, model_id: str) -> None:
        self.models.remove(model_id)

    def delete_run(self, run_id: str) -> None:
        self.runs.remove(run_id)


def test_evaluation_cleanup_is_exact_and_verified() -> None:
    client = EvaluationClient()
    manifest = collect_evaluation_manifest(
        client,
        tracking_uri="http://10.10.1.3:5000",
    )
    assert manifest == EvaluationCleanupManifest(
        "http://10.10.1.3:5000",
        "regime-engine-evaluation",
        "7",
        ("model-a", "model-b"),
        ("run-a", "run-b"),
    )
    proof = execute_evaluation_cleanup(
        client,
        manifest,
        expected_tracking_uri="http://10.10.1.3:5000",
    )
    assert proof["status"] == "verified"
    assert client.models == set()
    assert client.runs == set()
    retry = execute_evaluation_cleanup(
        client,
        manifest,
        expected_tracking_uri="http://10.10.1.3:5000",
    )
    assert retry["status"] == "verified"


def test_evaluation_cleanup_rejects_uri_mismatch() -> None:
    client = EvaluationClient()
    manifest = collect_evaluation_manifest(client, tracking_uri="http://10.10.1.3:5000")
    with pytest.raises(ValueError, match="tracking URI"):
        execute_evaluation_cleanup(client, manifest, expected_tracking_uri="http://other")


class RegisteredClient:
    def __init__(self) -> None:
        self.versions = {
            "1": SimpleNamespace(
                version="1", source="runs:/legacy/package", tags={"profile_config_version": "1"}
            ),
            "2": SimpleNamespace(
                version="2",
                source="runs:/v4/package",
                tags={
                    "regime_engine.package_schema": "RegimeEngineProductionModel.v4",
                    "regime_engine.profile_config_version": "4",
                },
            ),
        }
        self.aliases = {"champion": "2", "challenger": "1"}

    def get_model_version_by_alias(self, name: str, alias: str) -> object:
        assert name == "regime-xetra"
        return self.versions[self.aliases[alias]]

    def get_model_version(self, name: str, version: str) -> object:
        assert name == "regime-xetra"
        if version not in self.versions:
            raise KeyError(version)
        return self.versions[version]

    def delete_model_version(self, name: str, version: str) -> None:
        assert name == "regime-xetra"
        del self.versions[version]

    def delete_registered_model_alias(self, name: str, alias: str) -> None:
        assert name == "regime-xetra"
        del self.aliases[alias]

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        assert name == "regime-xetra"
        self.aliases[alias] = version


def test_retired_cleanup_requires_v4_champion_and_retargets_alias() -> None:
    client = RegisteredClient()
    inventory = {
        "tracking_uri": "http://10.10.1.3:5000",
        "model_name": "regime-xetra",
        "versions": [{"version": "1"}],
        "aliases": [{"alias": "challenger", "version": "1", "retarget_version": "2"}],
    }
    proof = execute_retired_registered_model_cleanup(
        client,
        inventory,
        expected_tracking_uri="http://10.10.1.3:5000",
    )
    assert proof["status"] == "verified"
    assert set(client.versions) == {"2"}
    assert client.aliases["challenger"] == "2"
    retry = execute_retired_registered_model_cleanup(
        client,
        inventory,
        expected_tracking_uri="http://10.10.1.3:5000",
    )
    assert retry["status"] == "verified"


def test_retired_cleanup_refuses_v4_target() -> None:
    client = RegisteredClient()
    inventory = {
        "tracking_uri": "http://10.10.1.3:5000",
        "model_name": "regime-xetra",
        "versions": [{"version": "2"}],
        "aliases": [],
    }
    with pytest.raises(ValueError, match="refusing to delete accepted v4"):
        execute_retired_registered_model_cleanup(
            client,
            inventory,
            expected_tracking_uri="http://10.10.1.3:5000",
        )
