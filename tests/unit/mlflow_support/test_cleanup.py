from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import market_regime_engine.mlflow_support.cleanup as cleanup_module
from market_regime_engine.mlflow_support.cleanup import (
    EvaluationCleanupManifest,
    collect_evaluation_manifest,
    execute_evaluation_cleanup,
    execute_retired_registered_model_cleanup,
    require_production_tracking_uri,
    write_json,
)


class EvaluationClient:
    def __init__(self) -> None:
        self.experiment = SimpleNamespace(experiment_id="7")
        self.runs = {"run-b", "run-a"}
        self.models = {"model-b", "model-a"}

    def get_experiment_by_name(self, name: str) -> object:
        assert name == "macro-regime-evaluation"
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
        "macro-regime-evaluation",
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


def test_cleanup_requires_pinned_production_tracking_uri() -> None:
    assert require_production_tracking_uri("http://10.10.1.3:5000") == "http://10.10.1.3:5000"
    with pytest.raises(ValueError, match="pinned production URI"):
        require_production_tracking_uri("file:///tmp/mlruns")


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
        self.operations: list[tuple[str, str]] = []

    def get_model_version_by_alias(self, name: str, alias: str) -> object:
        assert name == "regime-xetra"
        return self.versions[self.aliases[alias]]

    def get_registered_model(self, name: str) -> object:
        assert name == "regime-xetra"
        return SimpleNamespace(aliases=dict(self.aliases))

    def search_model_versions(self, filter_string: str) -> list[object]:
        assert filter_string == "name='regime-xetra'"
        return list(self.versions.values())

    def get_model_version(self, name: str, version: str) -> object:
        assert name == "regime-xetra"
        if version not in self.versions:
            raise KeyError(version)
        return self.versions[version]

    def delete_model_version(self, name: str, version: str) -> None:
        assert name == "regime-xetra"
        if version in self.aliases.values():
            raise RuntimeError("cannot delete a model version with an attached alias")
        self.operations.append(("delete_model_version", version))
        del self.versions[version]

    def delete_registered_model_alias(self, name: str, alias: str) -> None:
        assert name == "regime-xetra"
        self.operations.append(("delete_alias", alias))
        del self.aliases[alias]

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        assert name == "regime-xetra"
        self.operations.append(("retarget_alias", alias))
        self.aliases[alias] = version


def test_retired_cleanup_requires_v4_champion_and_retargets_alias() -> None:
    client = RegisteredClient()
    inventory: dict[str, object] = {
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
    assert client.operations == [
        ("retarget_alias", "challenger"),
        ("delete_model_version", "1"),
    ]
    retry = execute_retired_registered_model_cleanup(
        client,
        inventory,
        expected_tracking_uri="http://10.10.1.3:5000",
    )
    assert retry["status"] == "verified"


def test_retired_cleanup_refuses_v4_target() -> None:
    client = RegisteredClient()
    inventory: dict[str, object] = {
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


def test_retired_cleanup_rejects_incomplete_inventory_before_mutation() -> None:
    client = RegisteredClient()
    client.versions["3"] = SimpleNamespace(
        version="3", source="runs:/legacy/package-3", tags={"profile_config_version": "2"}
    )
    inventory: dict[str, object] = {
        "tracking_uri": "http://10.10.1.3:5000",
        "model_name": "regime-xetra",
        "versions": [{"version": "1"}],
        "aliases": [{"alias": "challenger", "version": "1", "retarget_version": "2"}],
    }

    with pytest.raises(ValueError, match="every current legacy model version"):
        execute_retired_registered_model_cleanup(
            client,
            inventory,
            expected_tracking_uri="http://10.10.1.3:5000",
        )

    assert client.operations == []
    assert set(client.versions) == {"1", "2", "3"}


def test_cleanup_manifest_handles_missing_experiment_and_writes_deterministic_json(
    tmp_path: Path,
) -> None:
    client = SimpleNamespace(get_experiment_by_name=lambda name: None)
    manifest = collect_evaluation_manifest(client, tracking_uri="http://10.10.1.3:5000")
    assert manifest.experiment_id is None
    assert manifest.as_dict()["backend_gc_required"] is True
    path = write_json(tmp_path / "nested" / "manifest.json", manifest.as_dict())
    assert (
        path.read_text(encoding="utf-8")
        == json.dumps(manifest.as_dict(), sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    )


def test_cleanup_provider_missing_errors_are_narrowly_ignored() -> None:
    class Missing(Exception):
        error_code = "NOT_FOUND"

    assert cleanup_module._is_missing_error(KeyError("missing"))
    assert cleanup_module._is_missing_error(Missing("provider"))
    assert cleanup_module._is_missing_error(RuntimeError("resource does not exist"))

    def fail(_identifier: str) -> None:
        raise RuntimeError("unexpected")

    with pytest.raises(RuntimeError, match="unexpected"):
        cleanup_module._delete_if_present(fail, "x")


def test_evaluation_cleanup_reports_surviving_targets() -> None:
    client = EvaluationClient()
    manifest = collect_evaluation_manifest(client, tracking_uri="http://10.10.1.3:5000")

    def retain_model(model_id: str) -> None:
        client.models.remove(model_id)
        if model_id == "model-a":
            client.models.add(model_id)

    client.delete_logged_model = retain_model
    with pytest.raises(RuntimeError, match="left targeted objects"):
        execute_evaluation_cleanup(client, manifest, expected_tracking_uri="http://10.10.1.3:5000")


def test_cleanup_identity_and_registry_absence_contracts() -> None:
    with pytest.raises(ValueError, match="experiment_id"):
        collect_evaluation_manifest(
            SimpleNamespace(
                get_experiment_by_name=lambda name: SimpleNamespace(experiment_id=" 7")
            ),
            tracking_uri="http://10.10.1.3:5000",
        )
    assert cleanup_module._version_is_v4(SimpleNamespace(tags=None)) is False

    class MissingRegistered(RegisteredClient):
        def get_registered_model(self, name: str) -> object:
            raise KeyError(name)

    client = MissingRegistered()
    assert cleanup_module._registry_legacy_survivors(client) == (("1",), ())

    class MissingAliasTarget(RegisteredClient):
        def __init__(self) -> None:
            super().__init__()
            self.aliases = {"orphan": "9"}

        def get_model_version(self, name: str, version: str) -> object:
            if version == "9":
                raise KeyError(version)
            return super().get_model_version(name, version)

    assert cleanup_module._registry_legacy_survivors(MissingAliasTarget())[1] == (
        {"alias": "orphan", "version": "9", "reason": "missing target"},
    )


def test_retired_cleanup_rejects_champion_and_inventory_contracts() -> None:
    client = RegisteredClient()
    client.aliases["champion"] = "1"
    base = {"tracking_uri": "http://10.10.1.3:5000", "model_name": "regime-xetra"}
    with pytest.raises(ValueError, match="v4 production champion"):
        execute_retired_registered_model_cleanup(
            client,
            {**base, "versions": [], "aliases": []},
            expected_tracking_uri=base["tracking_uri"],
        )
    client.aliases["champion"] = "2"
    with pytest.raises(ValueError, match="invalid model version"):
        execute_retired_registered_model_cleanup(
            client,
            {**base, "versions": ["1"], "aliases": []},
            expected_tracking_uri=base["tracking_uri"],
        )
    with pytest.raises(ValueError, match="unsupported alias"):
        execute_retired_registered_model_cleanup(
            client,
            {
                **base,
                "versions": [{"version": "1"}],
                "aliases": [
                    {"alias": "challenger", "version": "1"},
                    {"alias": "unknown", "version": "1"},
                ],
            },
            expected_tracking_uri=base["tracking_uri"],
        )


def test_retired_cleanup_rejects_changed_or_non_v4_alias_targets() -> None:
    client = RegisteredClient()
    base = {"tracking_uri": "http://10.10.1.3:5000", "model_name": "regime-xetra"}
    with pytest.raises(ValueError, match="changed since inventory"):
        execute_retired_registered_model_cleanup(
            client,
            {
                **base,
                "versions": [{"version": "1"}],
                "aliases": [
                    {"alias": "challenger", "version": "1"},
                    {"alias": "challenger", "version": "2"},
                ],
            },
            expected_tracking_uri=base["tracking_uri"],
        )
    with pytest.raises(ValueError, match="retargeted to v4"):
        client.aliases["challenger"] = "2"
        execute_retired_registered_model_cleanup(
            client,
            {
                **base,
                "versions": [{"version": "1"}],
                "aliases": [{"alias": "challenger", "version": "2", "retarget_version": "1"}],
            },
            expected_tracking_uri=base["tracking_uri"],
        )


def test_retired_cleanup_ignores_provider_missing_mutations() -> None:
    client = RegisteredClient()
    client.aliases.pop("challenger")
    original_delete = client.delete_model_version

    def missing_delete(name: str, version: str) -> None:
        if version == "1":
            client.versions.pop(version)
            raise KeyError(version)
        original_delete(name, version)

    def missing_alias(name: str, alias: str) -> None:
        client.aliases.pop(alias)
        raise KeyError(alias)

    client.delete_model_version = missing_delete
    client.delete_registered_model_alias = missing_alias
    proof = execute_retired_registered_model_cleanup(
        client,
        {
            "tracking_uri": "http://10.10.1.3:5000",
            "model_name": "regime-xetra",
            "versions": [{"version": "1"}],
            "aliases": [{"alias": "challenger", "version": "1"}],
        },
        expected_tracking_uri="http://10.10.1.3:5000",
    )
    assert proof["status"] == "verified"


def test_retired_cleanup_rejects_scope_and_client_contract_drift() -> None:
    client = RegisteredClient()
    base: dict[str, object] = {
        "tracking_uri": "http://10.10.1.3:5000",
        "model_name": "regime-xetra",
        "versions": [],
        "aliases": [],
    }
    with pytest.raises(ValueError, match="legacy cleanup URI"):
        execute_retired_registered_model_cleanup(
            client, {**base, "tracking_uri": "other"}, expected_tracking_uri="http://10.10.1.3:5000"
        )
    with pytest.raises(ValueError, match="limited to regime-xetra"):
        execute_retired_registered_model_cleanup(
            client, {**base, "model_name": "other"}, expected_tracking_uri="http://10.10.1.3:5000"
        )
    with pytest.raises(ValueError, match="versions/aliases must be lists"):
        execute_retired_registered_model_cleanup(
            client, {**base, "versions": {}}, expected_tracking_uri="http://10.10.1.3:5000"
        )
    with pytest.raises(TypeError, match="must enumerate"):
        incomplete_client = SimpleNamespace(
            get_model_version_by_alias=lambda *args: RegisteredClient().versions["2"]
        )
        execute_retired_registered_model_cleanup(
            incomplete_client,
            base,
            expected_tracking_uri="http://10.10.1.3:5000",
        )
