from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from market_regime_engine.evaluations.k_champion_contract import (
    KChampionSelection,
    feature_order_hash,
)
from market_regime_engine.mlflow_support.registry import MlflowModelRegistry

pytestmark = pytest.mark.integration


def _selection() -> KChampionSelection:
    feature_order = ("f0", "f1")
    return KChampionSelection(
        slot_id="k2",
        state_count=2,
        model_family="gaussian_hmm",
        candidate_identity="gaussian_hmm_k2_full",
        feature_order=feature_order,
        feature_order_hash=feature_order_hash(feature_order),
        source_snapshot_id="snapshot-process-qa",
        profile_id="xetra",
        profile_config_version=4,
        policy_id="k_champion_portfolio",
        policy_version="k_champion_portfolio.v1",
        validation_cutoff=datetime(2026, 8, 20, tzinfo=UTC),
        deployment_cutoff=datetime(2026, 8, 21, tzinfo=UTC),
        comparison_domain_id="k_slot_promotion.v1",
        promotion_score_version="k_slot_promotion.v1",
        reference_teacher_id="teacher-2",
        artifact_hash="a" * 64,
    )


def _run_crashing_child(
    tracking_uri: str,
    selection: KChampionSelection,
    *,
    operation: str,
    version: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the side effect in a separate interpreter and terminate abruptly."""

    payload = selection.canonical_json
    script = textwrap.dedent(
        """
        import json
        import os
        import sys
        from datetime import datetime

        from mlflow.tracking import MlflowClient

        from market_regime_engine.evaluations.k_champion_contract import KChampionSelection
        from market_regime_engine.mlflow_support.registry import MlflowModelRegistry

        uri, operation, payload, version = sys.argv[1:]
        raw = json.loads(payload)
        raw["feature_order"] = tuple(raw["feature_order"])
        raw["validation_cutoff"] = datetime.fromisoformat(
            raw["validation_cutoff"].replace("Z", "+00:00")
        )
        raw["deployment_cutoff"] = datetime.fromisoformat(
            raw["deployment_cutoff"].replace("Z", "+00:00")
        )
        selection = KChampionSelection(**raw)
        client = MlflowClient(tracking_uri=uri, registry_uri=uri)

        if operation == "version":
            original = client.create_model_version

            def create_model_version(**kwargs):
                original(**kwargs)
                os._exit(73)

            client.create_model_version = create_model_version
            MlflowModelRegistry(client).register_k_slot_package(
                selection,
                package_source_uri="runs:/crashed/version",
                artifact_hash=selection.artifact_hash,
            )
        elif operation == "alias":
            original = client.set_registered_model_alias

            def set_registered_model_alias(*args, **kwargs):
                original(*args, **kwargs)
                os._exit(73)

            client.set_registered_model_alias = set_registered_model_alias
            MlflowModelRegistry(client).compare_and_swap_alias(
                model_name="regime-xetra",
                alias=selection.alias,
                expected_current_version=None,
                new_version=version,
                reason="crash after alias side effect",
            )
        elif operation == "register":
            registered = MlflowModelRegistry(client).register_k_slot_package(
                selection,
                package_source_uri="runs:/concurrent/register",
                artifact_hash=selection.artifact_hash,
            )
            print(registered.exact_version)
            raise SystemExit(0)
        elif operation == "promote":
            audit = MlflowModelRegistry(client).compare_and_swap_alias_with_audit(
                model_name="regime-xetra",
                alias=selection.alias,
                expected_current_version=None,
                new_version=version,
                reason="concurrent process promotion race",
            )
            print(
                json.dumps(
                    {"changed": audit.changed, "observed": audit.observed_current_version}
                )
            )
            raise SystemExit(0)
        else:
            raise AssertionError(operation)
        raise AssertionError("child did not terminate at the injected side effect")
        """
    )
    arguments = [sys.executable, "-c", script, tracking_uri, operation, payload, version or ""]
    return subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    )


def test_process_kill_after_registry_side_effect_retries_without_duplicates(
    tmp_path: Path,
) -> None:
    """A new process reconciles both version and alias writes after a hard kill."""

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}"
    selection = _selection()

    crashed_version = _run_crashing_child(tracking_uri, selection, operation="version")
    assert crashed_version.returncode == 73, crashed_version.stderr

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    registry = MlflowModelRegistry(client)
    recovered = registry.register_k_slot_package(
        selection,
        package_source_uri="runs:/retry/version",
        artifact_hash=selection.artifact_hash,
    )
    assert recovered.exact_version == "1"
    assert len(client.search_model_versions("name='regime-xetra'")) == 1

    crashed_alias = _run_crashing_child(
        tracking_uri,
        selection,
        operation="alias",
        version=recovered.exact_version,
    )
    assert crashed_alias.returncode == 73, crashed_alias.stderr
    assert client.get_model_version_by_alias("regime-xetra", selection.alias).version == 1

    retry = registry.compare_and_swap_alias_with_audit(
        model_name="regime-xetra",
        alias=selection.alias,
        expected_current_version="1",
        new_version="1",
        reason="reconcile after process kill",
    )
    assert retry.changed is True
    assert client.get_model_version_by_alias("regime-xetra", selection.alias).version == 1
    assert len(client.search_model_versions("name='regime-xetra'")) == 1


def test_concurrent_process_clients_linearize_registration_and_alias_promotion(
    tmp_path: Path,
) -> None:
    """Independent registry clients create one version and one CAS winner."""

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}"
    selection = _selection()
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    # Initialize the MLflow schema before launching concurrent interpreters;
    # the registry adapter's file lock then covers all mutation sequences.
    client.search_registered_models()

    def register(_: int) -> subprocess.CompletedProcess[str]:
        return _run_crashing_child(tracking_uri, selection, operation="register")

    with ThreadPoolExecutor(max_workers=4) as executor:
        registrations = tuple(executor.map(register, range(4)))
    assert all(item.returncode == 0 for item in registrations), tuple(
        item.stderr for item in registrations
    )
    versions = tuple(item.stdout.strip().splitlines()[-1] for item in registrations)
    assert set(versions) == {"1"}
    assert len(client.search_model_versions("name='regime-xetra'")) == 1

    def promote(_: int) -> subprocess.CompletedProcess[str]:
        return _run_crashing_child(
            tracking_uri,
            selection,
            operation="promote",
            version="1",
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        promotions = tuple(executor.map(promote, range(4)))
    assert all(item.returncode == 0 for item in promotions), tuple(
        item.stderr for item in promotions
    )
    audits = tuple(json.loads(item.stdout.strip().splitlines()[-1]) for item in promotions)
    assert sum(bool(item["changed"]) for item in audits) == 1
    assert all(item["observed"] in {None, "1"} for item in audits)
    assert client.get_model_version_by_alias("regime-xetra", selection.alias).version == 1
    assert len(client.search_model_versions("name='regime-xetra'")) == 1
