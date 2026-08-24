from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from uuid import uuid4

import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

pytestmark = pytest.mark.external


def _uri() -> str:
    if os.environ.get("REGIME_RUN_EXTERNAL_MLFLOW") != "1":
        pytest.skip("set REGIME_RUN_EXTERNAL_MLFLOW=1 to run the local unified MLflow smoke")
    uri = os.environ.get("REGIME_EXTERNAL_MLFLOW_URI", "http://127.0.0.1:5000")
    parsed = urllib.parse.urlsplit(uri)
    assert parsed.scheme == "http"
    assert parsed.hostname in {"127.0.0.1", "localhost", "10.10.1.3"}
    assert parsed.port == 5000
    assert parsed.path in {"", "/"}
    return uri.rstrip("/")


def _verification() -> dict[str, str]:
    completed = subprocess.run(
        ["scripts/verify_local_compose.sh"],
        check=True,
        capture_output=True,
        text=True,
    )
    evidence = dict(
        line.split("=", 1) for line in completed.stdout.splitlines() if line and "=" in line
    )
    assert evidence["compose_project"] == "regime-engine"
    assert set(evidence["services"].split(",")) == {"mlflow", "mlflow-postgres"}
    assert evidence["application_image_id"].startswith("sha256:")
    assert evidence["repository_git_sha"]
    assert evidence["mlflow_version"] == "3.15.1"
    assert evidence["custom_image_repo_digests"] == "none"
    return evidence


def _json_request(url: str, *, body: dict[str, object] | None = None) -> tuple[int, object]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        return exc.code, payload


def test_local_unified_mlflow_tracking_registry_artifact_and_custom_routes(tmp_path: Path) -> None:
    uri = _uri()
    evidence = _verification()
    status, health = _json_request(f"{uri}/regime-engine/v1/health")
    assert status == 200
    assert isinstance(health, dict)
    assert health["schema_version"] == "RegimeHealth.v1"

    client = MlflowClient(tracking_uri=uri)
    suffix = uuid4().hex
    experiment_name = f"regime-engine-pr034-{suffix}"
    model_name = f"regime-engine-pr034-{suffix}"
    experiment_id = client.create_experiment(experiment_name)
    run = client.create_run(
        experiment_id,
        tags={
            "regime_engine.smoke": "pr034",
            "regime_engine.local_image_id": evidence["application_image_id"],
        },
    )
    run_id = run.info.run_id
    artifact = tmp_path / "fold_001.json"
    artifact.write_text(
        json.dumps({"fold_id": "fold_001", "oos_pll": -1.25}, sort_keys=True),
        encoding="utf-8",
    )
    client.log_param(run_id, "candidate_id", "gaussian_hmm_k2_full")
    client.log_metric(run_id, "oos_pll", -1.25)
    client.log_artifact(run_id, str(artifact), artifact_path="fold-history")
    client.set_terminated(run_id)

    downloaded = Path(
        client.download_artifacts(run_id, "fold-history/fold_001.json", str(tmp_path / "dl"))
    )
    assert json.loads(downloaded.read_text(encoding="utf-8"))["fold_id"] == "fold_001"
    fetched = client.get_run(run_id)
    assert fetched.data.params["candidate_id"] == "gaussian_hmm_k2_full"
    assert fetched.data.metrics["oos_pll"] == pytest.approx(-1.25)

    client.create_registered_model(model_name)
    version = client.create_model_version(
        model_name,
        source=f"runs:/{run_id}/fold-history",
        run_id=run_id,
    )
    fetched_version = client.get_model_version(model_name, version.version)
    assert fetched_version.run_id == run_id

    try:
        champion = client.get_model_version_by_alias("regime-xetra", "champion")
    except MlflowException:
        champion = None
    if champion is not None:
        latest_status, latest = _json_request(
            f"{uri}/regime-engine/v1/profiles/xetra/invocations",
            body={"operation": "latest"},
        )
        assert latest_status == 200
        assert isinstance(latest, dict)
        assert latest.get("profile_id") == "xetra"

    client.delete_registered_model(model_name)
    client.delete_experiment(experiment_id)
