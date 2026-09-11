from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from uuid import uuid4

import pytest
from mlflow import MlflowClient

pytestmark = pytest.mark.external


def _uri() -> str:
    if os.environ.get("REGIME_RUN_EXTERNAL_MLFLOW") != "1":
        pytest.skip("set REGIME_RUN_EXTERNAL_MLFLOW=1 to run the local unified MLflow smoke")
    uri = os.environ.get("REGIME_EXTERNAL_MLFLOW_URI", "http://10.10.1.3:5000")
    parsed = urllib.parse.urlsplit(uri)
    assert parsed.scheme == "http"
    assert parsed.hostname in {"127.0.0.1", "localhost", "10.10.1.3"}
    assert parsed.port == 5000
    assert parsed.path in {"", "/"}
    return uri.rstrip("/")


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


def test_external_mlflow_tracking_registry_and_artifacts(
    tmp_path: Path,
) -> None:
    uri = _uri()
    status, health = _json_request(f"{uri}/health")
    assert status == 200
    assert health == "OK"

    client = MlflowClient(tracking_uri=uri)
    suffix = uuid4().hex
    experiment_name = f"regime-engine-pr034-{suffix}"
    model_name = f"regime-engine-pr034-{suffix}"
    experiment_id = client.create_experiment(experiment_name)
    run = client.create_run(
        experiment_id,
        tags={
            "regime_engine.smoke": "pr034",
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

    client.delete_registered_model(model_name)
    client.delete_experiment(experiment_id)
