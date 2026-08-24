from __future__ import annotations

from importlib.metadata import entry_points

from market_regime_engine.mlflow_app.app import create_app


def test_factory_preserves_standard_mlflow_health_and_adds_placeholders() -> None:
    app = create_app()
    client = app.test_client()
    standard = client.get("/health")
    assert standard.status_code == 200

    custom = client.get("/regime-engine/v1/health")
    assert custom.status_code == 200
    assert custom.get_json()["ready"] is False

    invocation = client.post("/regime-engine/v1/profiles/xetra/invocations", json={})
    assert invocation.status_code == 503
    assert invocation.get_json()["error_code"] == "service_not_composed"

    oos = client.get("/regime-engine/v1/profiles/xetra/oos-builds/build-1")
    assert oos.status_code == 503


def test_mlflow_app_entry_point_is_registered_exactly() -> None:
    matches = [entry for entry in entry_points(group="mlflow.app") if entry.name == "regime-engine"]
    assert len(matches) == 1
    assert matches[0].value == "market_regime_engine.mlflow_app.app:create_app"
