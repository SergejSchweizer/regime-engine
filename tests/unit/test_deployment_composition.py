from __future__ import annotations

from market_regime_engine import deployment


def test_compose_serving_dependencies_is_external_and_readiness_ready(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeFeatureSettings:
        @classmethod
        def from_env(cls, env):
            calls.append(("feature-settings", env))
            return cls()

        def connection_kwargs(self):
            return {"host": "external-host", "dbname": "postgres"}

    class FakeMlflowSettings:
        tracking_uri = "http://10.10.1.3:5000"
        registry_uri = tracking_uri

        @classmethod
        def from_environment(cls):
            calls.append(("mlflow-settings", None))
            return cls()

    monkeypatch.setattr(deployment, "FeaturePostgresSettings", FakeFeatureSettings)
    monkeypatch.setattr(deployment, "MLflowSettings", FakeMlflowSettings)
    monkeypatch.setattr(
        deployment.mlflow, "set_tracking_uri", lambda uri: calls.append(("tracking", uri))
    )
    monkeypatch.setattr(
        deployment.mlflow, "set_registry_uri", lambda uri: calls.append(("registry", uri))
    )
    monkeypatch.setattr(
        deployment.psycopg, "connect", lambda **kwargs: calls.append(("connect", kwargs))
    )
    monkeypatch.setattr(
        deployment,
        "PostgresFeatureSource",
        lambda connect: calls.append(("source", connect)) or "source",
    )
    monkeypatch.setattr(
        deployment, "MlflowClient", lambda **kwargs: calls.append(("client", kwargs)) or "client"
    )
    monkeypatch.setattr(
        deployment,
        "MlflowModelRegistry",
        lambda client: calls.append(("registry-adapter", client)) or "registry",
    )
    monkeypatch.setattr(
        deployment,
        "ProfileRegistry",
        lambda targets: calls.append(("profiles", targets)) or "profiles",
    )
    monkeypatch.setattr(
        deployment,
        "ModelResolver",
        lambda *args, **kwargs: calls.append(("resolver", (args, kwargs))) or "resolver",
    )
    monkeypatch.setattr(
        deployment,
        "mlflow_package_loader",
        lambda **kwargs: calls.append(("loader", kwargs)) or "loader",
    )
    monkeypatch.setattr(
        deployment.ReplayLimits,
        "from_env",
        classmethod(
            lambda cls, env: (
                calls.append(("limits", env))
                or deployment.ReplayLimits(max_concurrency_per_worker=1)
            )
        ),
    )
    monkeypatch.setattr(deployment, "LatestHandler", lambda *args: ("latest", args))
    monkeypatch.setattr(deployment, "ReplayHandler", lambda *args: ("replay", args))
    monkeypatch.setattr(deployment, "OOSPredictionHandler", lambda *args: ("oos", args))
    monkeypatch.setattr(deployment, "PredictionStore", lambda *args: ("store", args))

    dependencies = deployment.compose_serving_dependencies()

    assert dependencies.latest_handler[0] == "latest"
    assert dependencies.replay_handler[0] == "replay"
    assert dependencies.oos_handler[0] == "oos"
    assert dependencies.readiness().status == "healthy"
    assert dependencies.readiness().ready is True
    assert ("tracking", "http://10.10.1.3:5000") in calls
    assert ("registry", "http://10.10.1.3:5000") in calls
    assert any(
        name == "client" and value["tracking_uri"] == "http://10.10.1.3:5000"
        for name, value in calls
    )
