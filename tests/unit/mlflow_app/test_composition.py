from __future__ import annotations

import datetime
import importlib


contracts = importlib.import_module("market_regime_engine.contracts")
mlflow_app = importlib.import_module("market_regime_engine.mlflow_app.app")
app_dependencies = importlib.import_module("market_regime_engine.mlflow_app.dependencies")
replay_limits = importlib.import_module("market_regime_engine.serving.replay_limits")

NOW = datetime.datetime(2026, 8, 24, 13, 45, tzinfo=datetime.UTC)
HEALTHY = app_dependencies.ReadinessSnapshot("healthy", True)


class FakeLatest:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.error: Exception | None = None

    def handle(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        invocation = kwargs["invocation"]
        assert isinstance(invocation, contracts.LatestInvocation)
        return {
            "schema_version": "RegimeInvocationResponse.v1",
            "request_id": kwargs["request_id"],
            "profile_id": kwargs["profile_id"],
            "operation": invocation.operation,
            "prediction_mode": "fixed_model_latest",
            "predictions": [{"timestamp": NOW}],
        }


class FakeReplay:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.error: Exception | None = None

    def handle(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        invocation = kwargs["invocation"]
        assert isinstance(invocation, contracts.ReplayInvocation)
        return {
            "schema_version": "RegimeInvocationResponse.v1",
            "request_id": kwargs["request_id"],
            "profile_id": kwargs["profile_id"],
            "operation": invocation.operation,
            "prediction_mode": "fixed_model_replay",
            "predictions": [{"timestamp": invocation.start}, {"timestamp": invocation.end}],
        }


class FakeOOS:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.error: Exception | None = None

    def handle(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {
            "profile_id": kwargs["profile_id"],
            "build_id": kwargs["build_id"],
            "prediction_mode": "walk_forward_oos",
            "start": kwargs["start"],
            "end": kwargs["end"],
        }


def dependencies(
    latest: FakeLatest | None = None,
    replay: FakeReplay | None = None,
    oos: FakeOOS | None = None,
    *,
    readiness: app_dependencies.ReadinessSnapshot | None = None,
) -> app_dependencies.ServiceDependencies:
    current_readiness = readiness or HEALTHY
    return app_dependencies.ServiceDependencies(
        latest_handler=latest or FakeLatest(),  # type: ignore[arg-type]
        replay_handler=replay or FakeReplay(),  # type: ignore[arg-type]
        oos_handler=oos or FakeOOS(),  # type: ignore[arg-type]
        readiness=lambda: current_readiness,
        request_id_factory=lambda: "request-fixed",
        request_time_factory=lambda: NOW,
    )


def test_composed_latest_and_replay_dispatch_to_injected_handlers() -> None:
    latest = FakeLatest()
    replay = FakeReplay()
    app = mlflow_app.create_app(dependencies=dependencies(latest, replay))
    client = app.test_client()

    latest_response = client.post(
        "/regime-engine/v1/profiles/xetra/invocations",
        json={
            "operation": "latest",
            "as_of": "2026-08-24T12:00:00Z",
            "model_version": "7",
        },
    )
    assert latest_response.status_code == 200
    assert latest_response.get_json()["request_id"] == "request-fixed"
    assert latest_response.get_json()["prediction_mode"] == "fixed_model_latest"
    assert len(latest.calls) == 1
    latest_invocation = latest.calls[0]["invocation"]
    assert isinstance(latest_invocation, contracts.LatestInvocation)
    assert latest_invocation.model_version == "7"
    assert latest.calls[0]["request_time_utc"] == NOW

    replay_response = client.post(
        "/regime-engine/v1/profiles/xetra/invocations",
        json={
            "operation": "replay",
            "start": "2026-08-20T00:00:00Z",
            "end": "2026-08-21T00:00:00Z",
        },
    )
    assert replay_response.status_code == 200
    assert replay_response.get_json()["prediction_mode"] == "fixed_model_replay"
    assert len(replay.calls) == 1
    assert isinstance(replay.calls[0]["invocation"], contracts.ReplayInvocation)


def test_composed_invocation_rejects_forbidden_unknown_malformed_and_non_json_input() -> None:
    app = mlflow_app.create_app(dependencies=dependencies())
    client = app.test_client()
    path = "/regime-engine/v1/profiles/xetra/invocations"

    cases = (
        ({"operation": "latest", "profile_id": "xetra"}, "profile_in_body_forbidden"),
        ({"operation": "latest", "unexpected": True}, "unknown_request_field"),
        ({"operation": "latest", "as_of": "2026-08-24T14:00:00+02:00"}, "invalid_timestamp"),
    )
    for body, error_code in cases:
        response = client.post(path, json=body)
        assert response.status_code == 400
        payload = response.get_json()
        assert payload["schema_version"] == "RegimeError.v1"
        assert payload["request_id"] == "request-fixed"
        assert payload["error_code"] == error_code

    malformed = client.post(path, data="{", content_type="application/json")
    assert malformed.status_code == 400
    assert malformed.get_json()["error_code"] == "malformed_json"

    non_json = client.post(path, data="operation=latest", content_type="text/plain")
    assert non_json.status_code == 400
    assert non_json.get_json()["error_code"] == "invalid_content_type"


def test_replay_guardrail_error_mapping_is_preserved_by_route() -> None:
    replay = FakeReplay()
    replay.error = replay_limits.ReplayGuardrailError(
        413,
        "replay_row_limit",
        "too many rows",
        False,
    )
    app = mlflow_app.create_app(dependencies=dependencies(replay=replay))
    response = app.test_client().post(
        "/regime-engine/v1/profiles/xetra/invocations",
        json={
            "operation": "replay",
            "start": "2026-01-01T00:00:00Z",
            "end": "2026-01-02T00:00:00Z",
        },
    )
    assert response.status_code == 413
    payload = response.get_json()
    assert payload["error_code"] == "replay_row_limit"
    assert payload["retryable"] is False


def test_oos_route_requires_explicit_build_and_supports_bounded_utc_slice() -> None:
    oos = FakeOOS()
    app = mlflow_app.create_app(dependencies=dependencies(oos=oos))
    client = app.test_client()
    response = client.get(
        "/regime-engine/v1/profiles/xetra/oos-builds/build-7"
        "?start=2026-01-01T00:00:00Z&end=2026-01-02T00:00:00Z"
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["build_id"] == "build-7"
    assert payload["prediction_mode"] == "walk_forward_oos"
    assert payload["start"] == "2026-01-01T00:00:00Z"
    assert payload["end"] == "2026-01-02T00:00:00Z"
    assert oos.calls[0]["profile_id"] == "xetra"
    assert oos.calls[0]["build_id"] == "build-7"

    unknown = client.get("/regime-engine/v1/profiles/xetra/oos-builds/build-7?latest=true")
    assert unknown.status_code == 400
    assert unknown.get_json()["error_code"] == "unknown_query_field"

    duplicate = client.get(
        "/regime-engine/v1/profiles/xetra/oos-builds/build-7?start=2026-01-01T00:00:00Z"
        "&start=2026-01-02T00:00:00Z"
    )
    assert duplicate.status_code == 400
    assert duplicate.get_json()["error_code"] == "duplicate_query_field"


def test_oos_missing_build_maps_to_404_without_leaking_path() -> None:
    oos = FakeOOS()
    oos.error = FileNotFoundError("/private/artifacts/xetra/build-secret")
    app = mlflow_app.create_app(dependencies=dependencies(oos=oos))
    response = app.test_client().get(
        "/regime-engine/v1/profiles/xetra/oos-builds/missing-build"
    )
    assert response.status_code == 404
    payload = response.get_json()
    assert payload["error_code"] == "oos_build_not_found"
    assert "/private" not in str(payload)


def test_health_is_readiness_only_and_standard_mlflow_route_survives() -> None:
    app = mlflow_app.create_app(
        dependencies=dependencies(readiness=app_dependencies.ReadinessSnapshot("degraded", True))
    )
    client = app.test_client()
    standard = client.get("/health")
    assert standard.status_code == 200

    custom = client.get("/regime-engine/v1/health")
    assert custom.status_code == 200
    assert custom.get_json() == {
        "schema_version": "RegimeHealth.v1",
        "status": "degraded",
        "ready": True,
    }
    assert "password" not in str(custom.get_json()).lower()
    assert "10.10.1.3" not in str(custom.get_json())


def test_uncomposed_service_remains_safe_not_ready_without_route_placeholders() -> None:
    app = mlflow_app.create_app(dependencies=None)
    client = app.test_client()
    health = client.get("/regime-engine/v1/health")
    assert health.status_code == 200
    assert health.get_json()["ready"] is False
    assert health.get_json()["status"] == "initializing"

    invocation = client.post(
        "/regime-engine/v1/profiles/xetra/invocations",
        json={"operation": "latest"},
    )
    assert invocation.status_code == 503
    assert invocation.get_json()["error_code"] == "service_not_composed"
    assert invocation.get_json()["schema_version"] == "RegimeError.v1"
