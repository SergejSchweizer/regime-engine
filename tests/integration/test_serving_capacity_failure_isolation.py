from __future__ import annotations

from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import pytest
import yaml
from mlflow.tracking import MlflowClient
from psycopg_pool import PoolTimeout

from market_regime_engine.features.postgres_pool import ProcessLocalPostgresPool
from market_regime_engine.features.postgres_settings import FeaturePostgresSettings
from market_regime_engine.mlflow_app.app import create_app
from market_regime_engine.mlflow_app.dependencies import ReadinessSnapshot, ServiceDependencies
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.production_artifact import ProductionModelArtifact
from market_regime_engine.preprocessing.scaling import StandardScalerArtifact
from market_regime_engine.serving.model_cache import ModelCache
from market_regime_engine.serving.replay_admission import ReplayAdmission
from market_regime_engine.serving.replay_limits import ReplayGuardrailError, ReplayLimits

pytestmark = pytest.mark.integration
NOW = datetime(2026, 8, 24, 12, tzinfo=UTC)


def _artifact(build: str) -> ProductionModelArtifact:
    features = ("f0",)
    return ProductionModelArtifact(
        profile_id="xetra",
        profile_config_version=1,
        registered_model="regime-xetra",
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        source_build_id=build,
        source_data_sha256="d" * 64,
        source_schema_version=1,
        source_feature_version=1,
        data_time_semantics="current_vintage_observation_day",
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        evaluation_plan_hash="c" * 64,
        evaluation_cutoff=NOW,
        feature_order=features,
        scaler=StandardScalerArtifact(features, (0.0,), (1.0,), (1.0,)),
        hmm=GaussianHMMArtifact(
            state_count=2,
            feature_order=features,
            start_probabilities=(0.5, 0.5),
            transition_matrix=((0.9, 0.1), (0.1, 0.9)),
            means=((-1.0,), (1.0,)),
            full_covariances=(((1.0,),), ((1.5,),)),
        ),
        winning_seed=11,
        inference_origin_timestamp=datetime(2020, 1, 1, tzinfo=UTC),
        trained_through_timestamp=NOW - timedelta(days=1),
        terminal_filtered_probabilities=(0.4, 0.6),
        retained_observation_count=1500,
        skipped_incomplete_observation_count=0,
    )


def test_model_cache_single_flight_lru_and_two_version_atomic_retention() -> None:
    cache = ModelCache()
    started = Event()
    release_loader = Event()
    lock = Lock()
    load_count = 0
    leases: list[Any] = []

    def loader() -> ProductionModelArtifact:
        nonlocal load_count
        with lock:
            load_count += 1
        started.set()
        assert release_loader.wait(timeout=5)
        return _artifact("build-1")

    def acquire_first() -> None:
        leases.append(cache.acquire("regime-xetra:1", loader))

    threads = [Thread(target=acquire_first) for _ in range(4)]
    threads[0].start()
    assert started.wait(timeout=5)
    for thread in threads[1:]:
        thread.start()
    release_loader.set()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert load_count == 1
    assert cache.snapshot() == (("regime-xetra:1", 4),)
    for lease in leases:
        lease.release()

    old = cache.acquire("regime-xetra:1", lambda: _artifact("unexpected"))
    new = cache.acquire("regime-xetra:2", lambda: _artifact("build-2"))
    assert old.artifact.source_build_id == "build-1"
    assert new.artifact.source_build_id == "build-2"
    old.release()
    new.release()

    refreshed = cache.acquire("regime-xetra:2", lambda: _artifact("unexpected"))
    refreshed.release()
    third = cache.acquire("regime-xetra:3", lambda: _artifact("build-3"))
    assert cache.snapshot() == (("regime-xetra:2", 0), ("regime-xetra:3", 1))
    assert third.artifact.source_build_id == "build-3"
    third.release()


class _TimeoutContext:
    def __enter__(self) -> object:
        raise PoolTimeout("synthetic acquire timeout")

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback


class _TimeoutPool:
    def __init__(self) -> None:
        self.opened = False

    def open(self, *, wait: bool = False) -> None:
        assert wait is False
        self.opened = True

    def connection(self, timeout: float) -> _TimeoutContext:
        assert timeout == 5.0
        return _TimeoutContext()

    def close(self) -> None:
        self.opened = False


def test_feature_pool_acquire_timeout_is_bounded_and_propagated_without_external_postgres() -> None:
    fake = _TimeoutPool()
    settings = FeaturePostgresSettings(database="fixture", password="not-a-real-secret")
    pool = ProcessLocalPostgresPool(settings, workers=4, pool_factory=lambda _: fake)
    with pytest.raises(PoolTimeout, match="synthetic acquire timeout"):
        with pool.connection():
            raise AssertionError("unreachable")
    assert fake.opened
    pool.close()
    assert not fake.opened


def test_all_replay_413_503_504_paths_fail_closed_and_release_capacity() -> None:
    limits = ReplayLimits()
    start = datetime(2000, 1, 1, tzinfo=UTC)
    with pytest.raises(ReplayGuardrailError) as range_error:
        limits.validate_interval(start, start + timedelta(days=limits.max_range_days + 1))
    assert (range_error.value.status_code, range_error.value.error_code) == (
        413,
        "replay_range_too_large",
    )

    estimates = (
        {"response_rows": 10_001, "internal_rows": 10_001, "estimated_response_bytes": 1},
        {"response_rows": 1, "internal_rows": 15_001, "estimated_response_bytes": 1},
        {"response_rows": 1, "internal_rows": 1, "estimated_response_bytes": 26_214_401},
    )
    expected_codes = (
        "replay_row_limit",
        "replay_internal_row_limit",
        "replay_response_too_large",
    )
    for values, expected_code in zip(estimates, expected_codes, strict=True):
        with pytest.raises(ReplayGuardrailError) as error:
            limits.validate_estimates(**values)
        assert (error.value.status_code, error.value.error_code) == (413, expected_code)

    clock = [100.0]
    admission = ReplayAdmission(limits, clock=lambda: clock[0])
    with admission.admit() as permit:
        with pytest.raises(ReplayGuardrailError) as capacity:
            with admission.admit():
                raise AssertionError("unreachable")
        assert (capacity.value.status_code, capacity.value.error_code) == (
            503,
            "replay_capacity_exhausted",
        )
        clock[0] = permit.deadline + 0.001
        work = ["before-deadline-check"]
        with pytest.raises(ReplayGuardrailError) as timeout:
            permit.check_deadline()
            work.append("hidden-work-after-timeout")
        assert (timeout.value.status_code, timeout.value.error_code) == (504, "replay_timeout")
        assert work == ["before-deadline-check"]

    clock[0] = 200.0
    with admission.admit() as next_permit:
        next_permit.check_deadline()


class _Latest:
    def handle(self, **kwargs: object) -> dict[str, object]:
        return {
            "schema_version": "RegimeInvocationResponse.v1",
            "request_id": kwargs["request_id"],
            "profile_id": kwargs["profile_id"],
            "operation": "latest",
            "prediction_mode": "fixed_model_latest",
            "predictions": [],
        }


class _Replay:
    def handle(self, **kwargs: object) -> dict[str, object]:
        raise AssertionError(f"replay handler must remain occupied, not invoked: {tuple(kwargs)}")


class _OOS:
    def handle(self, **kwargs: object) -> dict[str, object]:
        return {"profile_id": kwargs["profile_id"], "build_id": kwargs["build_id"]}


def _dependencies() -> ServiceDependencies:
    return ServiceDependencies(
        latest_handler=_Latest(),  # type: ignore[arg-type]
        replay_handler=_Replay(),  # type: ignore[arg-type]
        oos_handler=_OOS(),  # type: ignore[arg-type]
        readiness=lambda: ReadinessSnapshot("healthy", True),
        request_id_factory=lambda: "capacity-request",
        request_time_factory=lambda: NOW,
    )


def test_exact_four_worker_capacity_does_not_block_health_tracking_registry_or_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose = yaml.safe_load(Path("compose.yaml").read_text(encoding="utf-8"))
    env = compose["services"]["mlflow"]["environment"]
    assert env["MLFLOW_WORKERS"] == "${MLFLOW_WORKERS:-4}"
    assert env["MLFLOW_THREADS_PER_WORKER"] == "${MLFLOW_THREADS_PER_WORKER:-4}"
    assert env["REGIME_PG_POOL_MAX_SIZE"] == "${REGIME_PG_POOL_MAX_SIZE:-4}"
    assert env["REGIME_FEATURE_PG_CONNECTION_BUDGET"] == (
        "${REGIME_FEATURE_PG_CONNECTION_BUDGET:-16}"
    )
    assert env["REGIME_REPLAY_MAX_CONCURRENCY_PER_WORKER"] == (
        "${REGIME_REPLAY_MAX_CONCURRENCY_PER_WORKER:-1}"
    )
    assert compose["services"]["mlflow"]["pull_policy"] == "never"
    assert compose["services"]["mlflow"]["image"] == "regime-engine-mlflow:local"

    workers = [ReplayAdmission(ReplayLimits()) for _ in range(4)]
    with ExitStack() as stack:
        permits = [stack.enter_context(worker.admit()) for worker in workers]
        assert len(permits) == 4

        app = create_app(dependencies=_dependencies())
        client = app.test_client()
        assert client.get("/health").status_code == 200
        custom_health = client.get("/regime-engine/v1/health")
        assert custom_health.status_code == 200
        assert custom_health.get_json()["ready"] is True
        latest = client.post(
            "/regime-engine/v1/profiles/xetra/invocations",
            json={"operation": "latest"},
        )
        assert latest.status_code == 200
        assert latest.get_json()["prediction_mode"] == "fixed_model_latest"

        monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
        tracking_uri = (tmp_path / "mlruns").as_uri()
        mlflow_client = MlflowClient(tracking_uri=tracking_uri)
        experiment_id = mlflow_client.create_experiment("capacity-proof")
        run = mlflow_client.create_run(experiment_id)
        fetched = mlflow_client.get_run(run.info.run_id)
        assert fetched.info.run_id == run.info.run_id
        model_name = "capacity-proof-model"
        mlflow_client.create_registered_model(model_name)
        registered = mlflow_client.get_registered_model(model_name)
        assert registered.name == model_name


def test_capacity_proof_never_embeds_secrets_raw_features_or_remote_image_assumptions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.clear()
    compose_text = Path("compose.yaml").read_text(encoding="utf-8").lower()
    assert "pull_policy: never" in compose_text
    assert "regime-engine-mlflow:local" in compose_text
    for forbidden in (
        "regime_feature_pgpassword=",
        "mlflow_backend_db_password=",
        "raw_feature_vector",
        "ghcr.io/",
        ":5001",
        "prometheus",
    ):
        assert forbidden not in caplog.text.lower()
        assert forbidden not in compose_text
