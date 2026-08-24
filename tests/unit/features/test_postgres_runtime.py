from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType

import pytest

from market_regime_engine.features.postgres_pool import (
    ConnectionLike,
    ProcessLocalPostgresPool,
)
from market_regime_engine.features.postgres_settings import FeaturePostgresSettings


class FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...] | None]] = []

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> object:
        self.executed.append((query, params))
        return object()


class FakeConnectionContext(AbstractContextManager[ConnectionLike]):
    def __init__(self, connection: FakeConnection) -> None:
        self.connection_value = connection

    def __enter__(self) -> ConnectionLike:
        return self.connection_value

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class FakePool:
    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.connection_value = FakeConnection()
        self.timeouts: list[float] = []

    def open(self, *, wait: bool = False) -> None:
        assert wait is False
        self.opened = True

    def connection(self, timeout: float) -> AbstractContextManager[ConnectionLike]:
        self.timeouts.append(timeout)
        return FakeConnectionContext(self.connection_value)

    def close(self) -> None:
        self.closed = True


def test_exact_defaults_plain_transport_password_file_and_safe_summary(tmp_path: Path) -> None:
    secret = tmp_path / "feature-password"
    secret.write_text("s3cr3t\n", encoding="utf-8")
    settings = FeaturePostgresSettings.from_env(
        {
            "REGIME_FEATURE_PGDATABASE": "features",
            "REGIME_FEATURE_PGPASSWORD_FILE": str(secret),
        }
    )
    assert settings.host == "10.10.1.3"
    assert settings.port == 54321
    assert settings.user == "regime-engine"
    assert settings.sslmode == "disable"
    assert settings.pool_min_size == 1
    assert settings.pool_max_size == 4
    assert settings.acquire_timeout_seconds == 5
    assert settings.statement_timeout_seconds == 30
    assert settings.connection_budget == 16
    assert settings.connection_kwargs()["password"] == "s3cr3t"
    assert "s3cr3t" not in repr(settings)
    assert "s3cr3t" not in repr(settings.safe_summary())


def test_required_database_password_and_transport_contract_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="PGDATABASE"):
        FeaturePostgresSettings.from_env({"REGIME_FEATURE_PGPASSWORD": "x"})
    with pytest.raises(ValueError, match="password"):
        FeaturePostgresSettings.from_env({"REGIME_FEATURE_PGDATABASE": "features"})
    secret = tmp_path / "secret"
    secret.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="only one"):
        FeaturePostgresSettings.from_env(
            {
                "REGIME_FEATURE_PGDATABASE": "features",
                "REGIME_FEATURE_PGPASSWORD": "x",
                "REGIME_FEATURE_PGPASSWORD_FILE": str(secret),
            }
        )
    with pytest.raises(ValueError, match="sslmode"):
        FeaturePostgresSettings.from_env(
            {
                "REGIME_FEATURE_PGDATABASE": "features",
                "REGIME_FEATURE_PGPASSWORD": "x",
                "REGIME_FEATURE_PGSSLMODE": "require",
            }
        )


def test_worker_budget_and_lazy_pool_statement_timeout_and_close() -> None:
    settings = FeaturePostgresSettings(database="features", password="secret")
    settings.validate_worker_budget(4)
    with pytest.raises(ValueError, match="budget"):
        settings.validate_worker_budget(5)

    fake = FakePool()
    created = 0

    def factory(_: FeaturePostgresSettings) -> FakePool:
        nonlocal created
        created += 1
        return fake

    runtime = ProcessLocalPostgresPool(settings, workers=4, pool_factory=factory)
    assert created == 0
    assert runtime.is_open is False
    assert "secret" not in repr(runtime)
    with runtime.connection() as connection:
        assert connection is fake.connection_value
    assert created == 1
    assert fake.opened
    assert fake.timeouts == [5.0]
    assert fake.connection_value.executed == [("SET statement_timeout = %s", (30000,))]
    runtime.close()
    assert fake.closed
    assert runtime.is_open is False


def test_numeric_env_validation() -> None:
    with pytest.raises(ValueError, match="integer"):
        FeaturePostgresSettings.from_env(
            {
                "REGIME_FEATURE_PGDATABASE": "features",
                "REGIME_FEATURE_PGPASSWORD": "x",
                "REGIME_PG_POOL_MAX_SIZE": "bad",
            }
        )
    with pytest.raises(ValueError, match="numeric"):
        FeaturePostgresSettings.from_env(
            {
                "REGIME_FEATURE_PGDATABASE": "features",
                "REGIME_FEATURE_PGPASSWORD": "x",
                "REGIME_PG_STATEMENT_TIMEOUT_SECONDS": "bad",
            }
        )
