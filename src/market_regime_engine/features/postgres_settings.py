"""Production feature-PostgreSQL runtime settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_FEATURE_HOST = "10.10.1.3"
_FEATURE_PORT = 54321
_FEATURE_USER = "regime-engine"
_FEATURE_SSLMODE = "require"
_POOL_MIN = 1
_POOL_MAX = 4
_ACQUIRE_TIMEOUT = 5.0
_STATEMENT_TIMEOUT = 30.0
_CONNECTION_BUDGET = 16


@dataclass(frozen=True, slots=True)
class FeaturePostgresSettings:
    database: str
    password: str
    host: str = _FEATURE_HOST
    port: int = _FEATURE_PORT
    user: str = _FEATURE_USER
    sslmode: str = _FEATURE_SSLMODE
    pool_min_size: int = _POOL_MIN
    pool_max_size: int = _POOL_MAX
    acquire_timeout_seconds: float = _ACQUIRE_TIMEOUT
    statement_timeout_seconds: float = _STATEMENT_TIMEOUT
    connection_budget: int = _CONNECTION_BUDGET

    def __post_init__(self) -> None:
        for field_name in ("database", "password"):
            value = getattr(self, field_name)
            if not value or value.strip() != value:
                raise ValueError(f"{field_name} must be a non-empty trimmed value")
        if self.host != _FEATURE_HOST or self.port != _FEATURE_PORT:
            raise ValueError("feature PostgreSQL host/port differ from the production contract")
        if self.user != _FEATURE_USER:
            raise ValueError("feature PostgreSQL user must be exactly regime-engine")
        if self.sslmode != _FEATURE_SSLMODE:
            raise ValueError("feature PostgreSQL sslmode must be exactly require")
        invalid_pool_bounds = (
            self.pool_min_size < 0
            or self.pool_max_size < 1
            or self.pool_min_size > self.pool_max_size
        )
        if invalid_pool_bounds:
            raise ValueError("invalid feature PostgreSQL pool bounds")
        if self.acquire_timeout_seconds <= 0.0 or self.statement_timeout_seconds <= 0.0:
            raise ValueError("PostgreSQL timeouts must be positive")
        if self.connection_budget < 1:
            raise ValueError("feature PostgreSQL connection budget must be positive")

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> FeaturePostgresSettings:
        database = _required(env, "REGIME_FEATURE_PGDATABASE")
        direct = env.get("REGIME_FEATURE_PGPASSWORD")
        password_file = env.get("REGIME_FEATURE_PGPASSWORD_FILE")
        if direct and password_file:
            raise ValueError("configure only one feature PostgreSQL password source")
        if password_file:
            password = Path(password_file).read_text(encoding="utf-8").strip()
        elif direct:
            password = direct
        else:
            raise ValueError("feature PostgreSQL password or password file is required")

        return cls(
            database=database,
            password=password,
            host=env.get("REGIME_FEATURE_PGHOST", _FEATURE_HOST),
            port=_integer(env, "REGIME_FEATURE_PGPORT", _FEATURE_PORT),
            user=env.get("REGIME_FEATURE_PGUSER", _FEATURE_USER),
            sslmode=env.get("REGIME_FEATURE_PGSSLMODE", _FEATURE_SSLMODE),
            pool_min_size=_integer(env, "REGIME_PG_POOL_MIN_SIZE", _POOL_MIN),
            pool_max_size=_integer(env, "REGIME_PG_POOL_MAX_SIZE", _POOL_MAX),
            acquire_timeout_seconds=_float(
                env,
                "REGIME_PG_ACQUIRE_TIMEOUT_SECONDS",
                _ACQUIRE_TIMEOUT,
            ),
            statement_timeout_seconds=_float(
                env, "REGIME_PG_STATEMENT_TIMEOUT_SECONDS", _STATEMENT_TIMEOUT
            ),
            connection_budget=_integer(
                env, "REGIME_FEATURE_PG_CONNECTION_BUDGET", _CONNECTION_BUDGET
            ),
        )

    def connection_kwargs(self) -> dict[str, str | int]:
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.database,
            "user": self.user,
            "password": self.password,
            "sslmode": self.sslmode,
        }

    def validate_worker_budget(self, workers: int) -> None:
        if workers < 1:
            raise ValueError("MLflow worker count must be positive")
        if workers * self.pool_max_size > self.connection_budget:
            raise ValueError("feature PostgreSQL worker/pool product exceeds connection budget")

    def safe_summary(self) -> dict[str, str | int | float]:
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "sslmode": self.sslmode,
            "pool_min_size": self.pool_min_size,
            "pool_max_size": self.pool_max_size,
            "acquire_timeout_seconds": self.acquire_timeout_seconds,
            "statement_timeout_seconds": self.statement_timeout_seconds,
            "connection_budget": self.connection_budget,
        }


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name)
    if value is None or not value or value.strip() != value:
        raise ValueError(f"{name} is required")
    return value


def _integer(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(env.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
