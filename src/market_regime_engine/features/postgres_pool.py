"""Lazy process-local psycopg pool for production feature reads."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from threading import Lock
from typing import Any, Protocol, cast

from psycopg_pool import ConnectionPool

from market_regime_engine.features.postgres_settings import FeaturePostgresSettings


class ConnectionLike(Protocol):
    def execute(self, query: str, params: tuple[object, ...] | None = None) -> object: ...


class PoolLike(Protocol):
    def open(self, *, wait: bool = False) -> None: ...

    def connection(self, timeout: float) -> AbstractContextManager[ConnectionLike]: ...

    def close(self) -> None: ...


PoolFactory = Callable[[FeaturePostgresSettings], PoolLike]


def _default_pool_factory(settings: FeaturePostgresSettings) -> PoolLike:
    pool = ConnectionPool[Any](
        kwargs=settings.connection_kwargs(),
        min_size=settings.pool_min_size,
        max_size=settings.pool_max_size,
        timeout=settings.acquire_timeout_seconds,
        open=False,
        name="regime-engine-feature-pool",
    )
    return cast(PoolLike, pool)


class ProcessLocalPostgresPool:
    """Create a pool only on first use and never share an opened pool across forked workers."""

    def __init__(
        self,
        settings: FeaturePostgresSettings,
        *,
        workers: int,
        pool_factory: PoolFactory = _default_pool_factory,
    ) -> None:
        settings.validate_worker_budget(workers)
        self._settings = settings
        self._pool_factory = pool_factory
        self._pool: PoolLike | None = None
        self._owner_pid = os.getpid()
        self._lock = Lock()

    def _ensure_pool(self) -> PoolLike:
        current_pid = os.getpid()
        with self._lock:
            if current_pid != self._owner_pid:
                self._pool = None
                self._owner_pid = current_pid
            if self._pool is None:
                pool = self._pool_factory(self._settings)
                pool.open(wait=False)
                self._pool = pool
            return self._pool

    @contextmanager
    def connection(self) -> Iterator[ConnectionLike]:
        pool = self._ensure_pool()
        with pool.connection(timeout=self._settings.acquire_timeout_seconds) as connection:
            milliseconds = int(self._settings.statement_timeout_seconds * 1000)
            connection.execute("SET statement_timeout = %s", (milliseconds,))
            yield connection

    def close(self) -> None:
        with self._lock:
            if self._pool is not None:
                self._pool.close()
                self._pool = None

    @property
    def is_open(self) -> bool:
        return self._pool is not None

    def __repr__(self) -> str:
        summary = self._settings.safe_summary()
        return f"ProcessLocalPostgresPool(settings={summary!r}, open={self.is_open})"
