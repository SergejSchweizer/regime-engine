"""Process-local single-flight production model cache with safe two-version retention."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Condition, RLock
from time import monotonic

from market_regime_engine.models.production_artifact import ProductionModelArtifact


class ModelCacheCapacityError(RuntimeError):
    """Raised when all retained versions are in use and a third version cannot be admitted."""


@dataclass(slots=True)
class _CacheEntry:
    artifact: ProductionModelArtifact
    references: int
    last_used: float


class ModelLease:
    def __init__(
        self,
        cache: ModelCache,
        cache_key: str,
        artifact: ProductionModelArtifact,
    ) -> None:
        self._cache = cache
        self.cache_key = cache_key
        self.artifact = artifact
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._cache._release(self.cache_key)
            self._released = True

    def __enter__(self) -> ModelLease:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.release()


class ModelCache:
    """At most two exact versions, with ref-count-safe LRU eviction and single-flight loads."""

    def __init__(self, *, max_versions: int = 2) -> None:
        if max_versions != 2:
            raise ValueError("production model cache max_versions must be exactly 2")
        self._max_versions = max_versions
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._entries: dict[str, _CacheEntry] = {}
        self._loading: set[str] = set()

    def acquire(
        self,
        cache_key: str,
        loader: Callable[[], ProductionModelArtifact],
    ) -> ModelLease:
        if not cache_key:
            raise ValueError("cache key cannot be empty")

        with self._condition:
            while True:
                entry = self._entries.get(cache_key)
                if entry is not None:
                    entry.references += 1
                    entry.last_used = monotonic()
                    return ModelLease(self, cache_key, entry.artifact)
                if cache_key not in self._loading:
                    self._loading.add(cache_key)
                    break
                self._condition.wait()

        try:
            artifact = loader()
            if type(artifact) is not ProductionModelArtifact:
                raise TypeError("model cache loader must return a ProductionModelArtifact")
        except BaseException:
            with self._condition:
                self._loading.remove(cache_key)
                self._condition.notify_all()
            raise

        with self._condition:
            try:
                self._evict_for_insert()
                self._entries[cache_key] = _CacheEntry(
                    artifact=artifact,
                    references=1,
                    last_used=monotonic(),
                )
                return ModelLease(self, cache_key, artifact)
            finally:
                self._loading.remove(cache_key)
                self._condition.notify_all()

    def _evict_for_insert(self) -> None:
        if len(self._entries) < self._max_versions:
            return
        candidates = (
            (entry.last_used, key) for key, entry in self._entries.items() if entry.references == 0
        )
        try:
            _, victim = min(candidates)
        except ValueError as exc:
            raise ModelCacheCapacityError(
                "two cached model versions are currently referenced; cannot admit a third"
            ) from exc
        del self._entries[victim]

    def _release(self, cache_key: str) -> None:
        with self._condition:
            entry = self._entries.get(cache_key)
            if entry is None or entry.references < 1:
                raise RuntimeError("model lease release has no matching cache reference")
            entry.references -= 1
            entry.last_used = monotonic()
            self._condition.notify_all()

    def snapshot(self) -> tuple[tuple[str, int], ...]:
        with self._condition:
            return tuple(sorted((key, entry.references) for key, entry in self._entries.items()))
