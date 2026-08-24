"""Synchronous per-worker replay admission and cooperative deadline checks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import BoundedSemaphore
from time import monotonic

from market_regime_engine.serving.replay_limits import ReplayGuardrailError, ReplayLimits


@dataclass(frozen=True, slots=True)
class ReplayPermit:
    deadline: float
    clock: Callable[[], float]

    def check_deadline(self) -> None:
        if self.clock() > self.deadline:
            raise ReplayGuardrailError(
                504,
                "replay_timeout",
                "replay cooperative deadline exceeded",
                True,
            )


class ReplayAdmission:
    """Bound admitted synchronous replay work without introducing an executor."""

    def __init__(
        self,
        limits: ReplayLimits,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._limits = limits
        self._clock = clock
        self._semaphore = BoundedSemaphore(limits.max_concurrency_per_worker)

    @contextmanager
    def admit(self) -> Iterator[ReplayPermit]:
        acquired = self._semaphore.acquire(blocking=False)
        if not acquired:
            raise ReplayGuardrailError(
                503,
                "replay_capacity_exhausted",
                "replay capacity is exhausted for this worker",
                True,
            )
        try:
            permit = ReplayPermit(
                deadline=self._clock() + self._limits.timeout_seconds,
                clock=self._clock,
            )
            yield permit
        finally:
            self._semaphore.release()
