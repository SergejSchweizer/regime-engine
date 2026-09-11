"""Deterministic eight-seed multistart selection for Gaussian HMM fitting."""

from __future__ import annotations

import multiprocessing
import pickle
import threading
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

import numpy.typing as npt

from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import FitResult, GaussianHMMAdapter
from market_regime_engine.runtime.cpu import available_cpu_count, cpu_worker_count
from market_regime_engine.training.adapter_factory import CandidateAdapterFactory

if TYPE_CHECKING:
    from market_regime_engine.evaluation_runs.hmm_units import HMMSeedCheckpoint

MULTISTART_SEEDS = (11, 23, 37, 53, 71, 89, 107, 131)
MINIMUM_VALID_STARTS = 6
MINIMUM_SUCCESS_RATE = 0.75
TRAIN_LOGLIK_TIE_ABS_TOLERANCE = 1e-12
_CPU_SLOT_COUNT = available_cpu_count()
_CPU_SLOTS = threading.BoundedSemaphore(_CPU_SLOT_COUNT)


@dataclass(frozen=True, slots=True)
class StartDiagnostic:
    seed: int
    success: bool
    converged: bool
    iterations: int | None
    train_log_likelihood: float | None
    artifact: GaussianHMMArtifact | None
    failure_reason: str | None

    def __post_init__(self) -> None:
        if self.seed not in MULTISTART_SEEDS:
            raise ValueError("diagnostic seed is outside the pinned multistart set")
        if self.success:
            if not self.converged:
                raise ValueError("successful start must be converged")
            if self.iterations is None or self.iterations < 1:
                raise ValueError("successful start requires positive iterations")
            if self.train_log_likelihood is None or not isfinite(self.train_log_likelihood):
                raise ValueError("successful start requires finite TRAIN log likelihood")
            if self.artifact is None or self.failure_reason is not None:
                raise ValueError("successful start requires artifact and no failure reason")
        elif not self.failure_reason:
            raise ValueError("failed start requires a failure reason")


@dataclass(frozen=True, slots=True)
class MultistartResult:
    state_count: int
    winner: FitResult
    diagnostics: tuple[StartDiagnostic, ...]

    def __post_init__(self) -> None:
        if self.state_count not in (2, 3, 4, 5):
            raise ValueError("state_count must be K=2,3,4,5")
        if tuple(item.seed for item in self.diagnostics) != MULTISTART_SEEDS:
            raise ValueError("diagnostics must retain all eight starts in pinned seed order")
        valid = sum(item.success for item in self.diagnostics)
        if valid < MINIMUM_VALID_STARTS or valid / len(MULTISTART_SEEDS) < MINIMUM_SUCCESS_RATE:
            raise ValueError("multistart result does not satisfy the 6/8 and 0.75 gates")
        if self.winner.seed not in {item.seed for item in self.diagnostics if item.success}:
            raise ValueError("winner must be one of the valid retained starts")

    @property
    def valid_start_count(self) -> int:
        return sum(item.success for item in self.diagnostics)

    @property
    def success_rate(self) -> float:
        return self.valid_start_count / len(self.diagnostics)


AdapterFactory = Callable[[], GaussianHMMAdapter]


def _successful_diagnostic(result: FitResult) -> StartDiagnostic:
    return StartDiagnostic(
        seed=result.seed,
        success=True,
        converged=True,
        iterations=result.iterations,
        train_log_likelihood=result.train_log_likelihood,
        artifact=result.artifact,
        failure_reason=None,
    )


def _failure(seed: int, reason: str, *, converged: bool = False) -> StartDiagnostic:
    return StartDiagnostic(
        seed=seed,
        success=False,
        converged=converged,
        iterations=None,
        train_log_likelihood=None,
        artifact=None,
        failure_reason=reason,
    )


def _evaluate_start(
    train_rows: npt.ArrayLike,
    *,
    state_count: int,
    adapter_factory: AdapterFactory,
    seed: int,
) -> tuple[StartDiagnostic, FitResult | None]:
    try:
        result = adapter_factory().fit(train_rows, state_count, seed)
        if result.seed != seed:
            raise ValueError("adapter returned a mismatched seed")
        if not result.converged:
            return _failure(seed, "not converged"), None
        if result.iterations < 1:
            return _failure(seed, "invalid iteration count", converged=True), None
        if not isfinite(result.train_log_likelihood):
            return _failure(seed, "non-finite TRAIN log likelihood", converged=True), None
        return _successful_diagnostic(result), result
    except Exception as exc:
        return _failure(seed, f"{type(exc).__name__}: {exc}"), None


def _pickleable(value: object) -> bool:
    try:
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except pickle.PickleError, TypeError, AttributeError:
        return False
    return True


def _anchored_winner(valid_results: list[FitResult]) -> FitResult:
    """Choose from starts tied to the exact global maximum likelihood."""

    if not valid_results:
        raise ValueError("winner selection requires successful starts")
    global_maximum = max(result.train_log_likelihood for result in valid_results)
    tied_results = tuple(
        result
        for result in valid_results
        if global_maximum - result.train_log_likelihood <= TRAIN_LOGLIK_TIE_ABS_TOLERANCE
    )
    return min(tied_results, key=lambda result: result.seed)


@contextmanager
def _reserve_cpu_slots(requested: int) -> Iterator[int]:
    """Reserve host CPU slots for one independent multistart batch."""

    acquired = 0
    target = min(max(1, requested), _CPU_SLOT_COUNT)
    try:
        while acquired < target:
            if _CPU_SLOTS.acquire(blocking=False):
                acquired += 1
                continue
            if acquired == 0:
                _CPU_SLOTS.acquire()
                acquired = 1
            break
        yield acquired
    finally:
        for _ in range(acquired):
            _CPU_SLOTS.release()


def run_multistart(
    train_rows: npt.ArrayLike,
    *,
    state_count: int,
    adapter_factory: AdapterFactory,
    max_workers: int | None = None,
    checkpoint: HMMSeedCheckpoint | None = None,
) -> MultistartResult:
    """Fit exactly eight starts and choose the valid TRAIN-loglik winner deterministically."""

    if state_count not in (2, 3, 4, 5):
        raise ValueError("state_count must be K=2,3,4,5")

    # There are only eight independent starts in one multistart.  Use all of
    # those lanes, while the slot reservation below bounds concurrent nested
    # multistarts across outer folds and candidate grids by host CPU count.
    worker_limit = cpu_worker_count(max_workers, task_count=len(MULTISTART_SEEDS))

    evaluated_by_seed: dict[int, tuple[StartDiagnostic, FitResult | None]] = {}
    pending_seeds: list[int] = []
    for seed in MULTISTART_SEEDS:
        if checkpoint is not None:
            cached = checkpoint.load(seed)
            if cached is not None:
                evaluated_by_seed[seed] = (cached.diagnostic, cached.result)
                continue
        pending_seeds.append(seed)

    if pending_seeds:
        pending_worker_limit = min(worker_limit, len(pending_seeds))
        save_results_in_parent = False
        if pending_worker_limit == 1:
            pending_results = {}
            for seed in pending_seeds:
                outcome = _evaluate_start(
                    train_rows,
                    state_count=state_count,
                    adapter_factory=adapter_factory,
                    seed=seed,
                )
                pending_results[seed] = outcome
                if checkpoint is not None:
                    from market_regime_engine.evaluation_runs.hmm_units import SeedFitOutcome

                    checkpoint.save(seed, SeedFitOutcome(*outcome))
        elif isinstance(adapter_factory, CandidateAdapterFactory) and _pickleable(adapter_factory):
            # hmmlearn fitting is CPU-bound and its Python-facing orchestration
            # does not scale reliably in a thread pool.  Process workers give
            # each independent seed its own interpreter/GIL while preserving
            # the exact pinned seed order in the parent.  A threaded caller
            # must use spawn so workers do not inherit its locks; a top-level
            # caller can use fork without that nested-parent hazard.
            process_context_name = (
                "fork" if threading.current_thread() is threading.main_thread() else "spawn"
            )
            with (
                _reserve_cpu_slots(pending_worker_limit) as reserved_workers,
                warnings.catch_warnings(),
            ):
                if process_context_name == "fork":
                    warnings.filterwarnings(
                        "ignore",
                        message=(
                            r"This process .* is multi-threaded, use os.fork\(\) may lead "
                            r"to deadlocks"
                        ),
                        category=DeprecationWarning,
                        module=r"multiprocessing\.popen_fork",
                    )
                with ProcessPoolExecutor(
                    max_workers=reserved_workers,
                    mp_context=multiprocessing.get_context(process_context_name),
                ) as process_executor:
                    futures = {
                        seed: process_executor.submit(
                            _evaluate_start,
                            train_rows,
                            state_count=state_count,
                            adapter_factory=adapter_factory,
                            seed=seed,
                        )
                        for seed in pending_seeds
                    }
                    future_seeds = {future: seed for seed, future in futures.items()}
                    pending_results = {}
                    try:
                        for future in as_completed(futures.values()):
                            seed = future_seeds[future]
                            outcome = future.result()
                            pending_results[seed] = outcome
                            if checkpoint is not None:
                                from market_regime_engine.evaluation_runs.hmm_units import (
                                    SeedFitOutcome,
                                )

                                checkpoint.save(seed, SeedFitOutcome(*outcome))
                    except BaseException:
                        # Persist every successful future that completed before
                        # the interruption surfaced.  Pending futures remain
                        # reclaimable and will be retried on restart.
                        for future, seed in future_seeds.items():
                            if not future.done() or future.cancelled():
                                continue
                            try:
                                outcome = future.result()
                            except BaseException:
                                continue
                            pending_results[seed] = outcome
                            if checkpoint is not None:
                                from market_regime_engine.evaluation_runs.hmm_units import (
                                    SeedFitOutcome,
                                )

                                checkpoint.save(seed, SeedFitOutcome(*outcome))
                        raise
            save_results_in_parent = True
        else:
            # Custom adapters used by unit tests and extension callers may be
            # closures and cannot be sent to process workers.
            def evaluate_thread(seed: int) -> tuple[StartDiagnostic, FitResult | None]:
                outcome = _evaluate_start(
                    train_rows,
                    state_count=state_count,
                    adapter_factory=adapter_factory,
                    seed=seed,
                )
                if checkpoint is not None:
                    from market_regime_engine.evaluation_runs.hmm_units import SeedFitOutcome

                    checkpoint.save(seed, SeedFitOutcome(*outcome))
                return outcome

            with (
                _reserve_cpu_slots(pending_worker_limit) as reserved_workers,
                ThreadPoolExecutor(max_workers=reserved_workers) as thread_executor,
            ):
                futures = {
                    seed: thread_executor.submit(evaluate_thread, seed) for seed in pending_seeds
                }
                pending_results = {seed: futures[seed].result() for seed in pending_seeds}

        for seed, outcome in pending_results.items():
            evaluated_by_seed[seed] = outcome
            if checkpoint is not None and save_results_in_parent:
                from market_regime_engine.evaluation_runs.hmm_units import SeedFitOutcome

                checkpoint.save(seed, SeedFitOutcome(*outcome))

    evaluated = tuple(evaluated_by_seed[seed] for seed in MULTISTART_SEEDS)

    diagnostics = [diagnostic for diagnostic, _result in evaluated]
    valid_results: list[FitResult] = []
    valid_results.extend(result for _diagnostic, result in evaluated if result is not None)

    valid_count = len(valid_results)
    success_rate = valid_count / len(MULTISTART_SEEDS)
    if valid_count < MINIMUM_VALID_STARTS or success_rate < MINIMUM_SUCCESS_RATE:
        raise ValueError(
            "multistart gate failed: "
            f"valid_starts={valid_count}/8 success_rate={success_rate:.6f}; "
            f"failures={[item.failure_reason for item in diagnostics if not item.success]}"
        )

    winner = _anchored_winner(valid_results)
    return MultistartResult(
        state_count=state_count,
        winner=winner,
        diagnostics=tuple(diagnostics),
    )
