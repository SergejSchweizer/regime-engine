"""Deterministic eight-seed multistart selection for Gaussian HMM fitting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from market_regime_engine.evaluation.errors import RecoverableEvaluationInvalidity
from market_regime_engine.evaluations.process_parallel import is_pickleable
from market_regime_engine.evaluations.task_frontier import FrontierTask, SharedTaskFrontier
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import FitResult, GaussianHMMAdapter
from market_regime_engine.runtime.cpu import cpu_worker_count
from market_regime_engine.runtime.parallel import ReadOnlyMatrix

if TYPE_CHECKING:
    from market_regime_engine.evaluation_runs.hmm_units import HMMSeedCheckpoint

MULTISTART_SEEDS = (11, 23, 37, 53, 71, 89, 107, 131)
MINIMUM_VALID_STARTS = 6
MINIMUM_SUCCESS_RATE = 0.75
TRAIN_LOGLIK_TIE_ABS_TOLERANCE = 1e-12


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


@dataclass(frozen=True, slots=True)
class _FrontierStartPayload:
    matrix_path: str
    matrix_shape: tuple[int, int]
    matrix_dtype: str
    state_count: int
    adapter_factory: AdapterFactory
    seed: int
    retryable_technical_failure: bool


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
    retryable_technical_failure: bool = False,
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
    except RecoverableEvaluationInvalidity as exc:
        # Only an explicitly typed statistical invalidity is safe to cache.
        return _failure(seed, f"{type(exc).__name__}: {exc}"), None
    except Exception:
        # An unexpected adapter/backend failure is never statistical evidence.
        # It must reach the parent unchanged, regardless of checkpoint mode.
        raise


def _evaluate_start_in_frontier(
    task: FrontierTask[_FrontierStartPayload],
) -> tuple[StartDiagnostic, FitResult | None]:
    payload = task.payload
    mapped = np.memmap(
        payload.matrix_path,
        dtype=np.dtype(payload.matrix_dtype),
        mode="r",
        shape=payload.matrix_shape,
    )
    mapped.flags.writeable = False
    try:
        return _evaluate_start(
            mapped,
            state_count=payload.state_count,
            adapter_factory=payload.adapter_factory,
            seed=payload.seed,
            retryable_technical_failure=payload.retryable_technical_failure,
        )
    finally:
        del mapped


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


def run_multistart(
    train_rows: npt.ArrayLike,
    *,
    state_count: int,
    adapter_factory: AdapterFactory,
    max_workers: int | None = None,
    checkpoint: HMMSeedCheckpoint | None = None,
    frontier: SharedTaskFrontier[_FrontierStartPayload, tuple[StartDiagnostic, FitResult | None]]
    | None = None,
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
        def map_frontier(
            active_frontier: SharedTaskFrontier[
                _FrontierStartPayload, tuple[StartDiagnostic, FitResult | None]
            ],
        ) -> dict[int, tuple[StartDiagnostic, FitResult | None]]:
            matrix = np.ascontiguousarray(np.asarray(train_rows))
            if matrix.ndim != 2:
                raise ValueError("multistart train_rows must be a two-dimensional numeric matrix")
            with (
                TemporaryDirectory(prefix="regime-multistart-") as matrix_directory,
                ReadOnlyMatrix.create(matrix, matrix_directory) as shared_matrix,
            ):
                frontier_tasks = tuple(
                    FrontierTask(
                        task_id=f"multistart:{state_count}:{seed}",
                        state_count=state_count,
                        fold_id="multistart",
                        candidate_subset=(f"seed:{seed}",),
                        seed=index,
                        profile_hash="0" * 64,
                        matrix_identity=shared_matrix.identity,
                        row_indices=(0, matrix.shape[0] - 1),
                        column_indices=(0, matrix.shape[1] - 1),
                        payload=_FrontierStartPayload(
                            str(shared_matrix.path),
                            (int(matrix.shape[0]), int(matrix.shape[1])),
                            matrix.dtype.str,
                            state_count,
                            adapter_factory,
                            seed,
                            checkpoint is not None,
                        ),
                    )
                    for index, seed in enumerate(pending_seeds)
                )
                frontier_result = active_frontier.map(frontier_tasks, _evaluate_start_in_frontier)
                by_task_id = {task.task_id: item for task, item in frontier_result.values}
                mapped_results = {
                    seed: by_task_id[f"multistart:{state_count}:{seed}"]
                    for seed in pending_seeds
                }
            if checkpoint is not None:
                from market_regime_engine.evaluation_runs.hmm_units import SeedFitOutcome

                for seed, outcome in mapped_results.items():
                    checkpoint.save(seed, SeedFitOutcome(*outcome))
            return mapped_results

        if frontier is not None:
            pending_results = map_frontier(frontier)
        elif pending_worker_limit == 1:
            pending_results = {}
            for seed in pending_seeds:
                outcome = _evaluate_start(
                    train_rows,
                    state_count=state_count,
                    adapter_factory=adapter_factory,
                    seed=seed,
                    retryable_technical_failure=checkpoint is not None,
                )
                pending_results[seed] = outcome
                if checkpoint is not None:
                    from market_regime_engine.evaluation_runs.hmm_units import SeedFitOutcome

                    checkpoint.save(seed, SeedFitOutcome(*outcome))
        elif is_pickleable(adapter_factory):
            # A direct caller gets the same shared frontier boundary; callers
            # spanning multiple candidates/folds can pass their own instance.
            with SharedTaskFrontier[
                _FrontierStartPayload, tuple[StartDiagnostic, FitResult | None]
            ](pending_worker_limit) as owned_frontier:
                pending_results = map_frontier(owned_frontier)
        else:
            raise TypeError(
                "CPU-bound multistart requires a pickleable adapter_factory when "
                f"max_workers={pending_worker_limit}; provide a process-safe factory "
                "or set max_workers=1 for serial execution"
            )

        for seed, outcome in pending_results.items():
            evaluated_by_seed[seed] = outcome

    evaluated = tuple(evaluated_by_seed[seed] for seed in MULTISTART_SEEDS)

    diagnostics = [diagnostic for diagnostic, _result in evaluated]
    valid_results: list[FitResult] = []
    valid_results.extend(result for _diagnostic, result in evaluated if result is not None)

    valid_count = len(valid_results)
    success_rate = valid_count / len(MULTISTART_SEEDS)
    if valid_count < MINIMUM_VALID_STARTS or success_rate < MINIMUM_SUCCESS_RATE:
        raise RecoverableEvaluationInvalidity(
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
