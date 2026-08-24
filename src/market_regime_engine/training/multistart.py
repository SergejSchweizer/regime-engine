"""Deterministic eight-seed multistart selection for Gaussian HMM fitting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import numpy.typing as npt

from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import FitResult, GaussianHMMAdapter

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
        if self.state_count not in (2, 3, 4):
            raise ValueError("state_count must be K=2,3,4")
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


def _better(candidate: FitResult, current: FitResult) -> bool:
    difference = candidate.train_log_likelihood - current.train_log_likelihood
    if difference > TRAIN_LOGLIK_TIE_ABS_TOLERANCE:
        return True
    if abs(difference) <= TRAIN_LOGLIK_TIE_ABS_TOLERANCE:
        return candidate.seed < current.seed
    return False


def run_multistart(
    train_rows: npt.ArrayLike,
    *,
    state_count: int,
    adapter_factory: AdapterFactory,
) -> MultistartResult:
    """Fit exactly eight starts and choose the valid TRAIN-loglik winner deterministically."""

    if state_count not in (2, 3, 4):
        raise ValueError("state_count must be K=2,3,4")

    diagnostics: list[StartDiagnostic] = []
    valid_results: list[FitResult] = []
    for seed in MULTISTART_SEEDS:
        try:
            result = adapter_factory().fit(train_rows, state_count, seed)
            if result.seed != seed:
                raise ValueError("adapter returned a mismatched seed")
            if not result.converged:
                diagnostics.append(_failure(seed, "not converged"))
                continue
            if result.iterations < 1:
                diagnostics.append(_failure(seed, "invalid iteration count", converged=True))
                continue
            if not isfinite(result.train_log_likelihood):
                diagnostics.append(
                    _failure(seed, "non-finite TRAIN log likelihood", converged=True)
                )
                continue
            diagnostics.append(_successful_diagnostic(result))
            valid_results.append(result)
        except Exception as exc:
            diagnostics.append(_failure(seed, f"{type(exc).__name__}: {exc}"))

    valid_count = len(valid_results)
    success_rate = valid_count / len(MULTISTART_SEEDS)
    if valid_count < MINIMUM_VALID_STARTS or success_rate < MINIMUM_SUCCESS_RATE:
        raise ValueError(
            "multistart gate failed: "
            f"valid_starts={valid_count}/8 success_rate={success_rate:.6f}; "
            f"failures={[item.failure_reason for item in diagnostics if not item.success]}"
        )

    winner = valid_results[0]
    for candidate in valid_results[1:]:
        if _better(candidate, winner):
            winner = candidate
    return MultistartResult(
        state_count=state_count,
        winner=winner,
        diagnostics=tuple(diagnostics),
    )
