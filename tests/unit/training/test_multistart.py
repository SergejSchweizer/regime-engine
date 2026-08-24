from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import nan

import pytest

from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.protocols import FitResult
from market_regime_engine.training.multistart import (
    MINIMUM_SUCCESS_RATE,
    MINIMUM_VALID_STARTS,
    MULTISTART_SEEDS,
    TRAIN_LOGLIK_TIE_ABS_TOLERANCE,
    run_multistart,
)


def artifact() -> GaussianHMMArtifact:
    return GaussianHMMArtifact(
        state_count=2,
        feature_order=("x",),
        start_probabilities=(0.5, 0.5),
        transition_matrix=((0.9, 0.1), (0.1, 0.9)),
        means=((-1.0,), (1.0,)),
        full_covariances=(((1.0,),), ((1.0,),)),
    )


def fit_result(seed: int, loglik: float, *, converged: bool = True) -> FitResult:
    return FitResult(
        artifact=artifact(),
        train_log_likelihood=loglik,
        converged=converged,
        iterations=17,
        seed=seed,
    )


class FakeAdapter:
    def __init__(self, outcomes: Mapping[int, FitResult | Exception]) -> None:
        self._outcomes = outcomes

    def fit(self, train_rows: object, state_count: int, seed: int) -> FitResult:
        del train_rows, state_count
        outcome = self._outcomes[seed]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def extract(self) -> GaussianHMMArtifact:
        raise AssertionError("unused")

    def reconstruct(self, artifact: GaussianHMMArtifact) -> None:
        raise AssertionError(f"unused: {artifact}")

    def causal_filter(
        self,
        rows: object,
        initial_filtered_probabilities: tuple[float, ...] | None = None,
    ) -> object:
        raise AssertionError(f"unused: {rows}, {initial_filtered_probabilities}")


def factory(outcomes: Mapping[int, FitResult | Exception]):
    return lambda: FakeAdapter(outcomes)


def test_exact_seed_set_gates_and_train_loglik_winner() -> None:
    assert MULTISTART_SEEDS == (11, 23, 37, 53, 71, 89, 107, 131)
    assert MINIMUM_VALID_STARTS == 6
    assert MINIMUM_SUCCESS_RATE == 0.75
    assert TRAIN_LOGLIK_TIE_ABS_TOLERANCE == 1e-12
    outcomes: dict[int, FitResult | Exception] = {
        seed: fit_result(seed, float(index)) for index, seed in enumerate(MULTISTART_SEEDS)
    }
    outcomes[107] = RuntimeError("numerical failure")
    outcomes[131] = fit_result(131, 100.0, converged=False)

    result = run_multistart([[0.0], [1.0]], state_count=2, adapter_factory=factory(outcomes))
    assert result.valid_start_count == 6
    assert result.success_rate == 0.75
    assert result.winner.seed == 89
    assert tuple(item.seed for item in result.diagnostics) == MULTISTART_SEEDS
    assert result.diagnostics[6].failure_reason == "RuntimeError: numerical failure"
    assert result.diagnostics[7].failure_reason == "not converged"


def test_numeric_tie_within_1e12_prefers_lower_seed() -> None:
    outcomes: dict[int, FitResult | Exception] = {
        seed: fit_result(seed, -100.0) for seed in MULTISTART_SEEDS
    }
    outcomes[11] = fit_result(11, 10.0)
    outcomes[23] = fit_result(23, 10.0 + 0.5e-12)
    result = run_multistart([[0.0]], state_count=2, adapter_factory=factory(outcomes))
    assert result.winner.seed == 11


def test_difference_above_tolerance_selects_higher_loglik() -> None:
    outcomes: dict[int, FitResult | Exception] = {
        seed: fit_result(seed, -100.0) for seed in MULTISTART_SEEDS
    }
    outcomes[11] = fit_result(11, 10.0)
    outcomes[23] = fit_result(23, 10.0 + 2e-12)
    result = run_multistart([[0.0]], state_count=2, adapter_factory=factory(outcomes))
    assert result.winner.seed == 23


def test_fewer_than_six_valid_starts_fails_with_failure_evidence() -> None:
    outcomes: dict[int, FitResult | Exception] = {
        seed: RuntimeError(f"fail-{seed}") for seed in MULTISTART_SEEDS
    }
    for seed in MULTISTART_SEEDS[:5]:
        outcomes[seed] = fit_result(seed, float(seed))
    with pytest.raises(ValueError, match="valid_starts=5/8") as exc_info:
        run_multistart([[0.0]], state_count=3, adapter_factory=factory(outcomes))
    assert "fail-89" in str(exc_info.value)


def test_invalid_result_paths_are_counted_as_failed_starts() -> None:
    outcomes: dict[int, FitResult | Exception] = {
        seed: fit_result(seed, float(seed)) for seed in MULTISTART_SEEDS
    }
    outcomes[11] = replace(outcomes[11], iterations=0)  # type: ignore[arg-type]
    outcomes[23] = replace(outcomes[23], train_log_likelihood=nan)  # type: ignore[arg-type]
    outcomes[37] = replace(outcomes[37], seed=999)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="valid_starts=5/8"):
        run_multistart([[0.0]], state_count=4, adapter_factory=factory(outcomes))


def test_invalid_state_count_fails_before_adapter_use() -> None:
    with pytest.raises(ValueError, match="K=2,3,4"):
        run_multistart([[0.0]], state_count=5, adapter_factory=factory({}))
